"""LangGraph: planner, extractor, eligibility, risk, drafter, verifier, with a bounded retry."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from operator import add
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, StateGraph

from bidready.decide import decide, estimated_cost_inr
from bidready.draft import draft_letter
from bidready.extract import candidate_chunks, extract_requirements, extract_risks_and_deadlines
from bidready.llm import LLMClient
from bidready.parsing import Chunk
from bidready.prompts import (
    DRAFT_SYSTEM,
    ELIGIBILITY_SCHEMA,
    ELIGIBILITY_SYSTEM,
    EXTRACT_SCHEMA,
    PLANNER_QUERIES,
    PROMPT_BY_VERSION,
    RISK_SYSTEM,
)
from bidready.rag import HybridIndex
from bidready.textutil import amounts_in, clause_from_text, quote_supported, sentences
from bidready.verify import first_stage, strip_unsupported, unsupported_claims

MAX_STAGE_RETRIES = 2
_OBLIGATION = re.compile(r"\b(shall|should|must|required|bidder|emd|earnest)\b", re.IGNORECASE)
_WATERMARK = re.compile(r"synthetic|fictional|udin", re.IGNORECASE)
_FACT = re.compile(
    r"turnover|gst|pan\b|blacklist|solvency|manpower|psara|iso|licen[cs]e|loss|work|\brs\.?\b|₹|\binr\b|\d",
    re.IGNORECASE,
)


class GraphState(TypedDict, total=False):
    plan: dict
    requirements: list
    matrix: list
    risks: list
    deadlines: list
    letter: str
    letter_quotes: list
    verification: dict
    trace: Annotated[list[dict], add]
    retries: dict
    feedback: str
    next_node: str


@dataclass
class RunContext:
    llm: LLMClient
    tender_index: HybridIndex
    company_index: HybridIndex
    prompt_version: str = "v1"
    title: str = ""
    allow_heuristic_fallback: bool = True
    hooks: dict[str, Any] = field(default_factory=dict)


def build_graph(ctx: RunContext):
    def planner(state: GraphState) -> dict:
        started = time.perf_counter()
        sections: list[str] = []
        for chunk in ctx.tender_index.chunks.values():
            if chunk.section and chunk.section not in sections:
                sections.append(chunk.section)
        queries = list(PLANNER_QUERIES)
        for section in sections[:8]:
            queries.append(section)
        plan = {
            "queries": queries,
            "sections": sections[:40],
            "prompt_version": ctx.prompt_version,
            "provider": ctx.llm.provider,
            "model": ctx.llm.model,
        }
        return {"plan": plan, "trace": [_event("planner", started, f"{len(queries)} queries, {len(sections)} sections")]}

    def extractor(state: GraphState) -> dict:
        started = time.perf_counter()
        policy = "v2" if state.get("feedback") and state.get("next_node") == "extractor" else ctx.prompt_version
        # The heuristic reads every tender chunk. Retrieval still plans the run and
        # feeds the LLM path, which only sees a short window of chunks.
        retrieved = chunks_for_llm(ctx.tender_index, (state.get("plan") or {}).get("queries"))
        source = "heuristic"
        if ctx.hooks.get("extract"):
            requirements = ctx.hooks["extract"](state, retrieved)
            source = "hook"
        elif ctx.llm.provider == "mock":
            requirements = extract_requirements(list(ctx.tender_index.order), policy)
        else:
            requirements, source = _llm_requirements(ctx, retrieved, policy, state.get("feedback") or "")
        return {
            "requirements": requirements,
            "trace": [_event("extractor", started, f"{len(requirements)} requirements via {source} ({policy})")],
        }

    def eligibility(state: GraphState) -> dict:
        started = time.perf_counter()
        cost = estimated_cost_inr(list(ctx.tender_index.chunks.values()))
        rows = []
        source = "rules"
        requirements = list(state.get("requirements") or [])
        if ctx.llm.provider == "mock" or ctx.hooks.get("force_rules"):
            for requirement in requirements:
                hits = ctx.company_index.search(requirement.get("text") or "", k=8, mode="hybrid_rerank")
                rows.append(decide(requirement, hits, estimated_cost_inr=cost))
        else:
            rows, source = _llm_decisions(ctx, requirements, state.get("feedback") or "", cost)
        return {"matrix": rows, "trace": [_event("eligibility", started, f"{len(rows)} rows via {source}")]}

    def risk(state: GraphState) -> dict:
        started = time.perf_counter()
        chunks = _risk_chunks(ctx, state)
        source = "heuristic"
        if ctx.llm.provider == "mock":
            risks, deadlines = extract_risks_and_deadlines(chunks)
        else:
            try:
                risks, deadlines = _llm_risks(ctx, chunks, state.get("feedback") or "")
                source = "llm"
                if not risks and not deadlines and ctx.allow_heuristic_fallback:
                    risks, deadlines = extract_risks_and_deadlines(chunks)
                    source = "llm_empty_fallback_heuristic"
            except Exception as exc:
                if ctx.allow_heuristic_fallback:
                    risks, deadlines = extract_risks_and_deadlines(chunks)
                    source = f"fallback:{type(exc).__name__}"
                else:
                    risks, deadlines = [], []
                    source = f"llm_error:{type(exc).__name__}"
        return {
            "risks": risks,
            "deadlines": deadlines,
            "trace": [_event("risk", started, f"{len(risks)} risks, {len(deadlines)} deadlines via {source}")],
        }

    def drafter(state: GraphState) -> dict:
        started = time.perf_counter()
        letter, quotes = draft_letter(
            title=ctx.title,
            matrix=list(state.get("matrix") or []),
            deadlines=list(state.get("deadlines") or []),
        )
        source = "template"
        if ctx.llm.provider != "mock" and quotes:
            letter, source = _llm_questions(ctx, letter, quotes, state.get("feedback") or "")
        return {
            "letter": letter,
            "letter_quotes": quotes,
            "trace": [_event("drafter", started, f"{len(quotes)} cited queries via {source}")],
        }

    def verifier(state: GraphState) -> dict:
        started = time.perf_counter()
        failures = unsupported_claims(state, ctx.tender_index, ctx.company_index)
        retries = dict(state.get("retries") or {})
        if not failures:
            return {
                "verification": {"passed": True, "failures": [], "stripped": [], "exhausted": False},
                "next_node": "end",
                "trace": [_event("verifier", started, "passed")],
            }
        stage = first_stage(failures)
        retries[stage] = retries.get(stage, 0) + 1
        detail = f"{len(failures)} unsupported claim(s); first stage {stage}"
        if retries[stage] > MAX_STAGE_RETRIES or sum(retries.values()) > 6:
            cleaned = strip_unsupported(state, failures, title=ctx.title)
            return {
                **cleaned,
                "retries": retries,
                "feedback": detail,
                "verification": {
                    "passed": True,
                    "exhausted": True,
                    "failures": failures,
                    "stripped": failures,
                },
                "next_node": "end",
                "trace": [_event("verifier", started, detail + "; stripped after retries")],
            }
        return {
            "retries": retries,
            "feedback": detail,
            "verification": {"passed": False, "exhausted": False, "failures": failures, "stripped": []},
            "next_node": stage,
            "trace": [_event("verifier", started, detail + "; retrying")],
        }

    graph = StateGraph(GraphState)
    graph.add_node("planner", planner)
    graph.add_node("extractor", extractor)
    graph.add_node("eligibility", eligibility)
    graph.add_node("risk", risk)
    graph.add_node("drafter", drafter)
    graph.add_node("verifier", verifier)
    graph.set_entry_point("planner")
    graph.add_edge("planner", "extractor")
    graph.add_edge("extractor", "eligibility")
    graph.add_edge("eligibility", "risk")
    graph.add_edge("risk", "drafter")
    graph.add_edge("drafter", "verifier")
    graph.add_conditional_edges(
        "verifier",
        _route,
        {
            "extractor": "extractor",
            "eligibility": "eligibility",
            "risk": "risk",
            "drafter": "drafter",
            "end": END,
        },
    )
    return graph.compile()


def initial_state() -> dict:
    return {
        "trace": [],
        "retries": {},
        "feedback": "",
        "requirements": [],
        "matrix": [],
        "risks": [],
        "deadlines": [],
        "letter": "",
        "letter_quotes": [],
    }


def _route(state: GraphState) -> str:
    verification = state.get("verification") or {}
    if verification.get("passed"):
        return "end"
    nxt = state.get("next_node") or "end"
    if nxt not in {"extractor", "eligibility", "risk", "drafter"}:
        return "end"
    return nxt


def _event(node: str, started: float, detail: str) -> dict:
    return {"node": node, "ms": int((time.perf_counter() - started) * 1000), "detail": detail}


def _risk_chunks(ctx: RunContext, state: GraphState) -> list[Chunk]:
    # Mock mode still scans a short pack in full. A local model only reads a
    # retrieved window, otherwise the first pages crowd out penalties and dates.
    if ctx.llm.provider == "mock" and len(ctx.tender_index.chunks) <= 80:
        return list(ctx.tender_index.chunks.values())
    queries = ["penalty liquidated damages earnest money forfeiture", "pre-bid meeting bid submission opening date"]
    if ctx.llm.provider == "mock":
        return candidate_chunks(ctx.tender_index, queries, k=10)
    chosen: list[Chunk] = []
    seen: set[str] = set()
    for query in queries:
        for chunk in ctx.tender_index.search(query, k=6, mode="hybrid_rerank"):
            if chunk.id in seen:
                continue
            seen.add(chunk.id)
            chosen.append(chunk)
            if len(chosen) >= 8:
                return chosen
    return chosen


def _format_chunks(chunks: list[Chunk], limit: int = 12) -> str:
    blocks = []
    for chunk in chunks[:limit]:
        blocks.append(
            f"[chunk_id={chunk.id} page={chunk.page_start} clause={chunk.clause or '-'} file={chunk.filename}]\n"
            f"{chunk.text[:1400]}"
        )
    return "\n\n".join(blocks)


LLM_CHUNK_LIMIT = 24
LLM_BATCH_SIZE = 2
_SECTION_HINT = re.compile(
    r"eligib|qualif|earnest|\bemd\b|turnover|solvency|criteria|experience|blacklist|document",
    re.IGNORECASE,
)


def chunks_for_llm(index: HybridIndex, queries: list[str] | None = None, limit: int = LLM_CHUNK_LIMIT) -> list[Chunk]:
    """Chunks the local model reads, grouped by section after a retrieval pass.

    Retrieval fills the first slots. Sections whose heading looks like eligibility
    then contribute a few chunks each, so one retrieved page cannot crowd out the
    rest of the pack. The cap is `limit`, not the whole tender.
    """
    ranked: list[Chunk] = []
    seen: set[str] = set()

    def take(chunk: Chunk) -> None:
        if chunk.id in seen or len(ranked) >= limit:
            return
        seen.add(chunk.id)
        ranked.append(chunk)

    for query in (queries or PLANNER_QUERIES)[:6]:
        for chunk in index.search(query, k=3, mode="hybrid_rerank"):
            take(chunk)
            if len(ranked) >= 8:
                break
        if len(ranked) >= 8:
            break

    groups: dict[str, list[Chunk]] = {}
    for chunk in index.order:
        key = (chunk.section or "").strip() or f"page-{chunk.page_start}"
        groups.setdefault(key, []).append(chunk)

    def section_score(key: str, chunks: list[Chunk]) -> int:
        blob = f"{key} {chunks[0].text[:240]}"
        score = len(_SECTION_HINT.findall(blob))
        if any(_OBLIGATION.search(chunk.text) for chunk in chunks[:3]):
            score += 1
        return score

    ordered = sorted(groups.items(), key=lambda item: section_score(item[0], item[1]), reverse=True)
    for key, chunks in ordered:
        if len(ranked) >= limit:
            break
        if section_score(key, chunks) <= 0:
            continue
        added = 0
        for chunk in chunks:
            if chunk.id in seen:
                continue
            if not _OBLIGATION.search(chunk.text) and section_score(key, chunks) < 2:
                continue
            take(chunk)
            added += 1
            if added >= 3 or len(ranked) >= limit:
                break
    return ranked


def _llm_requirements(ctx: RunContext, chunks: list[Chunk], policy: str, feedback: str) -> tuple[list[dict], str]:
    """Map each batch of chunks to span ids, then dedupe. The quote is the span, not the restatement."""
    system = PROMPT_BY_VERSION.get(policy, PROMPT_BY_VERSION["v1"])
    rows: list[dict] = []
    errors = 0
    batches = 0
    for start in range(0, len(chunks), LLM_BATCH_SIZE):
        batch = chunks[start : start + LLM_BATCH_SIZE]
        menu = _span_menu(batch, prefix="S", per_chunk=8, limit=16)
        if not menu:
            continue
        batches += 1
        user = "Span menu:\n" + _format_menu(menu)
        if feedback:
            user += "\n\nVerifier feedback: " + feedback
        try:
            payload = ctx.llm.complete_json(system=system, user=user, schema=EXTRACT_SCHEMA)
        except Exception:
            errors += 1
            continue
        for item in payload.get("requirements") or []:
            built = _requirement_from_item(ctx, item, menu)
            if built is not None:
                rows.append(built)
    rows = _dedupe_requirements(rows)
    if rows:
        return rows, f"llm_sections:{len(chunks)}_batches:{batches}"
    if ctx.allow_heuristic_fallback:
        return extract_requirements(chunks, policy), f"llm_empty_fallback_heuristic:errors={errors}"
    return [], f"llm_empty:errors={errors}"


def _chunk_for_quote(ctx: RunContext, chunk_id: str, quote: str) -> Chunk | None:
    if len(" ".join(quote.split())) < 20:
        return None
    preferred = ctx.tender_index.chunks.get(chunk_id)
    if preferred is not None and quote_supported(quote, preferred.text):
        return preferred
    for chunk in ctx.tender_index.order:
        if quote_supported(quote, chunk.text):
            return chunk
    return None


def _llm_decisions(
    ctx: RunContext,
    requirements: list[dict],
    feedback: str,
    cost: int | None,
    *,
    show_rule: bool = False,
) -> tuple[list[dict], str]:
    """Decide from a menu of company-evidence spans.

    The decision is kept when the model names one. A span id that is not on the
    menu drops the citation, not the decision. show_rule adds the rule checker's
    proposal so the model can adjudicate it; that path is reported separately.
    """
    rows: list[dict] = []
    source = "hybrid" if show_rule else "llm"
    batch = 2
    for start in range(0, len(requirements), batch):
        group = requirements[start : start + batch]
        prepared: list[tuple[dict, list[Chunk], list[dict]]] = []
        blocks = []
        for index, requirement in enumerate(group, start=1):
            query = requirement.get("quote") or requirement.get("text") or ""
            hits = ctx.company_index.search(query, k=4, mode="hybrid_rerank")
            menu = _span_menu(hits, prefix="E", per_chunk=4, limit=6, drop_watermarks=True)
            prepared.append((requirement, hits, menu))
            ident = f"R{index}"
            requirement_text = requirement.get("text") or requirement.get("quote") or ""
            block = (
                f"REQUIREMENT {ident}\n"
                f"{requirement_text}{_amount_hint(requirement_text)}\n"
                f"Evidence spans (pick one id, or NONE):\n{_format_menu(menu, with_amounts=True) or 'NONE'}"
            )
            if show_rule:
                proposal = decide(requirement, hits, estimated_cost_inr=cost)
                cited = proposal.get("evidence_quote") or ""
                block += (
                    f"\nRule checker pre-screen: {proposal['decision']}."
                    f" Cited: {cited[:240]}\nAdjudicate this proposal. Agree or override."
                )
            blocks.append(block)
        user = "Decide each requirement.\n\n" + "\n\n".join(blocks)
        if feedback:
            user += "\n\nVerifier feedback: " + feedback
        try:
            payload = ctx.llm.complete_json(system=ELIGIBILITY_SYSTEM, user=user, schema=ELIGIBILITY_SCHEMA)
        except Exception as exc:
            if ctx.allow_heuristic_fallback:
                for requirement, hits, _menu in prepared:
                    rows.append(decide(requirement, hits, estimated_cost_inr=cost))
                source = f"fallback:{type(exc).__name__}"
                continue
            for requirement in group:
                row = decide(requirement, [], estimated_cost_inr=cost)
                row["decision"] = "unclear"
                row["rationale"] = f"The model call failed ({type(exc).__name__}). No rule fallback was applied."
                row["evidence_quote"] = None
                row["evidence_chunk_id"] = None
                rows.append(row)
            source = "llm_error"
            continue
        if "decisions" not in payload and "decision" in payload:
            payload = {"decisions": [payload]}
        parsed = [item for item in payload.get("decisions") or [] if isinstance(item, dict)]
        by_id = {_norm_id(item.get("id")): item for item in parsed}
        for index, (requirement, hits, menu) in enumerate(prepared, start=1):
            item = _pick_decision(by_id, parsed, index, len(prepared))
            rows.append(_decision_from_menu(requirement, item, hits, menu, cost))
    return rows, source


def _span_menu(
    chunks: list[Chunk],
    *,
    prefix: str,
    per_chunk: int,
    limit: int,
    drop_watermarks: bool = False,
) -> list[dict]:
    menu: list[dict] = []
    for chunk in chunks:
        added = 0
        parts = [part for part in sentences(chunk.text) if _OBLIGATION.search(part)] or sentences(chunk.text)
        for part in parts:
            if len(menu) >= limit or added >= per_chunk:
                break
            if len(part) < 40 or not quote_supported(part, chunk.text):
                continue
            if drop_watermarks and _watermark_only(part):
                continue
            menu.append({"id": f"{prefix}{len(menu) + 1}", "chunk": chunk, "text": part})
            added += 1
    return menu


def _watermark_only(text: str) -> bool:
    """Evaluation banners that say the file is synthetic and state no fact."""
    return bool(_WATERMARK.search(text)) and not bool(_FACT.search(text))


def _norm_id(raw: object) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(raw or "").upper())


def _pick_decision(by_id: dict[str, dict], parsed: list[dict], index: int, batch_size: int) -> dict:
    """Match R1 / 1. Fall back to position when the batch lines up and the id does not."""
    for key in (f"R{index}", str(index)):
        hit = by_id.get(_norm_id(key))
        if hit:
            return hit
    if len(parsed) == batch_size and 0 <= index - 1 < len(parsed):
        return parsed[index - 1]
    return {}


def _format_menu(menu: list[dict], *, with_amounts: bool = False) -> str:
    lines = []
    for item in menu:
        chunk = item["chunk"]
        hint = _amount_hint(item["text"]) if with_amounts else ""
        lines.append(f"{item['id']} chunk_id={chunk.id} page={chunk.page_start}: {item['text'][:420]}{hint}")
    return "\n".join(lines)


def _amount_hint(text: str) -> str:
    amounts = amounts_in(text)
    if not amounts:
        return ""
    shown = ", ".join(f"{amount} INR" for amount in amounts[:4])
    return f" [amounts: {shown}]"


def _span_by_id(menu: list[dict], raw: object) -> dict | None:
    token = re.sub(r"[^A-Z0-9]", "", str(raw or "").upper())
    if not token or token == "NONE":
        return None
    for item in menu:
        if item["id"].upper() == token:
            return item
    return None


def _requirement_from_item(ctx: RunContext, item: object, menu: list[dict]) -> dict | None:
    if not isinstance(item, dict):
        return None
    span = _span_by_id(menu, item.get("span_id"))
    quote = span["text"] if span else ""
    chunk = span["chunk"] if span else None
    if chunk is None:
        quote = str(item.get("quote") or "").strip()
        chunk = _chunk_for_quote(ctx, str(item.get("chunk_id") or ""), quote)
        if chunk is None:
            return None
    text = " ".join(str(item.get("text") or "").split())
    if len(text) < 20:
        text = quote
    return {
        "kind": item.get("kind") or "eligibility",
        "text": text,
        "quote": quote,
        "chunk_id": chunk.id,
        "page": chunk.page_start,
        "clause": clause_from_text(quote) or chunk.clause,
        "section": chunk.section,
        "filename": chunk.filename,
    }


def _dedupe_requirements(rows: list[dict]) -> list[dict]:
    kept: list[dict] = []
    seen: list[str] = []
    for row in rows:
        key = " ".join((row.get("quote") or "").lower().split())[:240]
        if not key:
            continue
        if any(key in previous or previous in key for previous in seen):
            continue
        seen.append(key)
        kept.append(row)
    return kept


def _decision_from_menu(
    requirement: dict,
    item: dict,
    hits: list[Chunk],
    menu: list[dict],
    cost: int | None,
) -> dict:
    """Keep the model's decision. Attach a citation only when the span id is on the menu."""
    row = decide(requirement, hits, estimated_cost_inr=cost)
    rule_decision = row["decision"]
    decision = str(item.get("decision") or "unclear").strip().lower().replace(" ", "_").replace("-", "_")
    if decision not in {"met", "not_met", "missing", "unclear"}:
        decision = "unclear"
    span = _span_by_id(menu, item.get("evidence_span"))
    row["rule_decision"] = rule_decision
    row["decision"] = decision
    row["obligation"] = item.get("obligation") or row["obligation"]
    row["rationale"] = str(item.get("rationale") or row["rationale"])
    if span is not None:
        chunk = span["chunk"]
        row.update(
            {
                "evidence_quote": span["text"],
                "evidence_chunk_id": chunk.id,
                "evidence_filename": chunk.filename,
                "evidence_page": chunk.page_start,
                "evidence_clause": chunk.clause,
            }
        )
    else:
        row["evidence_quote"] = None
        row["evidence_chunk_id"] = None
        row["evidence_filename"] = None
        row["evidence_page"] = None
        row["evidence_clause"] = None
    return row


def _llm_risks(ctx: RunContext, chunks: list[Chunk], feedback: str) -> tuple[list[dict], list[dict]]:
    user = _format_chunks(chunks)
    if feedback:
        user += "\n\nVerifier feedback: " + feedback
    payload = ctx.llm.complete_json(system=RISK_SYSTEM, user=user)
    risks = []
    deadlines = []
    for item in payload.get("risks") or []:
        mapped = _map_llm_span(ctx, item)
        if mapped:
            mapped["severity"] = item.get("severity") or "medium"
            risks.append(mapped)
    for item in payload.get("deadlines") or []:
        mapped = _map_llm_span(ctx, item)
        if mapped:
            mapped["event"] = item.get("event") or "dated event"
            deadlines.append(mapped)
    return risks, deadlines


def _map_llm_span(ctx: RunContext, item: dict) -> dict | None:
    if not isinstance(item, dict):
        return None
    chunk_id = str(item.get("chunk_id") or "")
    chunk = ctx.tender_index.chunks.get(chunk_id)
    quote = str(item.get("quote") or "").strip()
    if not quote or chunk is None or not quote_supported(quote, chunk.text):
        return None
    return {
        "quote": quote,
        "chunk_id": chunk.id,
        "page": chunk.page_start,
        "clause": clause_from_text(quote) or chunk.clause,
        "section": chunk.section,
        "filename": chunk.filename,
    }


def _llm_questions(ctx: RunContext, letter: str, quotes: list[dict], feedback: str) -> tuple[str, str]:
    # The template already contains the only quotes the verifier will accept.
    # The model may rewrite question sentences; quotes stay under code control.
    try:
        payload = ctx.llm.complete_json(
            system=DRAFT_SYSTEM,
            user="Letter draft follows. Do not add new factual claims.\n\n" + letter[:6000],
        )
    except Exception:
        return letter, "template_fallback"
    question = str(payload.get("question") or "").strip()
    if not question:
        return letter, "template"
    note = "\nModel note on the queries: " + question + "\n"
    return letter + note, "template_plus_llm_note"
