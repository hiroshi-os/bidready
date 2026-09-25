"""LangGraph: planner, extractor, eligibility, risk, drafter, verifier, with a bounded retry."""

from __future__ import annotations

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
    ELIGIBILITY_SYSTEM,
    PLANNER_QUERIES,
    PROMPT_BY_VERSION,
    RISK_SYSTEM,
)
from bidready.rag import HybridIndex
from bidready.textutil import clause_from_text, quote_supported
from bidready.verify import first_stage, strip_unsupported, unsupported_claims

MAX_STAGE_RETRIES = 2


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
        retrieved = candidate_chunks(ctx.tender_index, (state.get("plan") or {}).get("queries"))
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
        for requirement in state.get("requirements") or []:
            hits = ctx.company_index.search(requirement.get("text") or "", k=8, mode="hybrid_rerank")
            if ctx.llm.provider == "mock" or ctx.hooks.get("force_rules"):
                rows.append(decide(requirement, hits, estimated_cost_inr=cost))
            else:
                row, row_source = _llm_decision(ctx, requirement, hits, state.get("feedback") or "", cost)
                rows.append(row)
                source = row_source
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
                if not risks and not deadlines:
                    risks, deadlines = extract_risks_and_deadlines(chunks)
                    source = "llm_empty_fallback_heuristic"
            except Exception as exc:
                risks, deadlines = extract_risks_and_deadlines(chunks)
                source = f"fallback:{type(exc).__name__}"
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
    if len(ctx.tender_index.chunks) <= 80:
        return list(ctx.tender_index.chunks.values())
    queries = ["penalty liquidated damages earnest money forfeiture", "pre-bid meeting bid submission opening date"]
    return candidate_chunks(ctx.tender_index, queries, k=10)


def _format_chunks(chunks: list[Chunk], limit: int = 12) -> str:
    blocks = []
    for chunk in chunks[:limit]:
        blocks.append(
            f"[chunk_id={chunk.id} page={chunk.page_start} clause={chunk.clause or '-'} file={chunk.filename}]\n"
            f"{chunk.text[:1400]}"
        )
    return "\n\n".join(blocks)


def _llm_requirements(ctx: RunContext, chunks: list[Chunk], policy: str, feedback: str) -> tuple[list[dict], str]:
    system = PROMPT_BY_VERSION.get(policy, PROMPT_BY_VERSION["v1"])
    user = _format_chunks(chunks)
    if feedback:
        user += "\n\nVerifier feedback: " + feedback
    try:
        payload = ctx.llm.complete_json(system=system, user=user)
    except Exception as exc:
        return extract_requirements(chunks, policy), f"fallback:{type(exc).__name__}"
    rows = []
    for item in payload.get("requirements") or []:
        if not isinstance(item, dict):
            continue
        chunk_id = str(item.get("chunk_id") or "")
        chunk = ctx.tender_index.chunks.get(chunk_id)
        quote = str(item.get("quote") or "").strip()
        if not quote:
            continue
        rows.append(
            {
                "kind": item.get("kind") or "eligibility",
                "text": quote,
                "quote": quote,
                "chunk_id": chunk_id,
                "page": chunk.page_start if chunk else None,
                "clause": (clause_from_text(quote) if quote else None) or (chunk.clause if chunk else None),
                "section": chunk.section if chunk else None,
                "filename": chunk.filename if chunk else None,
            }
        )
    if not rows:
        return extract_requirements(chunks, policy), "llm_empty_fallback_heuristic"
    return rows[:40], "llm"


def _llm_decision(ctx: RunContext, requirement: dict, hits: list[Chunk], feedback: str, cost: int | None) -> tuple[dict, str]:
    user = (
        f"Requirement: {requirement.get('text')}\n"
        f"Tender quote chunk_id={requirement.get('chunk_id')} page={requirement.get('page')}\n"
        f"{requirement.get('quote')}\n\nCompany evidence:\n{_format_chunks(hits, limit=6)}"
    )
    if feedback:
        user += "\n\nVerifier feedback: " + feedback
    try:
        payload = ctx.llm.complete_json(system=ELIGIBILITY_SYSTEM, user=user)
    except Exception:
        return decide(requirement, hits, estimated_cost_inr=cost), "fallback_rules"
    decision = str(payload.get("decision") or "unclear")
    if decision not in {"met", "not_met", "missing", "unclear"}:
        decision = "unclear"
    evidence_id = str(payload.get("evidence_chunk_id") or "")
    evidence = ctx.company_index.chunks.get(evidence_id) or ctx.tender_index.chunks.get(evidence_id)
    quote = str(payload.get("evidence_quote") or "").strip()
    if decision in {"met", "not_met"} and (evidence is None or not quote_supported(quote, evidence.text)):
        # Keep the bad citation so the verifier can reject it.
        pass
    row = decide(requirement, hits, estimated_cost_inr=cost)
    # The model's decision is kept, but a rules decision is stored when the model cites nothing
    # usable for met/not_met. The quote fields always come from the model when present so the
    # verifier, not the rules, decides whether the model was faithful.
    if quote and evidence is not None:
        row.update(
            {
                "decision": decision,
                "obligation": payload.get("obligation") or row["obligation"],
                "rationale": str(payload.get("rationale") or row["rationale"]),
                "evidence_quote": quote,
                "evidence_chunk_id": evidence.id,
                "evidence_filename": evidence.filename,
                "evidence_page": evidence.page_start,
                "evidence_clause": evidence.clause,
            }
        )
        return row, "llm"
    if decision in {"missing", "unclear"}:
        row["decision"] = decision
        row["rationale"] = str(payload.get("rationale") or row["rationale"])
        row["evidence_quote"] = None
        row["evidence_chunk_id"] = None
        return row, "llm"
    return row, "llm_uncited_kept_rules"


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
    if not quote or chunk is None:
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
