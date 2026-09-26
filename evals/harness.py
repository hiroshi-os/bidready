"""Score retrieval, extraction, eligibility and citation faithfulness on the gold set."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import time
from datetime import date
from pathlib import Path

from bidready.config import Settings
from bidready.db import make_session_factory
from bidready.decide import decide
from bidready.extract import extract_requirements
from bidready.parsing import chunk_document, parse_path
from bidready.pipeline import analyse, latest_run
from bidready.prompts import CUES_BY_VERSION, EXTRACT_V1, EXTRACT_V2
from bidready.rag import HybridIndex, build_embedder, build_reranker
from bidready.synthetic import profile_files
from bidready.textutil import quote_supported

ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = ROOT / "gold" / "manifest.json"
REPO = ROOT.parent


def load_manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def pdf_dir(manifest: dict | None = None) -> Path:
    manifest = manifest or load_manifest()
    return REPO / manifest.get("pdf_dir", "data/gold_pdfs")


def missing_pdfs(manifest: dict | None = None) -> list[str]:
    manifest = manifest or load_manifest()
    folder = pdf_dir(manifest)
    missing = []
    for tender in manifest["tenders"]:
        path = folder / tender["filename"]
        if not path.exists():
            missing.append(tender["filename"])
    return missing


def fetch(manifest: dict | None = None) -> list[Path]:
    import httpx

    manifest = manifest or load_manifest()
    folder = pdf_dir(manifest)
    folder.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for tender in manifest["tenders"]:
        dest = folder / tender["filename"]
        if dest.exists() and dest.stat().st_size > 1000:
            _check_hash(dest, tender.get("sha256"))
            written.append(dest)
            continue
        response = httpx.get(
            tender["source_url"],
            follow_redirects=True,
            timeout=60,
            headers={"User-Agent": "bidready-eval/0.1 (public tender fetch)"},
        )
        response.raise_for_status()
        if not response.content.startswith(b"%PDF"):
            raise RuntimeError(f"{tender['id']} did not return a PDF from {tender['source_url']}")
        dest.write_bytes(response.content)
        _check_hash(dest, tender.get("sha256"))
        written.append(dest)
    return written


def index_tender(path: Path, embedder, reranker):
    parsed = parse_path(path)
    chunks = chunk_document(parsed, document_id=path.stem, filename=path.name, role="tender")
    return parsed, chunks, HybridIndex(chunks, embedder, reranker)


def evaluate(
    *,
    embedding_provider: str,
    embedding_model: str,
    reranker_name: str,
    reranker_model: str,
    prompt_version: str = "v1",
) -> dict:
    manifest = load_manifest()
    missing = missing_pdfs(manifest)
    if missing:
        raise SystemExit("missing gold PDFs: " + ", ".join(missing) + ". Run: python -m evals.cli fetch")
    embedder = build_embedder(embedding_provider, embedding_model)
    reranker = build_reranker(reranker_name, reranker_model)
    folder = pdf_dir(manifest)

    retrieval_rows = []
    extraction_rows = []
    eligibility_rows = []
    prompt_rows = {version: [] for version in ("v1", "v2")}
    per_tender = []

    for tender in manifest["tenders"]:
        path = folder / tender["filename"]
        parsed, chunks, index = index_tender(path, embedder, reranker)
        retrieval_rows.extend(_retrieval_rows(tender, index))
        predictions = {version: extract_requirements(chunks, version) for version in ("v1", "v2")}
        for version, rows in predictions.items():
            prompt_rows[version].append(_score_extraction(tender, rows))
        extraction_rows.append(prompt_rows[prompt_version][-1])
        eligibility_rows.extend(_eligibility_rows(tender, chunks, embedder, reranker))
        per_tender.append(
            {
                "id": tender["id"],
                "pages": parsed.page_count,
                "ocr_pages": parsed.ocr_page_count,
                "chunks": len(chunks),
                "extracted_v1": prompt_rows["v1"][-1]["predicted"],
                "extracted_v2": prompt_rows["v2"][-1]["predicted"],
            }
        )

    retrieval = {
        mode: _aggregate_retrieval([row for row in retrieval_rows if row["mode"] == mode])
        for mode in ("vector", "hybrid", "hybrid_rerank")
    }
    return {
        "embedding_provider": embedding_provider,
        "embedding_model": getattr(embedder, "name", embedding_model),
        "reranker": getattr(reranker, "name", reranker_name),
        "retrieval": retrieval,
        "extraction": _micro(extraction_rows),
        "prompt_policy": {version: _micro(rows) for version, rows in prompt_rows.items()},
        "eligibility": _eligibility_summary(eligibility_rows),
        "tenders": per_tender,
        "prompt_text_sha256": {
            "v1": hashlib.sha256(EXTRACT_V1.encode()).hexdigest()[:12],
            "v2": hashlib.sha256(EXTRACT_V2.encode()).hexdigest()[:12],
        },
        "cue_counts": {version: len(cues) for version, cues in CUES_BY_VERSION.items()},
    }


def run_cases(settings: Settings, tender_ids: list[str] | None = None) -> dict:
    manifest = load_manifest()
    if missing_pdfs(manifest):
        raise SystemExit("missing gold PDFs. Run: python -m evals.cli fetch")
    selected = manifest["tenders"]
    if tender_ids:
        wanted = set(tender_ids)
        selected = [tender for tender in selected if tender["id"] in wanted]
    folder = pdf_dir(manifest)
    factory = make_session_factory(settings)
    session = factory()
    runs = []
    faithfulness_claims = 0
    faithfulness_supported = 0
    try:
        for tender in selected:
            path = folder / tender["filename"]
            print(f"case {tender['id']}", flush=True)
            started = time.perf_counter()
            case_id = analyse(
                settings,
                session,
                tender_files=[(tender["filename"], path.read_bytes())],
                profile_id="sample-civil",
                title=tender["title"],
            )
            wall = time.perf_counter() - started
            session.expire_all()
            run = latest_run(session, case_id)
            report = run.report_json or {}
            supported, total = _faithfulness(session, case_id, report)
            faithfulness_supported += supported
            faithfulness_claims += total
            extraction = _score_extraction(
                tender,
                [
                    {"text": row.get("text") or "", "quote": row.get("tender_quote") or ""}
                    for row in (report.get("matrix") or [])
                ],
            )
            runs.append(
                {
                    "tender_id": tender["id"],
                    "case_id": case_id,
                    "status": run.status,
                    "go_no_go": report.get("go_no_go"),
                    "summary": report.get("summary"),
                    "requirements": len(report.get("matrix") or []),
                    "decisions": _decision_counts(report.get("matrix") or []),
                    "risks": len(report.get("risks") or []),
                    "deadlines": len(report.get("deadlines") or []),
                    "letter": report.get("letter"),
                    "verification": report.get("verification"),
                    "trace": report.get("trace"),
                    "latency_seconds": round(wall, 3),
                    "latency_ms_recorded": run.latency_ms,
                    "prompt_tokens": run.prompt_tokens,
                    "completion_tokens": run.completion_tokens,
                    "faithfulness": {"supported": supported, "claims": total},
                    "extraction": extraction,
                    "claims": _claim_pairs(report),
                    "matrix_preview": [
                        {
                            "decision": row.get("decision"),
                            "kind": row.get("kind"),
                            "page": row.get("tender_page"),
                            "clause": row.get("tender_clause"),
                            "text": (row.get("text") or "")[:240],
                        }
                        for row in (report.get("matrix") or [])[:8]
                    ],
                }
            )
            print(
                f"case {tender['id']} done {wall:.1f}s tokens {run.prompt_tokens}/{run.completion_tokens} "
                f"reqs {len(report.get('matrix') or [])}",
                flush=True,
            )
            import gc

            gc.collect()
    finally:
        session.close()
    rate = (faithfulness_supported / faithfulness_claims) if faithfulness_claims else None
    extraction_rows = [row["extraction"] for row in runs]
    exhaustive_rows = [
        row["extraction"]
        for row, tender in zip(runs, selected, strict=False)
        if tender.get("exhaustive")
    ]
    return {
        "profile": "sample-civil",
        "profile_note": "Synthetic company. Not a real bidder.",
        "extraction": _micro(extraction_rows),
        "extraction_exhaustive": _micro(exhaustive_rows) if exhaustive_rows else None,
        "runs": runs,
        "faithfulness": {
            "supported": faithfulness_supported,
            "claims": faithfulness_claims,
            "rate": rate,
        },
        "latency_seconds": {
            "per_tender": [{"id": row["tender_id"], "seconds": row["latency_seconds"]} for row in runs],
            "total": round(sum(row["latency_seconds"] for row in runs), 3),
        },
    }


def probe_llm() -> dict:
    import httpx

    ollama = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    try:
        response = httpx.get(f"{ollama.rstrip('/')}/api/tags", timeout=2)
        ollama_up = response.status_code == 200
    except Exception:
        ollama_up = False
    return {
        "status": "not_measured",
        "reason": (
            "No LLM sample was scored. OPENAI_API_KEY is unset and Ollama did not respond "
            f"at {ollama}. The prompt-policy table is the deterministic v1/v2 cue ablation, "
            "not a sampled model comparison."
        ),
        "openai_key_set": bool(os.environ.get("OPENAI_API_KEY")),
        "ollama_reachable": ollama_up,
    }


def environment() -> dict:
    mem_gb = None
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                mem_gb = round(int(line.split()[1]) / 1024 / 1024, 2)
    except OSError:
        mem_gb = None
    return {
        "measured_on": date.today().isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "cpu_count": os.cpu_count(),
        "mem_total_gb": mem_gb,
    }


def _retrieval_rows(tender: dict, index: HybridIndex) -> list[dict]:
    rows = []
    for query in tender.get("retrieval_queries") or []:
        relevant = {chunk.id for chunk in index.order if _contains_all(chunk.text, query["match_all"])}
        if not relevant:
            raise RuntimeError(f"{tender['id']} query {query['id']} matched no chunk. The label or the parser drifted.")
        for mode in ("vector", "hybrid", "hybrid_rerank"):
            hits = index.search(query["query"], k=10, mode=mode)
            rows.append(
                {
                    "mode": mode,
                    "relevant": relevant,
                    "retrieved": [hit.id for hit in hits],
                }
            )
    return rows


def _aggregate_retrieval(rows: list[dict]) -> dict:
    recalls5 = []
    recalls10 = []
    reciprocal = []
    for row in rows:
        relevant = row["relevant"]
        retrieved = row["retrieved"]
        recalls5.append(len(set(retrieved[:5]) & relevant) / len(relevant))
        recalls10.append(len(set(retrieved[:10]) & relevant) / len(relevant))
        rank = 0
        for index, chunk_id in enumerate(retrieved, start=1):
            if chunk_id in relevant:
                rank = index
                break
        reciprocal.append(1.0 / rank if rank else 0.0)
    n = len(rows)
    return {
        "queries": n,
        "recall@5": _mean(recalls5),
        "recall@10": _mean(recalls10),
        "mrr": _mean(reciprocal),
    }


def _score_extraction(tender: dict, predictions: list[dict]) -> dict:
    gold = tender.get("requirements") or []
    matched_gold = set()
    matched_pred = set()
    for pred_index, prediction in enumerate(predictions):
        blob = f"{prediction.get('text', '')} {prediction.get('quote', '')}"
        for gold_index, item in enumerate(gold):
            if gold_index in matched_gold:
                continue
            if _contains_all(blob, item["match_all"]):
                matched_gold.add(gold_index)
                matched_pred.add(pred_index)
                break
    return {
        "gold": len(gold),
        "predicted": len(predictions),
        "matched": len(matched_gold),
    }


def _micro(rows: list[dict]) -> dict:
    gold = sum(row["gold"] for row in rows)
    predicted = sum(row["predicted"] for row in rows)
    matched = sum(row["matched"] for row in rows)
    precision = matched / predicted if predicted else None
    recall = matched / gold if gold else None
    f1 = None
    if precision is not None and recall is not None and (precision + recall):
        f1 = 2 * precision * recall / (precision + recall)
    return {
        "gold": gold,
        "predicted": predicted,
        "matched": matched,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _eligibility_rows(tender: dict, chunks, embedder, reranker) -> list[dict]:
    by_id = {item["id"]: item for item in tender.get("requirements") or []}
    company = _company_indexes(embedder, reranker)
    rows = []
    for label in tender.get("eligibility") or []:
        requirement = by_id[label["requirement_id"]]
        span, chunk = _span_for(chunks, requirement["match_all"])
        if span is None:
            raise RuntimeError(f"{tender['id']} {requirement['id']} anchor is not in any chunk")
        hits = company[label["profile"]].search(span, k=8, mode="hybrid_rerank")
        decision = decide(
            {
                "kind": requirement["kind"],
                "text": span,
                "quote": span,
                "chunk_id": chunk.id,
                "page": chunk.page_start,
                "clause": chunk.clause,
                "filename": chunk.filename,
            },
            hits,
        )
        rows.append(
            {
                "tender_id": tender["id"],
                "requirement_id": requirement["id"],
                "profile": label["profile"],
                "expected": label["expected"],
                "predicted": decision["decision"],
                "correct": decision["decision"] == label["expected"],
            }
        )
    return rows


def _eligibility_summary(rows: list[dict]) -> dict:
    confusion: dict[str, int] = {}
    for row in rows:
        key = f"{row['expected']}->{row['predicted']}"
        confusion[key] = confusion.get(key, 0) + 1
    correct = sum(1 for row in rows if row["correct"])
    return {
        "n": len(rows),
        "correct": correct,
        "accuracy": (correct / len(rows)) if rows else None,
        "confusion": confusion,
        "rows": rows,
        "profiles": "synthetic",
    }


_COMPANY_CACHE: dict = {}


def _company_indexes(embedder, reranker) -> dict:
    key = getattr(embedder, "name", "embed")
    if key in _COMPANY_CACHE:
        return _COMPANY_CACHE[key]
    built = {}
    for profile_id in ("sample-civil", "sample-security"):
        chunks = []
        for filename, data in profile_files(profile_id):
            path = Path("/tmp") / f"bidready-{profile_id}-{filename}"
            path.write_bytes(data)
            parsed = parse_path(path)
            chunks.extend(
                chunk_document(parsed, document_id=f"{profile_id}-{filename}", filename=filename, role="company")
            )
        built[profile_id] = HybridIndex(chunks, embedder, reranker)
    _COMPANY_CACHE[key] = built
    return built


def _span_for(chunks, phrases: list[str]):
    for chunk in chunks:
        if not _contains_all(chunk.text, phrases):
            continue
        lines = [line.strip() for line in chunk.text.splitlines() if line.strip()]
        if not lines:
            lines = [chunk.text.strip()]
        for size in range(1, len(lines) + 1):
            for start in range(0, len(lines) - size + 1):
                window = " ".join(lines[start : start + size])
                if _contains_all(window, phrases):
                    return window, chunk
        return chunk.text, chunk
    return None, None


def _claim_pairs(report: dict) -> list[dict]:
    """Pairs for semantic support: does the cited span entail the claim?"""
    pairs = []
    for row in report.get("matrix") or []:
        tender_quote = row.get("tender_quote") or ""
        text = row.get("text") or ""
        if tender_quote and text:
            pairs.append(
                {
                    "kind": "tender_requirement",
                    "premise": tender_quote,
                    "hypothesis": text,
                    "identical": _norm_eq(tender_quote, text),
                }
            )
        if row.get("evidence_quote") and row.get("rationale"):
            pairs.append(
                {
                    "kind": "evidence_decision",
                    "premise": row["evidence_quote"],
                    "hypothesis": row["rationale"],
                    "decision": row.get("decision"),
                    "identical": _norm_eq(row["evidence_quote"], row["rationale"]),
                }
            )
    for item in report.get("risks") or []:
        if item.get("quote"):
            pairs.append(
                {
                    "kind": "risk",
                    "premise": item["quote"],
                    "hypothesis": f"This tender sentence states a contractual risk or penalty ({item.get('severity') or 'unspecified'}).",
                    "identical": False,
                }
            )
    for item in report.get("deadlines") or []:
        if item.get("quote"):
            pairs.append(
                {
                    "kind": "deadline",
                    "premise": item["quote"],
                    "hypothesis": f"This tender sentence states the date or time of the {item.get('event') or 'event'}.",
                    "identical": False,
                }
            )
    return pairs


def _norm_eq(left: str, right: str) -> bool:
    return " ".join((left or "").split()).lower() == " ".join((right or "").split()).lower()


def score_nli(cases: dict, model_name: str = "cross-encoder/nli-MiniLM2-L6-H768") -> dict:
    """Entailment of each claim by its cited span. Label order is contradiction, entailment, neutral."""
    import numpy as np
    from sentence_transformers import CrossEncoder

    model = CrossEncoder(model_name)
    labelled = []
    for run in cases.get("runs") or []:
        for pair in run.get("claims") or []:
            labelled.append({**pair, "tender_id": run.get("tender_id")})
    if not labelled:
        return {"model": model_name, "n": 0, "supported": 0, "rate": None, "by_kind": {}}
    scores = np.asarray(
        model.predict([(pair["premise"][:1200], pair["hypothesis"][:400]) for pair in labelled]),
        dtype=np.float32,
    )
    names = ["contradiction", "entailment", "neutral"]
    counts = {name: 0 for name in names}
    by_kind: dict[str, dict] = {}
    sample = []
    nonidentical_n = 0
    nonidentical_entailment = 0
    for pair, row in zip(labelled, scores, strict=True):
        label = names[int(row.argmax())]
        counts[label] += 1
        kind = pair["kind"]
        bucket = by_kind.setdefault(kind, {"n": 0, "entailment": 0, "identical": 0})
        bucket["n"] += 1
        if label == "entailment":
            bucket["entailment"] += 1
        if pair.get("identical"):
            bucket["identical"] += 1
        else:
            nonidentical_n += 1
            if label == "entailment":
                nonidentical_entailment += 1
        if len(sample) < 16:
            sample.append(
                {
                    "kind": kind,
                    "label": label,
                    "identical": pair.get("identical"),
                    "premise": pair["premise"][:500],
                    "hypothesis": pair["hypothesis"][:500],
                    "decision": pair.get("decision"),
                }
            )
    n = len(labelled)
    del model
    import gc

    gc.collect()
    return {
        "model": model_name,
        "n": n,
        "entailment": counts["entailment"],
        "contradiction": counts["contradiction"],
        "neutral": counts["neutral"],
        "rate": counts["entailment"] / n if n else None,
        "nonidentical_n": nonidentical_n,
        "nonidentical_entailment": nonidentical_entailment,
        "headline_rate": (nonidentical_entailment / nonidentical_n) if nonidentical_n else None,
        "headline": "entailment among pairs whose premise and hypothesis are not the same string",
        "by_kind": by_kind,
        "sample": sample,
        "rule": "argmax over contradiction, entailment, neutral. Supported means entailment.",
    }


def llm_eligibility(
    llm,
    embedder,
    reranker,
    tender_ids: list[str] | None = None,
    *,
    show_rule: bool = False,
) -> dict:
    """Gold eligibility. show_rule=False is the model alone. show_rule=True adjudicates the rule pre-screen."""
    from bidready.graph import RunContext, _llm_decisions

    manifest = load_manifest()
    folder = pdf_dir(manifest)
    company = _company_indexes(embedder, reranker)
    rows = []
    selected = manifest["tenders"]
    if tender_ids:
        wanted = set(tender_ids)
        selected = [tender for tender in selected if tender["id"] in wanted]
    for tender in selected:
        _, chunks, index = index_tender(folder / tender["filename"], embedder, reranker)
        by_id = {item["id"]: item for item in tender.get("requirements") or []}
        for label in tender.get("eligibility") or []:
            requirement = by_id[label["requirement_id"]]
            span, chunk = _span_for(chunks, requirement["match_all"])
            if span is None:
                rows.append(
                    {
                        "tender_id": tender["id"],
                        "requirement_id": requirement["id"],
                        "profile": label["profile"],
                        "expected": label["expected"],
                        "predicted": "error",
                        "correct": False,
                    }
                )
                continue
            ctx = RunContext(
                llm=llm,
                tender_index=index,
                company_index=company[label["profile"]],
                allow_heuristic_fallback=False,
            )
            # A one-row batch uses the same decision code as the pipeline.
            decided, source = _llm_decisions(
                ctx,
                [
                    {
                        "kind": requirement["kind"],
                        "text": span,
                        "quote": span,
                        "chunk_id": chunk.id,
                        "page": chunk.page_start,
                        "clause": chunk.clause,
                        "filename": chunk.filename,
                    }
                ],
                "",
                None,
                show_rule=show_rule,
            )
            predicted = decided[0]["decision"] if decided else "error"
            print(
                f"eligibility {tender['id']} {requirement['id']} {label['profile']} "
                f"{label['expected']} -> {predicted} span={bool(decided and decided[0].get('evidence_quote'))}",
                flush=True,
            )
            rows.append(
                {
                    "tender_id": tender["id"],
                    "requirement_id": requirement["id"],
                    "profile": label["profile"],
                    "expected": label["expected"],
                    "predicted": predicted,
                    "correct": predicted == label["expected"],
                    "source": source,
                    "rule_decision": decided[0].get("rule_decision") if decided else None,
                    "cited": bool(decided and decided[0].get("evidence_quote")),
                    "rationale": (decided[0].get("rationale") if decided else "")[:300],
                }
            )
            del ctx
    summary = _eligibility_summary(rows)
    summary["judge"] = "hybrid" if show_rule else "llm"
    summary["cited"] = sum(1 for row in rows if row.get("cited"))
    return summary


def _faithfulness(session, case_id: str, report: dict) -> tuple[int, int]:
    from sqlalchemy import select

    from bidready.models import ChunkRecord

    texts = {
        row.id: row.text
        for row in session.scalars(select(ChunkRecord).where(ChunkRecord.case_id == case_id))
    }
    claims: list[tuple[str | None, str | None]] = []
    for row in report.get("matrix") or []:
        claims.append((row.get("tender_quote"), row.get("tender_chunk_id")))
        if row.get("evidence_quote"):
            claims.append((row.get("evidence_quote"), row.get("evidence_chunk_id")))
    for item in list(report.get("risks") or []) + list(report.get("deadlines") or []):
        claims.append((item.get("quote"), item.get("chunk_id")))
    for item in report.get("letter_quotes") or []:
        claims.append((item.get("quote"), item.get("chunk_id")))
    total = 0
    supported = 0
    for quote, chunk_id in claims:
        if not quote:
            continue
        total += 1
        if quote_supported(quote, texts.get(chunk_id or "", "")):
            supported += 1
    return supported, total


def _decision_counts(matrix: list[dict]) -> dict:
    counts = {"met": 0, "not_met": 0, "missing": 0, "unclear": 0}
    for row in matrix:
        key = row.get("decision")
        if key in counts:
            counts[key] += 1
    return counts


def _contains_all(text: str, phrases: list[str]) -> bool:
    lowered = text.lower()
    return all(phrase.lower() in lowered for phrase in phrases)


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _check_hash(path: Path, expected: str | None) -> None:
    if not expected:
        return
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected:
        raise RuntimeError(f"sha256 mismatch for {path.name}: {digest} != {expected}")
