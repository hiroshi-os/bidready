"""Turn an uploaded tender pack and company documents into a cited report."""

from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from bidready.config import Settings
from bidready.decide import go_no_go
from bidready.draft import DISCLAIMER
from bidready.graph import RunContext, build_graph, initial_state
from bidready.llm import build_llm
from bidready.models import Case, ChunkRecord, Document, EvidenceLink, Requirement, Run, utcnow
from bidready.notify import notify_run
from bidready.parsing import DocumentParseError, chunk_document, parse_path
from bidready.rag import HybridIndex, build_embedder, build_reranker
from bidready.store import DocumentStore, build_store
from bidready.synthetic import get_profile, profile_files

logger = logging.getLogger(__name__)

SUMMARY_LIMIT = 12


def analyse(
    settings: Settings,
    session: Session,
    *,
    tender_files: list[tuple[str, bytes]],
    company_files: list[tuple[str, bytes]] | None = None,
    profile_id: str | None = None,
    title: str | None = None,
) -> str:
    company_files = list(company_files or [])
    if profile_id:
        company_files.extend(profile_files(profile_id))
    if not tender_files:
        raise ValueError("upload at least one tender file")

    case_id = uuid.uuid4().hex
    case = Case(
        id=case_id,
        title=title or _title_from(tender_files, profile_id),
        status="running",
        profile_id=profile_id,
    )
    session.add(case)
    session.flush()
    store = build_store(settings)
    try:
        _run(settings, session, store, case, tender_files, company_files)
    except Exception as exc:
        logger.exception("case %s failed", case_id)
        case.status = "failed"
        case.error = f"{type(exc).__name__}: {exc}"
        run = latest_run(session, case_id)
        if run is not None and run.status == "running":
            run.status = "failed"
            run.error = case.error
            run.finished_at = utcnow()
        session.commit()
    return case_id


def _run(
    settings: Settings,
    session: Session,
    store: DocumentStore,
    case: Case,
    tender_files: list[tuple[str, bytes]],
    company_files: list[tuple[str, bytes]],
) -> None:
    started = time.perf_counter()
    llm = build_llm(settings)
    run = Run(
        id=uuid.uuid4().hex,
        case_id=case.id,
        status="running",
        provider=llm.provider,
        model=llm.model,
        embedding_provider=settings.embedding_provider,
        reranker=settings.reranker,
        prompt_version=settings.prompt_version,
    )
    session.add(run)
    session.flush()

    tender_chunks = []
    company_chunks = []
    for role, files in (("tender", tender_files), ("company", company_files)):
        for filename, data in files:
            document_id = uuid.uuid4().hex
            uri = store.put(case.id, filename, data)
            path = store.path_for(uri)
            try:
                parsed = parse_path(path)
            except DocumentParseError:
                raise
            record = Document(
                id=document_id,
                case_id=case.id,
                role=role,
                filename=Path(filename).name,
                storage_uri=uri,
                page_count=parsed.page_count,
                ocr_page_count=parsed.ocr_page_count,
            )
            session.add(record)
            chunks = chunk_document(parsed, document_id=document_id, filename=record.filename, role=role)
            for chunk in chunks:
                session.add(
                    ChunkRecord(
                        id=chunk.id,
                        case_id=case.id,
                        document_id=document_id,
                        role=role,
                        filename=chunk.filename,
                        page_start=chunk.page_start,
                        page_end=chunk.page_end,
                        clause=chunk.clause,
                        section=chunk.section,
                        text=chunk.text,
                    )
                )
            if role == "tender":
                tender_chunks.extend(chunks)
            else:
                company_chunks.extend(chunks)
    if not tender_chunks:
        raise DocumentParseError("the tender files produced no text. A scanned pack needs tesseract.")

    embedder = build_embedder(settings.embedding_provider, settings.embedding_model)
    reranker = build_reranker(settings.reranker, settings.reranker_model)
    # One embedder instance may be a heavy model. Reuse it for both indexes.
    tender_index = HybridIndex(tender_chunks, embedder, reranker)
    company_index = HybridIndex(company_chunks, embedder, reranker)
    ctx = RunContext(
        llm=llm,
        tender_index=tender_index,
        company_index=company_index,
        prompt_version=settings.prompt_version,
        title=case.title,
    )
    graph = build_graph(ctx)
    final = graph.invoke(initial_state(), {"recursion_limit": 50})
    matrix = list(final.get("matrix") or [])
    decision = go_no_go(matrix)
    report = {
        "disclaimer": DISCLAIMER,
        "go_no_go": decision,
        "summary": _summary(decision, matrix),
        "matrix": matrix,
        "risks": list(final.get("risks") or []),
        "deadlines": list(final.get("deadlines") or []),
        "letter": final.get("letter") or "",
        "letter_quotes": list(final.get("letter_quotes") or []),
        "verification": final.get("verification") or {},
        "plan": final.get("plan") or {},
        "trace": list(final.get("trace") or []),
        "provider": llm.provider,
        "model": llm.model,
        "embedding_provider": settings.embedding_provider,
        "embedding_model": getattr(embedder, "name", settings.embedding_model),
        "reranker": getattr(reranker, "name", settings.reranker),
        "prompt_version": settings.prompt_version,
    }
    latency_ms = int((time.perf_counter() - started) * 1000)
    run.status = "succeeded"
    run.finished_at = utcnow()
    run.latency_ms = latency_ms
    run.prompt_tokens = int(getattr(llm, "prompt_tokens", 0) or 0)
    run.completion_tokens = int(getattr(llm, "completion_tokens", 0) or 0)
    run.report_json = report
    case.status = "succeeded"
    case.go_no_go = decision
    _persist_matrix(session, case.id, run.id, matrix)
    notify_run(session, settings, run, title=case.title, go_no_go=decision)
    session.commit()


def _persist_matrix(session: Session, case_id: str, run_id: str, matrix: list[dict]) -> None:
    for row in matrix:
        requirement_id = uuid.uuid4().hex
        session.add(
            Requirement(
                id=requirement_id,
                run_id=run_id,
                case_id=case_id,
                kind=row.get("kind") or "eligibility",
                text=row.get("text") or "",
                clause=row.get("tender_clause"),
                page=row.get("tender_page"),
                quote=row.get("tender_quote") or "",
                chunk_id=row.get("tender_chunk_id"),
                filename=row.get("tender_filename"),
            )
        )
        session.add(
            EvidenceLink(
                id=uuid.uuid4().hex,
                requirement_id=requirement_id,
                run_id=run_id,
                decision=row.get("decision") or "unclear",
                obligation=row.get("obligation") or "qualification",
                rationale=row.get("rationale") or "",
                evidence_quote=row.get("evidence_quote"),
                evidence_chunk_id=row.get("evidence_chunk_id"),
                evidence_filename=row.get("evidence_filename"),
                evidence_page=row.get("evidence_page"),
                evidence_clause=row.get("evidence_clause"),
            )
        )


def latest_run(session: Session, case_id: str) -> Run | None:
    return session.scalar(select(Run).where(Run.case_id == case_id).order_by(Run.started_at.desc()))


def _title_from(tender_files: list[tuple[str, bytes]], profile_id: str | None) -> str:
    name = Path(tender_files[0][0]).stem.replace("_", " ")
    profile = get_profile(profile_id) if profile_id else None
    if profile:
        return f"{name} — {profile['legal_name']}"
    return name


def _summary(decision: str, matrix: list[dict]) -> str:
    counts = {key: sum(1 for row in matrix if row.get("decision") == key) for key in ("met", "not_met", "missing", "unclear")}
    return (
        f"{decision}. {counts['met']} supported, {counts['not_met']} contradicted, "
        f"{counts['missing']} missing and {counts['unclear']} unclear "
        f"out of {len(matrix)} extracted requirements. {DISCLAIMER}"
    )
