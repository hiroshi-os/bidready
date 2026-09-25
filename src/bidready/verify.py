"""Reject claims whose quote is not a span of the cited chunk. The graph may regenerate once."""

from __future__ import annotations

from bidready.draft import draft_letter
from bidready.rag import HybridIndex
from bidready.textutil import quote_supported

_STAGE_ORDER = ("extractor", "eligibility", "risk", "drafter")


def unsupported_claims(state: dict, tender_index: HybridIndex, company_index: HybridIndex) -> list[dict]:
    failures: list[dict] = []
    for requirement in state.get("requirements") or []:
        _check(failures, "extractor", requirement.get("quote"), requirement.get("chunk_id"), tender_index, company_index)
    for row in state.get("matrix") or []:
        _check(
            failures,
            "eligibility",
            row.get("tender_quote"),
            row.get("tender_chunk_id"),
            tender_index,
            company_index,
        )
        if row.get("evidence_quote"):
            _check(
                failures,
                "eligibility",
                row.get("evidence_quote"),
                row.get("evidence_chunk_id"),
                tender_index,
                company_index,
            )
    for item in list(state.get("risks") or []) + list(state.get("deadlines") or []):
        _check(failures, "risk", item.get("quote"), item.get("chunk_id"), tender_index, company_index)
    for item in state.get("letter_quotes") or []:
        _check(failures, "drafter", item.get("quote"), item.get("chunk_id"), tender_index, company_index)
    return failures


def first_stage(failures: list[dict]) -> str:
    stages = {item["stage"] for item in failures}
    for stage in _STAGE_ORDER:
        if stage in stages:
            return stage
    return "extractor"


def strip_unsupported(state: dict, failures: list[dict], *, title: str) -> dict:
    bad = {(item["stage"], item.get("quote"), item.get("chunk_id")) for item in failures}

    def keep_requirement(row: dict) -> bool:
        return ("extractor", row.get("quote"), row.get("chunk_id")) not in bad

    requirements = [row for row in state.get("requirements") or [] if keep_requirement(row)]
    kept_quotes = {row.get("quote") for row in requirements}

    def keep_matrix(row: dict) -> bool:
        if row.get("tender_quote") not in kept_quotes and ("eligibility", row.get("tender_quote"), row.get("tender_chunk_id")) in bad:
            return False
        if ("eligibility", row.get("tender_quote"), row.get("tender_chunk_id")) in bad:
            return False
        if row.get("evidence_quote") and ("eligibility", row.get("evidence_quote"), row.get("evidence_chunk_id")) in bad:
            return False
        return True

    matrix = [row for row in state.get("matrix") or [] if keep_matrix(row)]
    risks = [
        row
        for row in state.get("risks") or []
        if ("risk", row.get("quote"), row.get("chunk_id")) not in bad
    ]
    deadlines = [
        row
        for row in state.get("deadlines") or []
        if ("risk", row.get("quote"), row.get("chunk_id")) not in bad
    ]
    letter, letter_quotes = draft_letter(title=title, matrix=matrix, deadlines=deadlines)
    letter_quotes = [
        row
        for row in letter_quotes
        if ("drafter", row.get("quote"), row.get("chunk_id")) not in bad
    ]
    return {
        "requirements": requirements,
        "matrix": matrix,
        "risks": risks,
        "deadlines": deadlines,
        "letter": letter,
        "letter_quotes": letter_quotes,
    }


def _check(
    failures: list[dict],
    stage: str,
    quote: str | None,
    chunk_id: str | None,
    tender_index: HybridIndex,
    company_index: HybridIndex,
) -> None:
    if not quote:
        return
    chunk = tender_index.chunks.get(chunk_id or "") or company_index.chunks.get(chunk_id or "")
    if chunk is None or not quote_supported(quote, chunk.text):
        failures.append({"stage": stage, "quote": quote, "chunk_id": chunk_id, "reason": "quote_not_in_chunk"})


def chunk_text(chunk_id: str | None, tender_index: HybridIndex, company_index: HybridIndex) -> str:
    chunk = tender_index.chunks.get(chunk_id or "") or company_index.chunks.get(chunk_id or "")
    return chunk.text if chunk else ""
