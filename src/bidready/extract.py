"""Deterministic requirement, risk and deadline extraction. Quotes are source sentences."""

from __future__ import annotations

import re

from bidready.parsing import Chunk
from bidready.prompts import CUES_BY_VERSION, PLANNER_QUERIES
from bidready.rag import HybridIndex
from bidready.textutil import (
    SIMILAR_WORK_RE,
    clause_from_text,
    has_date,
    sentences,
    whitespace_norm,
)

_MODAL = re.compile(r"\b(shall|must|should|required|requirement)\b", re.IGNORECASE)
_SHORT_CUES = {"gst", "pan", "emd", "iso"}
_DEADLINE_CUES = (
    "bid submission",
    "submission end",
    "pre-bid",
    "prebid",
    "opening of",
    "bid opening",
    "clarification",
    "due date",
    "last date",
)
_RISK_CUES = (
    "liquidated",
    "penalty",
    "forfeiture",
    "forfeited",
    "damages",
    "earnest",
    "emd",
    "bid security",
    "validity of the bid",
    "bid validity",
)


def candidate_chunks(index: HybridIndex, queries: list[str] | None = None, k: int = 8) -> list[Chunk]:
    """Retrieved chunks plus chunks whose section heading is an eligibility-style heading."""
    chosen: dict[str, Chunk] = {}
    for query in queries or PLANNER_QUERIES:
        for chunk in index.search(query, k=k, mode="hybrid_rerank"):
            chosen.setdefault(chunk.id, chunk)
    for chunk in index.chunks.values():
        blob = f"{chunk.section or ''} {chunk.text[:120]}".lower()
        if any(word in blob for word in ("eligibility", "qualification", "earnest", "pre-qualification")):
            chosen.setdefault(chunk.id, chunk)
    return list(chosen.values())


def extract_requirements(chunks: list[Chunk], policy: str = "v1") -> list[dict]:
    cues = CUES_BY_VERSION.get(policy, CUES_BY_VERSION["v1"])
    require_modal = policy == "v2"
    rows: list[dict] = []
    seen: set[str] = set()
    for chunk in chunks:
        for sentence in sentences(chunk.text):
            if len(sentence) < 40:
                continue
            lowered = sentence.lower()
            if require_modal and not _MODAL.search(sentence):
                continue
            if not _has_cue(lowered, cues):
                continue
            if _is_pure_deadline(sentence):
                continue
            key = whitespace_norm(lowered)[:220]
            if key in seen:
                continue
            seen.add(key)
            rows.append(_requirement(chunk, sentence))
    rows.sort(key=lambda row: (-_cue_hits(row["text"].lower(), cues), row["page"] or 0, row["text"]))
    return rows


def extract_risks_and_deadlines(chunks: list[Chunk]) -> tuple[list[dict], list[dict]]:
    risks: list[dict] = []
    deadlines: list[dict] = []
    seen_risk: set[str] = set()
    seen_deadline: set[str] = set()
    for chunk in chunks:
        for sentence in sentences(chunk.text):
            lowered = sentence.lower()
            key = whitespace_norm(lowered)[:220]
            if has_date(sentence) and any(cue in lowered for cue in _DEADLINE_CUES):
                if key not in seen_deadline:
                    seen_deadline.add(key)
                    deadlines.append({**_citation(chunk, sentence), "event": _deadline_label(lowered)})
            if any(cue in lowered for cue in _RISK_CUES) or re.search(r"\bemd\b", lowered):
                if key not in seen_risk and not _is_pure_deadline(sentence):
                    seen_risk.add(key)
                    severity = "high" if any(word in lowered for word in ("liquidated", "forfeit", "penalty")) else "medium"
                    risks.append({**_citation(chunk, sentence), "severity": severity})
    return risks[:24], deadlines[:16]


def _requirement(chunk: Chunk, sentence: str) -> dict:
    lowered = sentence.lower()
    return {
        **_citation(chunk, sentence),
        "kind": _kind(lowered),
        "text": whitespace_norm(sentence),
    }


def _citation(chunk: Chunk, sentence: str) -> dict:
    return {
        "quote": sentence.strip(),
        "chunk_id": chunk.id,
        "page": chunk.page_start,
        "clause": clause_from_text(sentence) or chunk.clause,
        "section": chunk.section,
        "filename": chunk.filename,
    }


def _kind(lowered: str) -> str:
    if any(word in lowered for word in ("turnover", "solvency", "net worth", "earnest money", "bid security")):
        return "financial"
    if re.search(r"\bemd\b", lowered):
        return "financial"
    if SIMILAR_WORK_RE.search(lowered) or any(
        word in lowered
        for word in ("manpower", "psara", "iso", "work experience", "successfully completed", "satisfactorily completed")
    ):
        return "technical"
    if any(word in lowered for word in ("gst", "pan card", "permanent account", "undertaking", "shall submit", "must submit", "msme", "udyam")):
        return "document"
    if re.search(r"\bpan\b", lowered):
        return "document"
    return "eligibility"


def _has_cue(lowered: str, cues: list[str]) -> bool:
    if SIMILAR_WORK_RE.search(lowered):
        return True
    for cue in cues:
        if cue in _SHORT_CUES:
            if re.search(rf"\b{re.escape(cue)}\b", lowered):
                return True
        elif cue in lowered:
            return True
    return False


def _cue_hits(lowered: str, cues: list[str]) -> int:
    return sum(1 for cue in cues if (re.search(rf"\b{cue}\b", lowered) if cue in _SHORT_CUES else cue in lowered))


def _is_pure_deadline(sentence: str) -> bool:
    lowered = sentence.lower()
    if not has_date(sentence):
        return False
    if not any(cue in lowered for cue in _DEADLINE_CUES):
        return False
    strong = ("turnover", "solvency", "similar work", "gst", "earnest", "emd", "pan", "psara")
    return not any(word in lowered for word in strong)


def _deadline_label(lowered: str) -> str:
    if "pre-bid" in lowered or "prebid" in lowered:
        return "pre-bid meeting"
    if "opening" in lowered:
        return "bid opening"
    if "clarification" in lowered:
        return "clarification window"
    if "submission" in lowered or "due date" in lowered or "last date" in lowered:
        return "bid submission"
    return "dated event"
