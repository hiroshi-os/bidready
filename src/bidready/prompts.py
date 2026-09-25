"""Prompt versions. Mock mode implements the inclusion rules written into these prompts."""

from __future__ import annotations

PLANNER_QUERIES = [
    "eligibility criteria qualification turnover similar work experience",
    "earnest money deposit EMD bid security",
    "GST PAN registration documents to be submitted undertaking",
    "solvency net worth financial capacity audited turnover",
    "liquidated damages penalty forfeiture risk",
    "pre-bid meeting bid submission deadline opening date",
    "PSARA ISO licence manpower blacklisted local supplier",
]

# Cues named in the prompts. Short cues are matched on word boundaries by the extractor.
V1_CUES = [
    "turnover",
    "solvency",
    "net worth",
    "similar work",
    "similar works",
    "gst",
    "pan card",
    "permanent account",
    "pan",
    "blacklisted",
    "black-listed",
    "iso",
    "psara",
    "manpower",
    "class-i",
    "class i local",
    "local supplier",
    "make in india",
    "msme",
    "udyam",
    "earnest money",
    "emd",
    "bid security",
    "shall submit",
    "must submit",
    "undertaking",
    "audited balance",
    "not have incurred",
    "not incurred",
    "pre-qualification",
    "prequalification",
    "eligibility criteria",
    "work experience",
    "satisfactorily completed",
    "successfully completed",
]

V2_CUES = [
    "turnover",
    "solvency",
    "net worth",
    "similar work",
    "similar works",
    "gst",
    "pan",
    "blacklisted",
    "black-listed",
    "psara",
    "iso",
    "manpower",
    "earnest money",
    "emd",
    "local supplier",
    "make in india",
    "not have incurred",
    "not incurred",
    "successfully completed",
    "satisfactorily completed",
]

EXTRACT_V1 = """You extract bidder requirements from Indian government tender chunks.
Include a sentence only when it states something the bidder must be, have, or submit.
Look for these cues: turnover, solvency, net worth, similar work, GST, PAN, blacklisted,
ISO, PSARA, manpower, local supplier, Make in India, MSME, earnest money, EMD, bid security,
shall submit, must submit, undertaking, audited balance, loss in years, eligibility criteria,
work experience, satisfactorily completed.
The quote must be a verbatim substring of exactly one chunk. Do not paraphrase.
Return JSON: {"requirements": [{"kind": "financial|technical|document|eligibility", "quote": "...", "chunk_id": "..."}]}
If nothing qualifies, return {"requirements": []}.
"""

EXTRACT_V2 = """You extract only mandatory qualification clauses from Indian government tender chunks.
Keep a sentence only when it contains a modal (shall, must, should, required) AND one of these cues:
turnover, solvency, net worth, similar work, GST, PAN, blacklisted, PSARA, ISO, manpower,
earnest money, EMD, local supplier, Make in India, loss, successfully completed.
Drop cover letters, deadlines that are only dates, and generic instructions.
The quote must be a verbatim substring of exactly one chunk. Do not paraphrase.
Return JSON: {"requirements": [{"kind": "financial|technical|document|eligibility", "quote": "...", "chunk_id": "..."}]}
"""

ELIGIBILITY_SYSTEM = """You match one tender requirement to company evidence chunks.
Decisions are only: met, not_met, missing, unclear.
met or not_met requires evidence_quote to be a verbatim substring of evidence_chunk_id.
missing means no uploaded document speaks to the requirement.
unclear means the threshold cannot be read from the quoted text.
obligation is "submission" when the bidder must attach a form or instrument with the bid,
and "qualification" when the bidder must already possess the capacity or licence.
Return JSON: {"decision": "...", "obligation": "...", "evidence_quote": "", "evidence_chunk_id": "", "rationale": "..."}
"""

RISK_SYSTEM = """You list risks and deadlines from tender chunks.
Every item needs a verbatim quote and the chunk_id it came from.
Deadlines are dates for bid submission, pre-bid meetings, opening, or EMD.
Risks are penalties, forfeiture, liquidated damages, and EMD exposure.
Return JSON: {"risks": [{"quote": "...", "chunk_id": "...", "severity": "high|medium"}],
"deadlines": [{"quote": "...", "chunk_id": "...", "event": "..."}]}
"""

DRAFT_SYSTEM = """You write one pre-bid query question for a single cited tender sentence.
Do not invent facts. Do not change the quote. Return JSON: {"question": "..."}
"""

PROMPT_BY_VERSION = {"v1": EXTRACT_V1, "v2": EXTRACT_V2}
CUES_BY_VERSION = {"v1": V1_CUES, "v2": V2_CUES}
