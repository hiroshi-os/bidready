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

EXTRACT_V1 = """You extract bidder requirements from a menu of spans in an Indian government tender.
A requirement is something the bidder must be, have, or submit: turnover, solvency, net worth,
similar work, GST, PAN, blacklisting, ISO, PSARA, manpower, local supplier, Make in India, MSME,
earnest money, EMD, bid security, undertaking, audited accounts, or work experience.
Skip a span that is only a date, a cover letter, or a post-award penalty.
Copy span_id from the menu. Write text as one short sentence in your own words. Do not paste the span.
Return one object for every qualifying span.
This example is fictional and is not from the tender you are reading.
Menu: S1: The bidder shall have an average annual turnover of Rs. 80 lakh during the last three financial years.
S2: Pre-bid meeting will be held on 02 January 2026 at 11:00 hours.
S3: The bidder shall not have been blacklisted by any government department.
{"requirements":[{"span_id":"S1","kind":"financial","text":"Average annual turnover must be at least Rs. 80 lakh over three years."},{"span_id":"S3","kind":"eligibility","text":"The bidder must not be blacklisted by a government department."}]}
Use that JSON shape. Leave requirements empty only when no span qualifies.
"""

EXTRACT_V2 = """You extract only mandatory qualification clauses from a menu of spans.
Keep a span only when it contains a modal (shall, must, should, required) AND one of these cues:
turnover, solvency, net worth, similar work, GST, PAN, blacklisted, PSARA, ISO, manpower,
earnest money, EMD, local supplier, Make in India, loss, successfully completed.
Drop dates, cover letters, and generic instructions.
Copy span_id from the menu. Write text as one short sentence in your own words. Do not paste the span.
Return one object for every qualifying span.
This example is fictional and is not from the tender you are reading.
Menu: S1: The bidder must have a solvency of not less than Rs. 40 lakh.
S2: Bids shall be submitted online on the portal.
S3: The bidder shall not have been blacklisted by any government department.
{"requirements":[{"span_id":"S1","kind":"financial","text":"A solvency certificate of at least Rs. 40 lakh is mandatory."},{"span_id":"S3","kind":"eligibility","text":"The bidder must not be blacklisted by a government department."}]}
Use that JSON shape. Leave requirements empty only when no span qualifies.
"""

ELIGIBILITY_SYSTEM = """You decide whether company evidence meets a tender requirement.
Choose evidence_span from the menu ids (E1, E2, ...) or NONE. Do not write a quote.
met: the chosen span shows the company satisfies the requirement, including any amount or count.
not_met: the chosen span shows the company fails that requirement.
missing: no span is about this requirement. Use NONE.
unclear: a span is on the topic but the amounts cannot be compared. Pick that span when you can.
The words synthetic and fictional are labels on these evaluation files. Judge the amounts, names, and facts in the span. Do not answer unclear only because a span says fictional.
Lines may end with an amounts hint in INR, already parsed from the text. Compare those integers. Do not recount Indian comma groups. A company amount above a stated minimum meets a turnover, solvency, or work-value requirement.
obligation is submission when the bidder must attach a form or instrument with the bid,
and qualification when the bidder must already possess the capacity or licence.
Each requirement has an id such as R1. Copy that id.
Return JSON: {"decisions":[{"id":"R1","decision":"met|not_met|missing|unclear","obligation":"qualification|submission","evidence_span":"E1","rationale":"one sentence"}]}
"""

EXTRACT_SCHEMA = {
    "type": "object",
    "properties": {
        "requirements": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "span_id": {"type": "string"},
                    "kind": {"type": "string"},
                    "text": {"type": "string"},
                },
                "required": ["span_id", "text"],
            },
        }
    },
    "required": ["requirements"],
}

ELIGIBILITY_SCHEMA = {
    "type": "object",
    "properties": {
        "decisions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "decision": {"type": "string"},
                    "obligation": {"type": "string"},
                    "evidence_span": {"type": "string"},
                    "rationale": {"type": "string"},
                },
                "required": ["id", "decision", "evidence_span"],
            },
        }
    },
    "required": ["decisions"],
}

RISK_SYSTEM = """You list risks and deadlines from tender chunks.
Every item needs a verbatim quote and the chunk_id it came from.
Deadlines are dates for bid submission, pre-bid meetings, opening, or EMD.
Risks are penalties, forfeiture, liquidated damages, and EMD exposure.
Return at most 4 risks and 4 deadlines. Keep each quote under 300 characters.
Return JSON: {"risks": [{"quote": "...", "chunk_id": "...", "severity": "high|medium"}],
"deadlines": [{"quote": "...", "chunk_id": "...", "event": "..."}]}
"""

DRAFT_SYSTEM = """You write one pre-bid query question for a single cited tender sentence.
Do not invent facts. Do not change the quote. Return JSON: {"question": "..."}
"""

PROMPT_BY_VERSION = {"v1": EXTRACT_V1, "v2": EXTRACT_V2}
CUES_BY_VERSION = {"v1": V1_CUES, "v2": V2_CUES}
