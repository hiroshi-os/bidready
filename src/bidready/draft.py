"""Draft a pre-bid query letter that quotes only spans already extracted from the tender."""

from __future__ import annotations

DISCLAIMER = (
    "Draft for a human to review. This is not legal advice and it is not a bid recommendation "
    "from the tendering authority."
)


def draft_letter(*, title: str, matrix: list[dict], deadlines: list[dict]) -> tuple[str, list[dict]]:
    queries = _queries(matrix, deadlines)
    lines = [
        "To",
        "The Tender Inviting Authority",
        "",
        f"Subject: Pre-bid queries regarding {title or 'the tender'}",
        "",
        "Sir or Madam,",
        "",
        "We intend to participate in the subject tender. Please clarify the points below. "
        "Each point quotes the tender text it refers to.",
        "",
    ]
    quotes: list[dict] = []
    if not queries:
        lines.append("We have no clarification on the clauses extracted from the tender pack.")
        lines.append("")
    for index, query in enumerate(queries, start=1):
        clause = query.get("clause") or "—"
        page = query.get("page") or "—"
        lines.append(f"{index}. Clause {clause}, page {page} ({query['filename']}).")
        lines.append(f'   Tender text: "{query["quote"]}"')
        lines.append(f"   Query: {query['question']}")
        lines.append("")
        quotes.append({"quote": query["quote"], "chunk_id": query["chunk_id"], "stage": "drafter"})
    lines.append(DISCLAIMER)
    lines.append("")
    lines.append("Yours faithfully,")
    lines.append("[Authorised signatory]")
    lines.append("[Company name]")
    return "\n".join(lines), quotes


def _queries(matrix: list[dict], deadlines: list[dict]) -> list[dict]:
    rank = {"not_met": 0, "unclear": 1, "missing": 2}
    rows = [row for row in matrix if row.get("decision") != "met" and row.get("tender_quote") and row.get("tender_chunk_id")]
    rows.sort(key=lambda row: rank.get(row.get("decision"), 9))
    queries = []
    for row in rows[:6]:
        queries.append(
            {
                "quote": row["tender_quote"],
                "chunk_id": row["tender_chunk_id"],
                "clause": row.get("tender_clause"),
                "page": row.get("tender_page"),
                "filename": row.get("tender_filename") or "tender",
                "question": _question(row),
            }
        )
    for deadline in deadlines[:2]:
        if not deadline.get("quote") or not deadline.get("chunk_id"):
            continue
        queries.append(
            {
                "quote": deadline["quote"],
                "chunk_id": deadline["chunk_id"],
                "clause": deadline.get("clause"),
                "page": deadline.get("page"),
                "filename": deadline.get("filename") or "tender",
                "question": "Please confirm that this date is unchanged by any corrigendum, and confirm the timezone.",
            }
        )
    return queries[:8]


def _question(row: dict) -> str:
    decision = row.get("decision")
    if decision == "missing" and row.get("obligation") == "submission":
        return (
            "Please confirm the acceptable form, the validity required, and the authority "
            "in whose favour this is to be drawn."
        )
    if decision == "missing":
        return "Please confirm whether this qualification is mandatory and which document will be accepted as evidence."
    if decision == "not_met":
        return (
            "Please confirm whether an MSME or startup relaxation applies to this threshold, "
            "and which document should demonstrate it."
        )
    return "Please confirm the exact threshold, the financial years to be counted, and the document that will be accepted."
