from bidready.graph import (
    _amount_hint,
    _decision_from_menu,
    _dedupe_requirements,
    _pick_decision,
    _requirement_from_item,
    _span_by_id,
    _span_menu,
)
from bidready.parsing import Chunk


def _chunk(text: str, chunk_id: str = "c1") -> Chunk:
    return Chunk(
        id=chunk_id,
        document_id="d",
        filename="company.txt",
        role="company",
        page_start=1,
        page_end=1,
        clause=None,
        section="Eligibility",
        text=text,
    )


def test_met_without_a_menu_span_keeps_the_decision():
    requirement = {
        "kind": "eligibility",
        "text": "The bidder shall have ISO 9001 certification.",
        "quote": "The bidder shall have ISO 9001 certification.",
        "chunk_id": "tender-1",
        "page": 2,
        "filename": "nit.pdf",
    }
    hits = [_chunk("The company has a GST registration and a PAN card on file for bidding.")]
    row = _decision_from_menu(requirement, {"decision": "met", "evidence_span": "NONE", "rationale": "Claimed."}, hits, [], None)
    assert row["decision"] == "met"
    assert row["rule_decision"] == "missing"
    assert row["evidence_quote"] is None


def test_menu_span_is_copied_verbatim():
    evidence = _chunk("The firm holds an ISO 9001 certificate issued in 2023 for its quality system.")
    menu = _span_menu([evidence], prefix="E", per_chunk=4, limit=4)
    assert menu
    requirement = {
        "kind": "eligibility",
        "text": "The bidder shall have ISO 9001 certification.",
        "quote": "The bidder shall have ISO 9001 certification.",
        "chunk_id": "tender-1",
        "page": 2,
        "filename": "nit.pdf",
    }
    row = _decision_from_menu(
        requirement,
        {"decision": "met", "evidence_span": menu[0]["id"], "obligation": "qualification", "rationale": "Certificate is present."},
        [evidence],
        menu,
        None,
    )
    assert row["decision"] == "met"
    assert row["evidence_quote"] == menu[0]["text"]
    assert row["evidence_chunk_id"] == evidence.id


def test_span_id_parsing_ignores_punctuation():
    menu = [{"id": "E2", "chunk": None, "text": "span"}]
    assert _span_by_id(menu, "e2.")["id"] == "E2"
    assert _span_by_id(menu, "NONE") is None


def test_requirement_text_is_not_forced_to_equal_the_quote():
    chunk = _chunk("The bidder shall have an average annual turnover of Rs. 80 lakh during the last three financial years.")
    menu = [{"id": "S1", "chunk": chunk, "text": chunk.text}]
    built = _requirement_from_item(
        None,
        {"span_id": "S1", "kind": "financial", "text": "The bidder must show turnover of at least Rs. 80 lakh over three years."},
        menu,
    )
    assert built["quote"] == chunk.text
    assert built["text"] != built["quote"]


def test_watermark_lines_are_dropped_from_the_evidence_menu():
    text = (
        "SYNTHETIC — fictional chartered-accountant certificate created for the bidready evaluation. UDIN is not real.\n"
        "Average annual financial turnover of Sample Civil Works on construction works for the three financial years ending 31 March 2024 is Rs. 1,20,00,000.\n"
        "The figure is fictional and exists only so eligibility checks can be scored.\n"
        "This statement is synthetic. The agency does not hold a PSARA licence in this evaluation pack."
    )
    menu = _span_menu([_chunk(text)], prefix="E", per_chunk=6, limit=6, drop_watermarks=True)
    joined = " ".join(item["text"] for item in menu)
    assert "1,20,00,000" in joined
    assert "PSARA" in joined
    assert "UDIN is not real" not in joined
    assert "exists only so eligibility" not in joined


def test_amount_hint_parses_indian_comma_groups_and_lakh():
    assert _amount_hint("turnover is Rs. 1,20,00,000.") == " [amounts: 12000000 INR]"
    assert _amount_hint("average annual turnover of Rs. 30 Lakh") == " [amounts: 3000000 INR]"
    assert _amount_hint("no figure here at all") == ""


def test_decisions_match_r_ids_not_a_shared_chunk_id():
    by_id = {"R1": {"id": "R1", "decision": "met"}, "R2": {"id": "R2", "decision": "missing"}}
    parsed = list(by_id.values())
    assert _pick_decision(by_id, parsed, 1, 2)["decision"] == "met"
    assert _pick_decision(by_id, parsed, 2, 2)["decision"] == "missing"
    positional = [{"id": "same-chunk", "decision": "not_met"}, {"id": "same-chunk", "decision": "unclear"}]
    assert _pick_decision({}, positional, 2, 2)["decision"] == "unclear"


def test_dedupe_drops_a_quote_contained_in_one_already_kept():
    rows = [
        {"quote": "The bidder shall submit earnest money of Rs. 20,000 along with the bid."},
        {"quote": "earnest money of Rs. 20,000"},
        {"quote": "The bidder must hold a valid GST registration certificate."},
    ]
    kept = _dedupe_requirements(rows)
    assert len(kept) == 2
