from bidready.decide import decide, go_no_go, parse_similar_alternatives
from bidready.parsing import Chunk
from bidready.textutil import amounts_in, quote_supported


def _chunk(text: str, filename: str = "evidence.txt") -> Chunk:
    return Chunk(
        id="c1",
        document_id="d1",
        filename=filename,
        role="company",
        page_start=1,
        page_end=1,
        clause=None,
        section=None,
        text=text,
    )


def test_indian_amounts_and_words():
    assert 22_60_000 in amounts_in("Average turnover Rs. 22,60,000 during the year")
    assert 12_000_000 in amounts_in("turnover is Rs. 1,20,00,000 for three years")
    assert 2_000_000 in amounts_in("average turnover of more than 20 Lakhs")
    assert 10_000_000 in amounts_in("minimum average annual turnover of Rs. One crore")


def test_similar_work_alternatives():
    simple = "The bidder should have completed one similar civil work of value not less than Rs. 20,00,000."
    assert parse_similar_alternatives(simple) == [(1, 2_000_000)]
    classic = (
        "3 (Three) similar works each of value not less than Rs. 9,04,000/- or "
        "2 (Two) similar works each of value not less than Rs. 13,56,000/- or "
        "1 (One) similar work of value at least Rs. 18,08,000/-"
    )
    assert parse_similar_alternatives(classic) == [(3, 904_000), (2, 1_356_000), (1, 1_808_000)]
    lakhs = "BIDDER SHOULD HAVE COMPLETED THREE SIMILAR WORKS EACH COSTING NOT LESS THAN RUPEES 70 LAKHS"
    assert parse_similar_alternatives(lakhs) == [(3, 7_000_000)]


def test_turnover_met_and_not_met():
    requirement = {
        "kind": "financial",
        "text": "Average annual financial turnover shall be at least Rs. 50,00,000 during the last three financial years.",
        "quote": "Average annual financial turnover shall be at least Rs. 50,00,000 during the last three financial years.",
        "chunk_id": "t1",
        "page": 1,
        "clause": "1.2",
        "filename": "nit.pdf",
    }
    high = _chunk(
        "Average annual financial turnover of the firm on construction works is Rs. 1,20,00,000."
    )
    met = decide(requirement, [high])
    assert met["decision"] == "met"
    assert quote_supported(met["evidence_quote"], high.text)

    low = _chunk("Average annual financial turnover on construction works is Rs. 10,00,000.")
    missed = decide(requirement, [low])
    assert missed["decision"] == "not_met"


def test_emd_missing_is_submission_and_conditional():
    requirement = {
        "kind": "financial",
        "text": "Earnest Money Deposit of Rs. 1,00,000 shall be submitted with the bid.",
        "quote": "Earnest Money Deposit of Rs. 1,00,000 shall be submitted with the bid.",
        "chunk_id": "t2",
        "page": 2,
        "clause": "2.1",
        "filename": "nit.pdf",
    }
    row = decide(requirement, [])
    assert row["decision"] == "missing"
    assert row["obligation"] == "submission"
    assert go_no_go([row]) == "CONDITIONAL"
    assert go_no_go([{**row, "decision": "not_met", "obligation": "qualification"}]) == "NO-GO"


def test_quote_must_be_a_real_span():
    source = "The bidder shall have a valid GST registration certificate issued in India."
    assert quote_supported("The bidder shall have a valid GST registration certificate issued in India.", source)
    assert not quote_supported("The bidder is the lowest in the state.", source)
    assert not quote_supported("GST", source)
