from pathlib import Path

from sqlalchemy import select

from bidready.db import make_session_factory
from bidready.graph import RunContext, build_graph, initial_state
from bidready.llm import MockLLM
from bidready.models import EvidenceLink, Notification, Requirement
from bidready.parsing import chunk_document, parse_path
from bidready.pipeline import analyse
from bidready.rag import HashEmbedder, HybridIndex, LexicalReranker
from bidready.textutil import quote_supported
from tests.conftest import write_text_pdf


def test_end_to_end_mock_report(settings, tender_pdf: Path):
    factory = make_session_factory(settings)
    session = factory()
    case_id = analyse(
        settings,
        session,
        tender_files=[("sample-nit.pdf", tender_pdf.read_bytes())],
        profile_id="sample-civil",
        title="Sample municipal works",
    )
    session.expire_all()
    from bidready.models import Case
    from bidready.pipeline import latest_run

    case = session.get(Case, case_id)
    run = latest_run(session, case_id)
    assert case.status == "succeeded", case.error
    report = run.report_json
    assert report["go_no_go"] == "CONDITIONAL"
    assert report["verification"]["passed"] is True
    kinds = {row["text"].lower(): row for row in report["matrix"]}
    gst = next(row for text, row in kinds.items() if "gst" in text)
    turnover = next(row for text, row in kinds.items() if "turnover" in text)
    similar = next(row for text, row in kinds.items() if "similar" in text)
    emd = next(row for text, row in kinds.items() if "earnest" in text)
    assert gst["decision"] == "met"
    assert turnover["decision"] == "met"
    assert similar["decision"] == "met"
    assert emd["decision"] == "missing"
    assert turnover["tender_page"] == 1
    assert emd["tender_page"] == 2
    assert "Pre-bid queries" in report["letter"]
    assert any(event["node"] == "verifier" for event in report["trace"])
    assert session.scalars(select(Requirement).where(Requirement.run_id == run.id)).first() is not None
    assert session.scalars(select(EvidenceLink).where(EvidenceLink.run_id == run.id)).first() is not None
    notes = list(session.scalars(select(Notification).where(Notification.run_id == run.id)))
    assert {note.channel for note in notes} == {"webhook", "email"}
    assert all(note.status == "stubbed" for note in notes)
    session.close()


def test_verifier_retries_a_fabricated_quote(tmp_path: Path):
    tender = tmp_path / "nit.pdf"
    write_text_pdf(
        tender,
        ["1.1 The bidder shall have a valid GST registration certificate for this work."],
    )
    parsed = parse_path(tender)
    chunks = chunk_document(parsed, document_id="doc123", filename="nit.pdf", role="tender")
    index = HybridIndex(chunks, HashEmbedder(), LexicalReranker())
    calls = {"n": 0}

    def extract(state, candidates):
        calls["n"] += 1
        chunk = candidates[0]
        if calls["n"] == 1:
            quote = "The bidder has completed airports on three continents without evidence."
        else:
            quote = chunk.text.strip().splitlines()[-1].strip()
        return [
            {
                "kind": "document",
                "text": quote,
                "quote": quote,
                "chunk_id": chunk.id,
                "page": chunk.page_start,
                "clause": "1.1",
                "section": None,
                "filename": chunk.filename,
            }
        ]

    ctx = RunContext(
        llm=MockLLM(),
        tender_index=index,
        company_index=HybridIndex([], HashEmbedder(), LexicalReranker()),
        title="Retry tender",
        hooks={"extract": extract},
    )
    final = build_graph(ctx).invoke(initial_state(), {"recursion_limit": 50})
    assert calls["n"] >= 2
    assert final["verification"]["passed"] is True
    for row in final["matrix"]:
        source = index.chunks[row["tender_chunk_id"]].text
        assert quote_supported(row["tender_quote"], source)
