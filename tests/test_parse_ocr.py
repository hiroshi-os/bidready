from pathlib import Path

import pymupdf
from PIL import Image, ImageDraw, ImageFont

from bidready.extract import extract_requirements
from bidready.parsing import chunk_document, parse_pdf
from tests.conftest import TENDER_PAGE_1, TENDER_PAGE_2, write_text_pdf


def test_ocr_fallback_reads_a_scanned_page(tmp_path: Path):
    image_path = tmp_path / "scan.png"
    image = Image.new("RGB", (1600, 360), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 42)
    draw.text((40, 140), "Clause 4.1 The bidder shall submit a PAN card.", fill="black", font=font)
    image.save(image_path)

    pdf_path = tmp_path / "scanned.pdf"
    document = pymupdf.open()
    page = document.new_page(width=595, height=842)
    page.insert_image(pymupdf.Rect(36, 36, 560, 220), filename=str(image_path))
    document.save(pdf_path)
    document.close()

    parsed = parse_pdf(pdf_path)
    assert parsed.ocr_page_count == 1
    assert "pan" in parsed.pages[0].text.lower()
    chunks = chunk_document(parsed, document_id="doc-ocr", filename="scanned.pdf", role="tender")
    assert chunks
    assert chunks[0].page_start == 1


def test_clause_heading_is_not_merged_into_the_next_requirement(tmp_path: Path):
    path = tmp_path / "nit.pdf"
    write_text_pdf(path, [TENDER_PAGE_1, TENDER_PAGE_2])
    parsed = parse_pdf(path)
    chunks = chunk_document(parsed, document_id="doc-heading", filename="nit.pdf", role="tender")
    rows = extract_requirements(chunks, "v1")
    similar = next(row for row in rows if "similar" in row["text"].lower())
    earnest = next(row for row in rows if "shall be submitted" in row["text"].lower())
    assert "earnest" not in similar["text"].lower()
    assert similar["page"] == 2
    assert earnest["page"] == 2


def test_all_caps_list_items_keep_wraps_and_stay_separate():
    from bidready.parsing import _chunk_blocks

    chunks = _chunk_blocks(
        [
            (4, "BIDDER SHOULD HAVE COMPLETED THREE SIMILAR WORKS"),
            (4, "EACH COSTING NOT LESS THAN RUPEES 70 LAKHS"),
            (4, "BIDDER SHOULD HAVE AVERAGE ANNUAL TURNOVER"),
            (4, "(GROSS) OF 88 LAKHS"),
            (4, "EMD OF RUPEES 3,52,160"),
        ],
        document_id="d",
        filename="t.pdf",
        role="tender",
    )
    lines = [line for chunk in chunks for line in chunk.text.splitlines()]
    similar = next(line for line in lines if "THREE SIMILAR WORKS" in line)
    turnover = next(line for line in lines if "AVERAGE ANNUAL TURNOVER" in line)
    emd = next(line for line in lines if "EMD OF RUPEES" in line)
    assert "70 LAKHS" in similar
    assert "88 LAKHS" in turnover
    assert "88 LAKHS" not in similar
    assert "TURNOVER" not in emd
    assert "EMD" not in turnover
