"""PDF parsing with an OCR fallback, then layout-aware chunking that keeps clause and page."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf
from PIL import Image

from bidready.textutil import clause_from_text

OCR_MIN_ALNUM = 40
TARGET_CHARS = 1000
HARD_MAX_CHARS = 1800


@dataclass
class ParsedPage:
    number: int
    text: str
    ocr: bool
    blocks: list[str] = field(default_factory=list)


@dataclass
class ParsedDocument:
    pages: list[ParsedPage]

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def ocr_page_count(self) -> int:
        return sum(1 for page in self.pages if page.ocr)


@dataclass
class Chunk:
    id: str
    document_id: str
    filename: str
    role: str
    page_start: int
    page_end: int
    clause: str | None
    section: str | None
    text: str


class DocumentParseError(ValueError):
    pass


def parse_path(path: Path) -> ParsedDocument:
    suffix = path.suffix.lower()
    if suffix == ".txt":
        text = path.read_text(encoding="utf-8", errors="replace")
        return ParsedDocument(pages=[ParsedPage(number=1, text=text, ocr=False, blocks=_blocks_from_text(text))])
    if suffix == ".pdf":
        return parse_pdf(path)
    raise DocumentParseError(f"unsupported file type: {suffix or path.name}")


def parse_pdf(path: Path) -> ParsedDocument:
    try:
        document = pymupdf.open(path)
    except Exception as exc:
        raise DocumentParseError(f"could not open PDF {path.name}: {exc}") from exc
    pages: list[ParsedPage] = []
    try:
        for index, page in enumerate(document, start=1):
            layout = _layout_blocks(page)
            text = "\n".join(layout).strip()
            alnum = sum(character.isalnum() for character in text)
            used_ocr = False
            if alnum < OCR_MIN_ALNUM:
                ocr_text = _ocr_page(page)
                if sum(character.isalnum() for character in ocr_text) > alnum:
                    text = ocr_text
                    layout = _blocks_from_text(text)
                    used_ocr = True
            pages.append(ParsedPage(number=index, text=text, ocr=used_ocr, blocks=layout or _blocks_from_text(text)))
    finally:
        document.close()
    return ParsedDocument(pages=pages)


def _layout_blocks(page: pymupdf.Page) -> list[str]:
    try:
        payload = page.get_text("dict")
    except Exception:
        return []
    blocks: list[str] = []
    for block in payload.get("blocks", []):
        if block.get("type") != 0:
            continue
        lines: list[str] = []
        for line in block.get("lines", []):
            spans = "".join(span.get("text", "") for span in line.get("spans", []))
            if spans.strip():
                lines.append(spans.strip())
        text = "\n".join(lines).strip()
        if text:
            blocks.append(text)
    return blocks


def _blocks_from_text(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]


def _ocr_page(page: pymupdf.Page) -> str:
    try:
        import pytesseract
    except ImportError:
        return ""
    pixmap = page.get_pixmap(dpi=200)
    image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
    try:
        return pytesseract.image_to_string(image) or ""
    except Exception:
        return ""


def chunk_document(parsed: ParsedDocument, *, document_id: str, filename: str, role: str) -> list[Chunk]:
    blocks: list[tuple[int, str]] = []
    for page in parsed.pages:
        source = page.blocks or _blocks_from_text(page.text)
        for block in source:
            blocks.append((page.number, block))
    return _chunk_blocks(blocks, document_id=document_id, filename=filename, role=role)


def _chunk_blocks(
    blocks: list[tuple[int, str]], *, document_id: str, filename: str, role: str
) -> list[Chunk]:
    # One visual line per block so headings and clause numbers are visible,
    # and so a chunk is not glued across a page boundary.
    expanded: list[tuple[int, str]] = []
    for page, text in blocks:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines and text.strip():
            lines = [text.strip()]
        expanded.extend((page, line) for line in lines)
    expanded = _merge_wrapped_lines(expanded)

    chunks: list[Chunk] = []
    buffer: list[tuple[int, str]] = []
    buffer_len = 0
    section: str | None = None
    clause: str | None = None

    def flush(*, final: bool) -> None:
        nonlocal buffer, buffer_len
        if not buffer:
            return
        text = "\n".join(part for _, part in buffer).strip()
        if text:
            pages = [page for page, _ in buffer]
            own_clause = clause_from_text(text) or clause
            chunks.append(
                Chunk(
                    id=f"{document_id}-{len(chunks) + 1}",
                    document_id=document_id,
                    filename=filename,
                    role=role,
                    page_start=min(pages),
                    page_end=max(pages),
                    clause=own_clause,
                    section=section,
                    text=text,
                )
            )
        if final or not buffer:
            buffer = []
            buffer_len = 0
            return
        last = buffer[-1]
        if len(last[1]) < 300:
            buffer = [last]
            buffer_len = len(last[1])
        else:
            buffer = []
            buffer_len = 0

    for page, text in expanded:
        flat = " ".join(text.split())
        if _is_heading(flat):
            flush(final=True)
            section = flat[:240]
            found = clause_from_text(flat)
            if found:
                clause = found
            continue
        found = clause_from_text(flat)
        if found:
            clause = found
        if buffer and page != buffer[0][0]:
            flush(final=True)
        if buffer and buffer_len + len(text) > TARGET_CHARS:
            flush(final=False)
        buffer.append((page, text))
        buffer_len += len(text)
        if buffer_len >= HARD_MAX_CHARS:
            flush(final=False)
    flush(final=True)
    return chunks


# A new numbered clause or a repeated "BIDDER SHOULD" / EMD line is its own requirement.
# "(GROSS) OF 88 LAKHS" and "EACH COSTING..." are wraps of the line above.
_NEW_ITEM = re.compile(
    r"^(?:"
    r"\d+(?:\.\d+)*\s*[).:\-]\s+\S"
    r"|(?::\s*)?(?:bidder\b|the\s+bidder\b|earnest\b|emd\b)"
    r")",
    re.IGNORECASE,
)


def _merge_wrapped_lines(lines: list[tuple[int, str]]) -> list[tuple[int, str]]:
    """Join visual wraps on the same page. Do not glue the next requirement on."""
    merged: list[tuple[int, str]] = []
    for page, text in lines:
        heading = _is_heading(text)
        new_item = bool(_NEW_ITEM.match(text))
        if (
            merged
            and not heading
            and not new_item
            and not _is_heading(merged[-1][1])
            and merged[-1][0] == page
            and merged[-1][1][-1] not in ".:;"
        ):
            previous_page, previous = merged[-1]
            merged[-1] = (previous_page, f"{previous} {text}")
        else:
            merged.append((page, text))
    return merged


def _is_heading(line: str) -> bool:
    """Section titles, not all-caps requirement lines.

    Indian tender PDFs often set the whole page in capitals, including
    "BIDDER SHOULD HAVE AVERAGE ANNUAL TURNOVER" and the wrapped amount
    "(GROSS) OF 88 LAKHS". Those stay in the chunk body so the threshold
    remains next to the criterion.
    """
    if len(line) < 4 or len(line) > 90:
        return False
    if re.search(
        r"\b(shall|must|should|rs\.?|inr|rupees?|lakh|lakhs|crore|crores)\b",
        line,
        re.IGNORECASE,
    ):
        return False
    if "₹" in line:
        return False
    if re.match(
        r"^(section|annexure|appendix|schedule|chapter|part|clause)\b",
        line,
        re.IGNORECASE,
    ):
        return True
    if line[:1] in "(:/" or re.search(r"\d", line):
        return False
    letters = [character for character in line if character.isalpha()]
    if letters and sum(character.isupper() for character in letters) / len(letters) > 0.75:
        return len(line.split()) <= 12
    return False
