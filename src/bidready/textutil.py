"""Shared text helpers: tokens, amounts, sentences, citation checks."""

from __future__ import annotations

import re

TOKEN_RE = re.compile(r"[a-z0-9]+")
AMOUNT_RE = re.compile(
    r"(?:₹|rs\.?|inr)\s*([0-9]{1,3}(?:,[0-9]{2,3})+|[0-9]{4,})",
    re.IGNORECASE,
)
WORD_AMOUNT_RE = re.compile(
    r"(?:(?:rs\.?|inr|₹)\s*)?"
    r"(\d+(?:\.\d+)?|one|two|three|four|five|ten|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)"
    r"\s*(lakh|lakhs|crore|crores)\b",
    re.IGNORECASE,
)
CLAUSE_RE = re.compile(r"^(?:clause\s+)?(\d+(?:\.\d+)*)(?:\s*[).:-]|\s+)", re.IGNORECASE)
SIMILAR_WORK_RE = re.compile(r"similar(?:\s+[a-z]+){0,4}\s+works?\b", re.IGNORECASE)
DATE_RE = re.compile(
    r"\b(\d{1,2}[./-]\d{1,2}[./-]\d{2,4}|"
    r"\d{1,2}[\s-](?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*[\s-]\d{2,4})\b",
    re.IGNORECASE,
)

_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "ten": 10,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}
_SCALES = {"lakh": 100_000, "lakhs": 100_000, "crore": 10_000_000, "crores": 10_000_000}


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def whitespace_norm(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def quote_supported(quote: str, source: str, min_len: int = 20) -> bool:
    """A citation is supported when the quote is a real span of the source chunk."""
    if not quote or not source:
        return False
    folded = whitespace_norm(quote)
    if len(folded) < min_len:
        return False
    if quote in source:
        return True
    return folded in whitespace_norm(source)


_ABBREV_END = re.compile(r"(?:^|[\s(])(?:rs|mr|mrs|ms|dr|no|cl|fig|vol|vs|etc)\.$", re.IGNORECASE)


def sentences(text: str) -> list[str]:
    """Split into sentences that still occur in the source after whitespace normalisation.

    Periods in Rs. / Mr. / No. are not sentence boundaries. Tender lines use them
    in front of amounts ("Rs. 50,00,000"), and splitting there drops the threshold.
    """
    if not text or not text.strip():
        return []
    raw_parts: list[str] = []
    for para in re.split(r"\n+", text):
        if not para.strip():
            continue
        start = 0
        for match in re.finditer(r"\.\s+(?=[A-Z0-9(\"'])", para):
            piece = para[start : match.start() + 1]
            if _ABBREV_END.search(piece.strip()):
                continue
            raw_parts.append(piece)
            start = match.end()
        raw_parts.append(para[start:])
    norm_source = whitespace_norm(text)
    kept: list[str] = []
    for part in raw_parts:
        seg = part.strip()
        if len(seg) < 20:
            continue
        if seg in text or whitespace_norm(seg) in norm_source:
            kept.append(seg)
    return kept


def clause_from_text(text: str) -> str | None:
    match = CLAUSE_RE.match(text.strip())
    if not match:
        return None
    return match.group(1)


def parse_inr_token(raw: str) -> int | None:
    digits = raw.replace(",", "").strip()
    if not digits.isdigit():
        return None
    value = int(digits)
    return value if value > 0 else None


def amounts_in(text: str) -> list[int]:
    found: list[int] = []
    for match in AMOUNT_RE.finditer(text or ""):
        value = parse_inr_token(match.group(1))
        if value is not None:
            found.append(value)
    for match in WORD_AMOUNT_RE.finditer(text or ""):
        number = match.group(1).lower()
        scale = _SCALES[match.group(2).lower()]
        if number.replace(".", "", 1).isdigit():
            amount = int(round(float(number) * scale))
        else:
            amount = _WORDS.get(number, 0) * scale
        if amount > 0:
            found.append(amount)
    return found


def has_date(text: str) -> bool:
    return DATE_RE.search(text or "") is not None
