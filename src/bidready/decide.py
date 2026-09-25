"""Match one requirement sentence to company evidence. The cited span is what the decision uses."""

from __future__ import annotations

import re

from bidready.parsing import Chunk
from bidready.textutil import amounts_in, sentences

_COUNT_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "1": 1, "2": 2, "3": 3, "4": 4}
_ALT_RE = re.compile(
    r"(?:(\d+)\s*\(\s*(one|two|three|four)\s*\)|(one|two|three|four|\d+))"
    r"\s+similar(?:\s+[a-z]+){0,4}\s+works?\b"
    r".{0,220}?"
    r"(?:not less than|at least)"
    r"(?:\s*(?:rs\.?|inr|₹|rupees))?"
    r"\s*([0-9][0-9,]*(?:\.\d+)?)"
    r"(?:\s*(lakh|lakhs|crore|crores))?",
    re.IGNORECASE | re.DOTALL,
)
_DOMAINS = ("civil", "construction", "security", "forensic", "interior", "software", "ambulance")
_DECISION_RANK = {"not_met": 3, "missing": 2, "unclear": 1, "met": 0}


def decide(requirement: dict, evidence_hits: list[Chunk], *, estimated_cost_inr: int | None = None) -> dict:
    text = requirement.get("text") or requirement.get("quote") or ""
    outcome = _apply(text, evidence_hits, estimated_cost_inr=estimated_cost_inr)
    return {
        "kind": requirement.get("kind") or "eligibility",
        "text": text,
        "decision": outcome["decision"],
        "obligation": outcome["obligation"],
        "rationale": outcome["rationale"],
        "evidence_quote": outcome.get("quote"),
        "evidence_chunk_id": outcome.get("chunk_id"),
        "evidence_filename": outcome.get("filename"),
        "evidence_page": outcome.get("page"),
        "evidence_clause": outcome.get("clause"),
        "tender_quote": requirement.get("quote") or text,
        "tender_chunk_id": requirement.get("chunk_id"),
        "tender_page": requirement.get("page"),
        "tender_clause": requirement.get("clause"),
        "tender_filename": requirement.get("filename"),
    }


def go_no_go(rows: list[dict]) -> str:
    """Qualification failures are NO-GO. A missing bid instrument is CONDITIONAL. This is not a legal opinion."""
    if not rows:
        return "CONDITIONAL"
    for row in rows:
        decision = row["decision"]
        if decision == "not_met":
            return "NO-GO"
        if decision == "missing" and row.get("obligation") == "qualification":
            return "NO-GO"
    if any(row["decision"] != "met" for row in rows):
        return "CONDITIONAL"
    return "GO"


def estimated_cost_inr(chunks: list[Chunk]) -> int | None:
    found: list[int] = []
    for chunk in chunks:
        for sentence in sentences(chunk.text) or [chunk.text]:
            lowered = sentence.lower()
            if any(label in lowered for label in ("estimated cost", "estimated value", "tender value", "ecpt")):
                found.extend(amounts_in(sentence))
    return max(found) if found else None


def parse_similar_alternatives(text: str) -> list[tuple[int, int]]:
    alternatives: list[tuple[int, int]] = []
    for match in _ALT_RE.finditer(text or ""):
        raw = (match.group(1) or match.group(3) or "").lower()
        count = _COUNT_WORDS.get(raw)
        amount = float(match.group(4).replace(",", ""))
        scale = (match.group(5) or "").lower()
        if scale in {"lakh", "lakhs"}:
            amount *= 100_000
        elif scale in {"crore", "crores"}:
            amount *= 10_000_000
        amount_inr = int(round(amount))
        if count and amount_inr > 0:
            alternatives.append((count, amount_inr))
    return alternatives


def _apply(text: str, hits: list[Chunk], *, estimated_cost_inr: int | None) -> dict:
    lowered = text.lower()
    if "turnover" in lowered:
        return _money_threshold(text, hits, keyword="turnover", estimated_cost_inr=estimated_cost_inr)
    if "solvency" in lowered:
        return _money_threshold(text, hits, keyword="solvency", estimated_cost_inr=estimated_cost_inr)
    if "net worth" in lowered:
        return _money_threshold(text, hits, keyword="net worth", estimated_cost_inr=estimated_cost_inr)
    if re.search(r"similar(?:\s+[a-z]+){0,4}\s+works?\b", lowered):
        return _similar(text, hits)
    if "loss" in lowered and "year" in lowered:
        return _negated_fact(
            hits,
            positive_markers=("not incurred a loss", "has not incurred", "no loss"),
            negative_markers=("incurred a loss", "incurred loss"),
            label="loss history",
        )
    if "psara" in lowered or "private security agencies" in lowered:
        return _presence(hits, ("psara", "private security agencies regulation"), "PSARA licence", qualification=True)
    if re.search(r"\bgst\b", lowered):
        return _presence(hits, ("gstin", "gst registration", "goods and services tax"), "GST registration", qualification=True)
    if re.search(r"\bpan\b", lowered) or "permanent account" in lowered:
        return _presence(hits, ("permanent account", "pan card", "pan:"), "PAN", qualification=True)
    if "blacklist" in lowered:
        return _negated_fact(
            hits,
            positive_markers=("never been blacklisted", "not been blacklisted", "has not been blacklisted"),
            negative_markers=("has been blacklisted", "was blacklisted"),
            label="blacklisting declaration",
        )
    if "manpower" in lowered or "on roll" in lowered or "security guards" in lowered:
        return _manpower(text, hits)
    if re.search(r"\biso\b", lowered):
        return _presence(hits, ("iso 9001", "iso9001", "iso certificate"), "ISO certificate", qualification=True)
    if "local supplier" in lowered or "make in india" in lowered or "class-i" in lowered or "class i" in lowered:
        return _presence(
            hits,
            ("class-i local", "class-i", "class i local supplier"),
            "Class-I local supplier declaration",
            qualification=True,
            missing_decision="unclear",
        )
    if "earnest" in lowered or re.search(r"\bemd\b", lowered) or "bid security" in lowered:
        return _emd(hits)
    return _result(
        "unclear",
        "qualification",
        "The sentence was extracted as a requirement, but no machine-readable check matched it.",
    )


def _money_threshold(text: str, hits: list[Chunk], *, keyword: str, estimated_cost_inr: int | None) -> dict:
    amount = _threshold_near(text, keyword)
    percent = _percent_near(text, keyword)
    if amount is None and percent is not None:
        if estimated_cost_inr:
            amount = int(estimated_cost_inr * percent / 100.0)
        else:
            return _result(
                "unclear",
                "qualification",
                f"{keyword.title()} is a percentage of estimated cost, and no estimated cost was read from the tender.",
            )
    evidence, quote = _best(hits, (keyword,))
    if amount is None:
        if quote:
            return _result("unclear", "qualification", f"A {keyword} clause was found but no rupee threshold was parsed.", evidence, quote)
        return _result("unclear", "qualification", f"A {keyword} clause was found but no rupee threshold was parsed.")
    if not quote:
        return _result("missing", "qualification", f"No uploaded document states {keyword}.")
    evidence_amounts = amounts_in(quote)
    if not evidence_amounts and evidence is not None:
        evidence_amounts = amounts_in(evidence.text)
    if not evidence_amounts:
        return _result("unclear", "qualification", f"{keyword.title()} evidence has no parsed rupee amount.", evidence, quote)
    observed = max(evidence_amounts)
    if observed >= amount:
        return _result(
            "met",
            "qualification",
            f"Cited {keyword} of Rs. {observed:,} meets the threshold of Rs. {amount:,}.",
            evidence,
            quote,
        )
    return _result(
        "not_met",
        "qualification",
        f"Cited {keyword} of Rs. {observed:,} is below the threshold of Rs. {amount:,}.",
        evidence,
        quote,
    )


def _similar(text: str, hits: list[Chunk]) -> dict:
    alternatives = parse_similar_alternatives(text)
    if not alternatives:
        return _result("unclear", "qualification", "A similar-work clause was found but the count and amount were not parsed.")
    works = _work_chunks(hits, text.lower())
    if not works:
        return _result("missing", "qualification", "No uploaded work completion or work order matches this similar-work clause.")
    for count, amount in alternatives:
        qualifying = [item for item in works if item[0] >= amount]
        if len(qualifying) >= count:
            evidence, quote = qualifying[0][1], qualifying[0][2]
            return _result(
                "met",
                "qualification",
                f"{len(qualifying)} cited work(s) meet the alternative of {count} work(s) at Rs. {amount:,}.",
                evidence,
                quote,
            )
    best = max(works, key=lambda item: item[0])
    return _result(
        "not_met",
        "qualification",
        f"The strongest cited similar work is Rs. {best[0]:,}, which does not meet any stated alternative.",
        best[1],
        best[2],
    )


def _work_chunks(hits: list[Chunk], requirement: str) -> list[tuple[int, Chunk, str]]:
    found: list[tuple[int, Chunk, str]] = []
    for chunk in hits:
        quote = _sentence_with(chunk, ("work", "completion", "order"))
        if quote is None:
            continue
        if not _domain_ok(requirement, quote.lower()):
            continue
        amounts = amounts_in(quote) or amounts_in(chunk.text)
        if not amounts:
            continue
        found.append((max(amounts), chunk, quote))
    return found


def _domain_ok(requirement: str, evidence: str) -> bool:
    requested = [domain for domain in _DOMAINS if domain in requirement]
    if not requested:
        return True
    return any(domain in evidence for domain in requested)


def _manpower(text: str, hits: list[Chunk]) -> dict:
    required = _headcount(text)
    evidence, quote = _best(hits, ("manpower", "on roll", "security guards", "personnel"))
    if required is None:
        return _result("unclear", "qualification", "A manpower clause was found but no headcount was parsed.", evidence, quote)
    if not quote:
        return _result("missing", "qualification", "No uploaded document states manpower on roll.")
    observed = _headcount(quote) or _headcount(evidence.text if evidence else "")
    if observed is None:
        return _result("unclear", "qualification", "Manpower evidence has no parsed headcount.", evidence, quote)
    if observed >= required:
        return _result("met", "qualification", f"Cited manpower of {observed} meets the stated {required}.", evidence, quote)
    return _result("not_met", "qualification", f"Cited manpower of {observed} is below the stated {required}.", evidence, quote)


def _headcount(text: str) -> int | None:
    if not text:
        return None
    match = re.search(
        r"(?:not less than|not be less than|at least|minimum of|minimum)\s+(\d{2,5})",
        text,
        re.IGNORECASE,
    )
    if match:
        return int(match.group(1))
    match = re.search(r"(?:manpower on roll is|on roll is)\s+(\d{1,5})", text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    match = re.search(r"(\d{2,5})\s+security guards", text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def _emd(hits: list[Chunk]) -> dict:
    evidence, quote = _best(hits, ("demand draft", "earnest money deposit", "emd", "bid security"))
    if quote and any(word in quote.lower() for word in ("demand draft", "bank guarantee", "payment", "remitted", "transferred")):
        return _result("met", "submission", "A company document cites an earnest-money instrument.", evidence, quote)
    return _result(
        "missing",
        "submission",
        "No earnest-money instrument was in the uploaded company documents. This is a submission item, not a capacity test.",
    )


def _presence(hits: list[Chunk], markers: tuple[str, ...], label: str, *, qualification: bool, missing_decision: str = "missing") -> dict:
    evidence, quote = _best(hits, markers)
    obligation = "qualification" if qualification else "submission"
    if quote:
        return _result("met", obligation, f"The uploaded documents include a {label}.", evidence, quote)
    return _result(missing_decision, obligation, f"No uploaded document shows a {label}.")


def _negated_fact(hits: list[Chunk], *, positive_markers: tuple[str, ...], negative_markers: tuple[str, ...], label: str) -> dict:
    evidence, quote = _best(hits, positive_markers + negative_markers)
    if not quote:
        return _result("missing", "qualification", f"No uploaded document speaks to {label}.")
    lowered = quote.lower()
    if any(marker in lowered for marker in positive_markers):
        return _result("met", "qualification", f"The cited document supports the {label} requirement.", evidence, quote)
    if any(marker in lowered for marker in negative_markers):
        return _result("not_met", "qualification", f"The cited document contradicts the {label} requirement.", evidence, quote)
    return _result("unclear", "qualification", f"A document mentions {label} but the sentence is not decisive.", evidence, quote)


def _best(hits: list[Chunk], markers: tuple[str, ...]) -> tuple[Chunk | None, str | None]:
    for chunk in hits:
        quote = _sentence_with(chunk, markers)
        if quote:
            return chunk, quote
    return None, None


def _sentence_with(chunk: Chunk, markers: tuple[str, ...]) -> str | None:
    for sentence in sentences(chunk.text):
        lowered = sentence.lower()
        if any(marker in lowered for marker in markers):
            return sentence.strip()
    folded = chunk.text.strip()
    if len(folded) <= 900 and any(marker in folded.lower() for marker in markers):
        return folded
    return None


def _threshold_near(text: str, keyword: str) -> int | None:
    lowered = text.lower()
    index = lowered.find(keyword)
    window = text if index < 0 else text[max(0, index - 90) : index + 180]
    amounts = amounts_in(window) or amounts_in(text)
    return max(amounts) if amounts else None


def _percent_near(text: str, keyword: str) -> float | None:
    lowered = text.lower()
    index = lowered.find(keyword)
    window = text if index < 0 else text[max(0, index - 80) : index + 100]
    match = re.search(r"(\d{1,3}(?:\.\d+)?)\s*(?:%|percent)", window, re.IGNORECASE)
    if not match:
        return None
    value = float(match.group(1))
    if value <= 0 or value > 100:
        return None
    return value


def _result(
    decision: str,
    obligation: str,
    rationale: str,
    chunk: Chunk | None = None,
    quote: str | None = None,
) -> dict:
    payload = {"decision": decision, "obligation": obligation, "rationale": rationale}
    if chunk is not None and quote:
        payload.update(
            {
                "quote": quote,
                "chunk_id": chunk.id,
                "filename": chunk.filename,
                "page": chunk.page_start,
                "clause": chunk.clause,
            }
        )
    return payload


def worst(decisions: list[str]) -> str:
    return max(decisions, key=lambda item: _DECISION_RANK.get(item, 0))
