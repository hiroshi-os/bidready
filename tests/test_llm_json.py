import json

import pytest

from bidready.llm import _parse_json, _repair_json


def test_parse_json_accepts_a_complete_object():
    assert _parse_json('{"risks": [], "deadlines": []}') == {"risks": [], "deadlines": []}


def test_repair_json_keeps_complete_items_when_the_reply_is_cut_off():
    text = """{
      "risks": [
        {"quote": "forfeit 50 percent of the earnest money", "chunk_id": "a-1", "severity": "high"},
        {"quote": "liquidated damages of 0.5 percent", "chunk_id": "a-2", "severity": "medium"}
      ],
      "deadlines": [
        {"quote": "Bid Submission End Date : 15/10/2026", "chunk_id": "a-4", "event": "submission"},
        {"quote": "Pre-bid meeting on
    """
    parsed = _repair_json(text)
    assert len(parsed["risks"]) == 2
    assert len(parsed["deadlines"]) == 1
    assert parsed["deadlines"][0]["event"] == "submission"


def test_parse_json_repairs_a_truncated_object():
    text = '{"requirements": [{"quote": "turnover of Rs. 50 lakh", "chunk_id": "c-1"}, {"quote": "gst'
    parsed = _parse_json(text)
    assert parsed["requirements"][0]["chunk_id"] == "c-1"
    assert len(parsed["requirements"]) == 1


def test_repair_json_rejects_text_with_no_object():
    with pytest.raises(json.JSONDecodeError):
        _repair_json("not json at all")
