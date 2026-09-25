from evals.harness import load_manifest

from bidready.synthetic import get_profile

DECISIONS = {"met", "not_met", "missing", "unclear"}


def test_gold_manifest_schema():
    manifest = load_manifest()
    assert len(manifest["tenders"]) >= 5
    ids = [tender["id"] for tender in manifest["tenders"]]
    assert len(ids) == len(set(ids))
    for tender in manifest["tenders"]:
        assert tender["source_url"].startswith("http")
        assert tender["sha256"]
        assert "not committed" in tender["license_note"].lower() or "not redistributed" in tender["license_note"].lower() or "fetch" in tender["license_note"].lower()
        requirement_ids = {item["id"] for item in tender["requirements"]}
        assert requirement_ids
        for query in tender["retrieval_queries"]:
            assert query["query"]
            assert query["match_all"]
        for label in tender["eligibility"]:
            assert label["requirement_id"] in requirement_ids
            assert label["expected"] in DECISIONS
            profile = get_profile(label["profile"])
            assert profile is not None
            assert profile["synthetic"] is True
