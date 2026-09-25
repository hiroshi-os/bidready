from fastapi.testclient import TestClient


def test_health_and_upload_flow(app, tender_pdf):
    client = TestClient(app)
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["llm_provider"] == "mock"

    home = client.get("/")
    assert home.status_code == 200
    assert "Not legal advice" in home.text
    assert "sample-civil" in home.text

    response = client.post(
        "/api/cases",
        data={"profile_id": "sample-civil", "title": "API sample"},
        files=[("tender_files", ("sample-nit.pdf", tender_pdf.read_bytes(), "application/pdf"))],
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["report"]["go_no_go"] == "CONDITIONAL"
    case_id = body["case"]["id"]

    page = client.get(f"/cases/{case_id}")
    assert page.status_code == 200
    assert "CONDITIONAL" in page.text
    assert "Compliance matrix" in page.text
    assert "Draft pre-bid query letter" in page.text
    assert "Agent trace" in page.text
