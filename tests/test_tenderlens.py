import base64

import httpx
import pytest

from bidready.tenderlens import TenderlensAdapter, TenderlensError, TenderlensNotConfigured


def test_unconfigured_adapter_does_not_call_the_network():
    adapter = TenderlensAdapter("")
    with pytest.raises(TenderlensNotConfigured):
        adapter.fetch("anything")


def test_adapter_reads_the_documented_json_contract():
    pdf = b"%PDF-1.4 synthetic"
    payload = {
        "id": "tl-1",
        "title": "Sample from tenderlens",
        "source_url": "https://example.test/tender",
        "documents": [
            {"filename": "nit.pdf", "content_base64": base64.b64encode(pdf).decode()}
        ],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/tenders/tl-1/"
        return httpx.Response(200, json=payload)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    pack = TenderlensAdapter("http://tenderlens.test", client=client).fetch("tl-1")
    assert pack.title == "Sample from tenderlens"
    assert pack.documents[0].content.startswith(b"%PDF")


def test_adapter_rejects_a_document_without_bytes():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"id": "x", "title": "x", "documents": [{"filename": "a.pdf"}]})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    with pytest.raises(TenderlensError):
        TenderlensAdapter("http://tenderlens.test", client=client).fetch("x")
