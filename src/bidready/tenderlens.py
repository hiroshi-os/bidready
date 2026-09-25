"""Optional adapter for hiroshi-os/tenderlens.

tenderlens is a separate crawler and is still a stub repository. bidready does not
import it and does not require it. When TENDERLENS_BASE_URL is set, this client
expects the JSON contract below.

GET {TENDERLENS_BASE_URL}/api/tenders/{tender_id}/

{
  "id": "string",
  "title": "string",
  "source_url": "string",
  "documents": [
    {"filename": "nit.pdf", "url": "https://..."}
  ]
}

A document may instead carry "content_base64" so a local tenderlens can hand over
bytes without a second hop. Unknown shapes raise TenderlensError.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass

import httpx


class TenderlensNotConfigured(RuntimeError):
    pass


class TenderlensError(RuntimeError):
    pass


@dataclass
class TenderDocument:
    filename: str
    content: bytes


@dataclass
class TenderPack:
    tender_id: str
    title: str
    source_url: str
    documents: list[TenderDocument]


class TenderlensAdapter:
    def __init__(self, base_url: str = "", client: httpx.Client | None = None) -> None:
        self.base_url = (base_url or "").rstrip("/")
        self._client = client

    @property
    def configured(self) -> bool:
        return bool(self.base_url)

    def fetch(self, tender_id: str) -> TenderPack:
        if not self.configured:
            raise TenderlensNotConfigured(
                "TENDERLENS_BASE_URL is unset. bidready analyses uploads on its own; "
                "the tenderlens crawler is optional."
            )
        client = self._client or httpx.Client(timeout=30)
        close = self._client is None
        try:
            response = client.get(f"{self.base_url}/api/tenders/{tender_id}/")
            response.raise_for_status()
            payload = response.json()
            documents = []
            for item in payload.get("documents") or []:
                filename = item.get("filename") or "tender.pdf"
                if item.get("content_base64"):
                    content = base64.b64decode(item["content_base64"])
                elif item.get("url"):
                    downloaded = client.get(item["url"])
                    downloaded.raise_for_status()
                    content = downloaded.content
                else:
                    raise TenderlensError(f"document {filename} has neither url nor content_base64")
                documents.append(TenderDocument(filename=filename, content=content))
            if not documents:
                raise TenderlensError("tenderlens returned no documents")
            return TenderPack(
                tender_id=str(payload.get("id") or tender_id),
                title=str(payload.get("title") or tender_id),
                source_url=str(payload.get("source_url") or ""),
                documents=documents,
            )
        finally:
            if close:
                client.close()
