import os
from pathlib import Path

import pymupdf
import pytest

os.environ.setdefault("LLM_PROVIDER", "mock")
os.environ.setdefault("EMBEDDING_PROVIDER", "hash")
os.environ.setdefault("RERANKER", "lexical")

from bidready.config import Settings
from bidready.main import create_app

TENDER_PAGE_1 = """NOTICE INVITING TENDER
Sample Municipal Works
Clause 1. Eligibility
1.1 The bidder shall have a valid GST registration certificate.
1.2 Average annual financial turnover shall be at least Rs. 50,00,000 during the last three financial years.
"""

TENDER_PAGE_2 = """Clause 2. Earnest Money Deposit
1.3 The bidder should have completed one similar civil work of value not less than Rs. 20,00,000.
2.1 Earnest Money Deposit of Rs. 1,00,000 shall be submitted with the bid.
2.2 Bid submission end date is 15-Oct-2026 up to 15:00 hours.
2.3 Pre-bid meeting will be held on 01-Oct-2026 at 11:00 hours.
Clause 3. Penalties
3.1 Liquidated damages at 0.5 percent per week shall be levied for delay.
"""


def write_text_pdf(path: Path, pages: list[str]) -> None:
    document = pymupdf.open()
    for text in pages:
        page = document.new_page()
        y = 72
        for line in text.splitlines():
            page.insert_text((72, y), line, fontsize=11)
            y += 16
    document.save(path)
    document.close()


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    data = tmp_path / "data"
    data.mkdir()
    return Settings(
        database_url=f"sqlite:///{tmp_path / 'test.db'}",
        data_dir=data,
        llm_provider="mock",
        llm_model="",
        openai_api_key="",
        openai_base_url="https://api.openai.com/v1",
        ollama_base_url="http://127.0.0.1:11434",
        embedding_provider="hash",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        reranker="lexical",
        reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        prompt_version="v1",
        webhook_url="",
        smtp_host="",
        smtp_port=587,
        smtp_from="",
        smtp_to="",
        tenderlens_base_url="",
        document_store="local",
        s3_endpoint_url="",
        s3_bucket="",
        s3_access_key="",
        s3_secret_key="",
        s3_region="us-east-1",
    )


@pytest.fixture
def app(settings: Settings):
    return create_app(settings)


@pytest.fixture
def tender_pdf(tmp_path: Path) -> Path:
    path = tmp_path / "sample-nit.pdf"
    write_text_pdf(path, [TENDER_PAGE_1, TENDER_PAGE_2])
    return path
