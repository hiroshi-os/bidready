from bidready.parsing import Chunk
from bidready.rag import HashEmbedder, HybridIndex, LexicalReranker


def _chunk(index: int, text: str) -> Chunk:
    return Chunk(
        id=f"c{index}",
        document_id="d",
        filename="nit.pdf",
        role="tender",
        page_start=index,
        page_end=index,
        clause=str(index),
        section=None,
        text=text,
    )


def test_hybrid_retrieval_finds_the_turnover_clause():
    chunks = [
        _chunk(1, "The site office shall remain open on working days for inspection of materials."),
        _chunk(2, "Average annual financial turnover on construction works shall be at least Rs. 22,60,000."),
        _chunk(3, "Drawings are attached as annexure and do not change the bill of quantities."),
    ]
    index = HybridIndex(chunks, HashEmbedder(), LexicalReranker())
    query = "minimum average annual turnover for construction works"
    for mode in ("bm25", "vector", "hybrid", "hybrid_rerank"):
        hits = index.search(query, k=2, mode=mode)
        assert hits, mode
        assert any("22,60,000" in hit.text for hit in hits), mode
