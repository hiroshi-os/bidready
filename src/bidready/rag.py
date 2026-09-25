"""Hybrid retrieval: BM25, a FAISS inner-product index, reciprocal rank fusion, then a reranker."""

from __future__ import annotations

import zlib
from typing import Protocol

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from bidready.parsing import Chunk
from bidready.textutil import tokenize


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray: ...


class HashEmbedder:
    """Feature hashing into a fixed vector. Lexical, deterministic, and offline.

    This is the mock embedding path. It is not a semantic model. The FAISS index
    and the hybrid fusion code are the same ones used with a real encoder.
    """

    name = "feature-hash-384"
    dim = 384

    def embed(self, texts: list[str]) -> np.ndarray:
        rows = np.zeros((len(texts), self.dim), dtype=np.float32)
        for index, text in enumerate(texts):
            for token in tokenize(text):
                digest = zlib.crc32(token.encode()) & 0xFFFFFFFF
                sign = 1.0 if digest & 0x100 else -1.0
                rows[index, digest % self.dim] += sign
        return l2_normalize(rows)


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "EMBEDDING_PROVIDER=sentence-transformers requires the local extra: "
                'pip install -e ".[local]"'
            ) from exc
        self.model_name = model_name
        self._model = SentenceTransformer(model_name)
        self.dim = int(self._model.get_sentence_embedding_dimension())
        self.name = model_name

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vectors, dtype=np.float32)


class Reranker(Protocol):
    name: str

    def rerank(self, query: str, chunks: list[Chunk], k: int) -> list[Chunk]: ...


class LexicalReranker:
    """Character-trigram Jaccard. A second stage that can reorder BM25 hits without a model."""

    name = "lexical-trigram-jaccard"

    def rerank(self, query: str, chunks: list[Chunk], k: int) -> list[Chunk]:
        query_grams = _trigrams(query)
        scored: list[tuple[float, int, Chunk]] = []
        for index, chunk in enumerate(chunks):
            grams = _trigrams(chunk.text[:2000])
            if not query_grams or not grams:
                score = 0.0
            else:
                score = len(query_grams & grams) / len(query_grams | grams)
            scored.append((score, -index, chunk))
        scored.sort(reverse=True)
        return [chunk for _, _, chunk in scored[:k]]


class CrossEncoderReranker:
    def __init__(self, model_name: str) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RuntimeError(
                "RERANKER=cross-encoder requires the local extra: pip install -e \".[local]\""
            ) from exc
        self.model_name = model_name
        self.name = model_name
        self._model = CrossEncoder(model_name)

    def rerank(self, query: str, chunks: list[Chunk], k: int) -> list[Chunk]:
        if not chunks:
            return []
        pairs = [(query, chunk.text[:1800]) for chunk in chunks]
        scores = np.asarray(self._model.predict(pairs))
        order = np.argsort(-scores)
        return [chunks[int(index)] for index in order[:k]]


def _trigrams(text: str) -> set[str]:
    folded = " ".join((text or "").lower().split())
    if len(folded) < 3:
        return set()
    return {folded[index : index + 3] for index in range(len(folded) - 2)}


class HybridIndex:
    def __init__(self, chunks: list[Chunk], embedder: Embedder, reranker: Reranker) -> None:
        self.chunks = {chunk.id: chunk for chunk in chunks}
        self.order = list(chunks)
        self.embedder = embedder
        self.reranker = reranker
        self._bm25: BM25Okapi | None = None
        self._index: faiss.Index | None = None
        if not chunks:
            return
        tokens = [tokenize(chunk.text) or ["empty"] for chunk in chunks]
        self._bm25 = BM25Okapi(tokens)
        matrix = l2_normalize(np.asarray(embedder.embed([chunk.text for chunk in chunks]), dtype=np.float32))
        self._index = faiss.IndexFlatIP(matrix.shape[1])
        self._index.add(matrix)

    def search(self, query: str, k: int = 8, mode: str = "hybrid_rerank") -> list[Chunk]:
        if not self.order or not query or not query.strip():
            return []
        limit = min(k, len(self.order))
        if mode == "vector":
            return self._vector(query, limit)
        if mode == "bm25":
            return self._bm25_top(query, limit)
        pool = min(len(self.order), max(limit, 30 if mode == "hybrid_rerank" else limit))
        fused = self._rrf(query, pool)
        if mode == "hybrid":
            return fused[:limit]
        if mode == "hybrid_rerank":
            return self.reranker.rerank(query, fused, limit)
        raise ValueError(f"unknown retrieval mode: {mode}")

    def _vector(self, query: str, k: int) -> list[Chunk]:
        if self._index is None:
            return []
        query_vector = l2_normalize(np.asarray(self.embedder.embed([query]), dtype=np.float32))
        _, ids = self._index.search(query_vector, k)
        found: list[Chunk] = []
        for raw in ids[0]:
            if int(raw) < 0:
                continue
            found.append(self.order[int(raw)])
        return found

    def _bm25_top(self, query: str, k: int) -> list[Chunk]:
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        order = np.argsort(-np.asarray(scores))
        return [self.order[int(index)] for index in order[:k]]

    def _rrf(self, query: str, k: int, rrf_k: int = 60) -> list[Chunk]:
        scores: dict[str, float] = {}
        for rank, chunk in enumerate(self._vector(query, k), start=1):
            scores[chunk.id] = scores.get(chunk.id, 0.0) + 1.0 / (rrf_k + rank)
        for rank, chunk in enumerate(self._bm25_top(query, k), start=1):
            scores[chunk.id] = scores.get(chunk.id, 0.0) + 1.0 / (rrf_k + rank)
        ordered = sorted(scores, key=lambda chunk_id: scores[chunk_id], reverse=True)
        return [self.chunks[chunk_id] for chunk_id in ordered[:k]]


def build_embedder(provider: str, model_name: str) -> Embedder:
    if provider == "hash":
        return HashEmbedder()
    if provider in {"sentence-transformers", "minilm", "local"}:
        return SentenceTransformerEmbedder(model_name)
    raise RuntimeError(f"unknown EMBEDDING_PROVIDER: {provider}")


def build_reranker(name: str, model_name: str) -> Reranker:
    if name == "lexical":
        return LexicalReranker()
    if name == "cross-encoder":
        return CrossEncoderReranker(model_name)
    raise RuntimeError(f"unknown RERANKER: {name}")
