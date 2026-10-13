"""Local sentence-transformer embeddings and FAISS cosine retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from time import perf_counter
from typing import Protocol

import faiss
import numpy as np

from src.documents import Chunk


class Encoder(Protocol):
    def encode(self, texts: list[str], **kwargs: object) -> np.ndarray: ...


@lru_cache(maxsize=2)
def load_encoder(model_name: str) -> Encoder:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name, device="cpu")


def embed_texts(texts: list[str], model_name: str) -> np.ndarray:
    if not texts:
        return np.empty((0, 384), dtype="float32")
    vectors = load_encoder(model_name).encode(
        texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
    )
    return np.asarray(vectors, dtype="float32")


@dataclass
class SearchResult:
    chunk: Chunk
    score: float
    rank: int


class VectorIndex:
    def __init__(self, chunks: list[Chunk], vectors: np.ndarray):
        if len(chunks) != len(vectors):
            raise ValueError("Every chunk must have exactly one vector")
        self.chunks = chunks
        self.dimension = int(vectors.shape[1]) if vectors.ndim == 2 else 0
        self.index: faiss.IndexFlatIP | None = None
        if len(vectors):
            normalized = np.asarray(vectors, dtype="float32").copy()
            faiss.normalize_L2(normalized)
            self.index = faiss.IndexFlatIP(self.dimension)
            self.index.add(normalized)

    @property
    def size(self) -> int:
        return len(self.chunks)

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> list[SearchResult]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.index is None:
            return []
        query = np.asarray(query_vector, dtype="float32").reshape(1, -1).copy()
        if query.shape[1] != self.dimension:
            raise ValueError("Query embedding dimension does not match index")
        faiss.normalize_L2(query)
        scores, positions = self.index.search(query, min(top_k, self.size))
        return [
            SearchResult(self.chunks[int(i)], float(score), rank)
            for rank, (score, i) in enumerate(zip(scores[0], positions[0]), start=1)
            if i >= 0
        ]


def retrieve(
    query: str, chunks: list[Chunk], *, top_k: int = 5,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> tuple[list[SearchResult], float]:
    """Small convenience path for evaluation; apps should reuse their index."""
    started = perf_counter()
    vectors = embed_texts([chunk.text for chunk in chunks], model_name)
    index = VectorIndex(chunks, vectors)
    query_vector = embed_texts([query], model_name)[0]
    results = index.search(query_vector, top_k)
    return results, (perf_counter() - started) * 1000

