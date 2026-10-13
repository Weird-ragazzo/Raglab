"""Deterministic retrieval, latency, and citation metrics."""

from __future__ import annotations


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    return len(set(retrieved[:k]) & relevant) / len(relevant) if relevant else 0.0


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    return len(set(retrieved[:k]) & relevant) / k if k > 0 else 0.0


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    return next((1.0 / rank for rank, item in enumerate(retrieved, 1) if item in relevant), 0.0)


def citation_validity(citations: list[dict[str, object]], retrieved: list[dict[str, object]]) -> float | None:
    if not citations:
        return None
    allowed = {(x.get("source"), x.get("page"), x.get("chunk_id")) for x in retrieved}
    valid = sum((c.get("source"), c.get("page"), c.get("chunk_id")) in allowed for c in citations)
    return valid / len(citations)


def reference_answer_match(answer: str, reference: str) -> float:
    """Transparent lexical coverage rubric; human review remains authoritative."""
    tokens = {word.lower().strip(".,:;()[]{}") for word in reference.split() if len(word) > 2}
    if not tokens:
        return 0.0
    answer_tokens = {word.lower().strip(".,:;()[]{}") for word in answer.split()}
    return len(tokens & answer_tokens) / len(tokens)


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None

