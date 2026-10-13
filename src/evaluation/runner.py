"""Run retrieval-only or end-to-end benchmark configurations."""

from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.config import ROOT, settings
from src.documents import Chunk, PageText, chunk_pages
from src.evaluation.metrics import (
    citation_validity, mean, precision_at_k, recall_at_k, reciprocal_rank,
    reference_answer_match,
)
from src.generation import answer_question
from src.retrieval import VectorIndex, embed_texts

DATASET = ROOT / "data" / "evaluation" / "benchmark.jsonl"
DOCS = ROOT / "data" / "evaluation" / "corpus"
RESULTS = ROOT / "data" / "evaluation" / "results"


def run(config: dict[str, int | str], split: str = "dev", generate: bool = False) -> dict[str, object]:
    entries = [json.loads(line) for line in DATASET.read_text().splitlines() if line.strip()]
    entries = [entry for entry in entries if entry["split"] == split]
    sources = sorted({source for entry in entries for source in entry["relevant_evidence_ids"]})
    pages: list[PageText] = []
    for source in sources:
        path = DOCS / f"{source}.txt"
        text = path.read_text(encoding="utf-8")
        pages.append(PageText(source, 1, text))
    chunks = chunk_pages(
        pages, chunk_size=int(config["chunk_size"]), overlap=int(config["chunk_overlap"])
    )
    vectors = embed_texts([c.text for c in chunks], settings.embedding_model)
    index = VectorIndex(chunks, vectors)
    query_vectors: dict[str, object] = {}
    # Embeddings are calculated per query so query timings include model encode time.
    recalls: list[float] = []
    precisions: list[float] = []
    mrrs: list[float] = []
    retrieval_times: list[float] = []
    e2e_times: list[float] = []
    citation_scores: list[float] = []
    correctness: list[float] = []
    abstention: list[float] = []
    for entry in entries:
        start = time.perf_counter()
        query_vector = embed_texts([entry["question"]], settings.embedding_model)[0]
        retrieved = index.search(query_vector, int(config["top_k"]))
        retrieval_ms = (time.perf_counter() - start) * 1000
        retrieval_times.append(retrieval_ms)
        if entry["answerable"]:
            relevant = set(entry["relevant_evidence_ids"])
            ids = [r.chunk.source for r in retrieved]
            recalls.append(recall_at_k(ids, relevant, int(config["top_k"])))
            precisions.append(precision_at_k(ids, relevant, int(config["top_k"])))
            mrrs.append(reciprocal_rank(ids, relevant))
        if generate:
            answer = answer_question(entry["question"], retrieved)
            e2e_times.append((time.perf_counter() - start) * 1000)
            retrieved_dicts = [
                {"source": x.chunk.source, "page": x.chunk.page, "chunk_id": x.chunk.chunk_id}
                for x in retrieved
            ]
            citation_score = citation_validity(answer.citations, retrieved_dicts)
            if citation_score is not None:
                citation_scores.append(citation_score)
            if entry["answerable"]:
                correctness.append(reference_answer_match(answer.answer, entry["reference_answer"]))
            abstention.append(float(answer.abstained == (not entry["answerable"])))
    return {
        "dataset": {"path": str(DATASET.relative_to(ROOT)), "version": "1.0.0", "split": split, "questions": len(entries)},
        "config": config,
        "metrics": {
            "recall_at_k": mean(recalls), "mrr": mean(mrrs), "precision_at_k": mean(precisions),
            "retrieval_questions": len(recalls),
            "retrieval_latency_ms_mean": mean(retrieval_times),
            "retrieval_latency_ms_p95": float(np.percentile(retrieval_times, 95)) if retrieval_times else None,
            "end_to_end_latency_ms_mean": mean(e2e_times) if generate else None,
            "citation_validity": mean(citation_scores) if generate else None,
            "reference_answer_lexical_coverage": mean(correctness) if generate else None,
            "abstention_accuracy": mean(abstention) if generate else None,
            "generation_status": "run" if generate else "not_run",
        },
        "rubric": "Reference answer lexical token coverage is an automatic proxy only; human judgment is authoritative.",
        "runtime": {
            "python": platform.python_version(),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "embedding_model": settings.embedding_model,
            "gemini_model": settings.gemini_model if generate else None,
            "retrieval": "faiss IndexFlatIP with cosine similarity",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["dev", "heldout"], default="dev")
    parser.add_argument("--chunk-size", type=int, default=900)
    parser.add_argument("--chunk-overlap", type=int, default=150)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--generate", action="store_true", help="Call Gemini and measure answer metrics")
    parser.add_argument("--compare", action="store_true", help="Compare the baseline with one-variable-at-a-time alternatives")
    parser.add_argument("--sweep", choices=["chunk_size", "chunk_overlap", "top_k"],
                        help="Vary only this setting around the selected baseline")
    args = parser.parse_args()
    if args.compare and args.sweep:
        parser.error("Use either --compare or --sweep, not both")
    configs = [
        {"name": "baseline", "chunk_size": args.chunk_size, "chunk_overlap": args.chunk_overlap, "top_k": args.top_k},
    ]
    if args.compare:
        configs.extend([
            {"name": "chunk-size-700", "chunk_size": 700, "chunk_overlap": args.chunk_overlap, "top_k": args.top_k},
            {"name": "chunk-size-1200", "chunk_size": 1200, "chunk_overlap": args.chunk_overlap, "top_k": args.top_k},
            {"name": "chunk-overlap-100", "chunk_size": args.chunk_size, "chunk_overlap": 100, "top_k": args.top_k},
            {"name": "chunk-overlap-250", "chunk_size": args.chunk_size, "chunk_overlap": 250, "top_k": args.top_k},
            {"name": "top-k-3", "chunk_size": args.chunk_size, "chunk_overlap": args.chunk_overlap, "top_k": 3},
            {"name": "top-k-8", "chunk_size": args.chunk_size, "chunk_overlap": args.chunk_overlap, "top_k": 8},
        ])
    elif args.sweep:
        values = {
            "chunk_size": (700, 1200),
            "chunk_overlap": (100, 250),
            "top_k": (3, 8),
        }[args.sweep]
        for value in values:
            config = {"name": f"{args.sweep}-{value}", "chunk_size": args.chunk_size,
                      "chunk_overlap": args.chunk_overlap, "top_k": args.top_k}
            config[args.sweep] = value
            configs.append(config)
    RESULTS.mkdir(parents=True, exist_ok=True)
    for config in configs:
        result = run(config, args.split, args.generate)
        path = RESULTS / f"{args.split}-{config['name']}-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
        path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"saved": str(path.relative_to(ROOT)), "metrics": result["metrics"]}, indent=2))


if __name__ == "__main__":
    main()

