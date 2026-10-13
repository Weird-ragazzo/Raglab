# Evaluation protocol and dataset

The JSONL schema uses one object per question: `id`, `split`, `question`, `reference_answer`, `relevant_evidence_ids`, `answerable`, and `source`. Its JSON Schema is in `data/evaluation/schema.json`. Dev is for tuning; heldout is reserved for one final comparison after settings are selected. Each entry is manually checked against the cited article. The small starter set is a smoke benchmark, not a general claim about RAG quality.

Recall@K is the share of labeled relevant evidence returned within K. Precision@K is relevant returned evidence divided by K. MRR is the inverse rank of the first relevant result. Retrieval latency includes query embedding and FAISS search; end-to-end latency includes Gemini generation. Citation validity is the fraction of citations that exactly match source, page, and chunk IDs in retrieved results. Abstention accuracy checks whether unanswerable questions are refused. Reference-answer lexical coverage is only an automatic overlap proxy; it is not ground truth and should be reviewed by a person. Generation metrics are null and marked `not_run` unless `--generate` is used.

Run retrieval only (no paid generation calls):

```bash
.venv/bin/python -m src.evaluation.runner --split dev --compare
```

`--compare` varies chunk size, overlap, and top-K one setting at a time. Use `--sweep top_k` (or `chunk_size` / `chunk_overlap`) to run only one parameter family.

Select settings using dev, then evaluate once on heldout:

```bash
.venv/bin/python -m src.evaluation.runner --split heldout --chunk-size 900 --chunk-overlap 150 --top-k 5
```

Add `--generate` to include Gemini metrics. Results are timestamped JSON under `data/evaluation/results/`; they are ignored by Git. No baseline or improvement numbers are asserted until a run produces them.

## Provenance and license

The starter benchmark is manually derived from Iyer et al. (2025), “Careful design of Large Language Model pipelines enables expert-level retrieval of evidence-based information from syntheses and databases,” PLOS ONE 20(5), e0323563, DOI [10.1371/journal.pone.0323563](https://doi.org/10.1371/journal.pone.0323563). PLOS marks the article as available under its Creative Commons Attribution License; retain attribution when redistributing source-derived material. The benchmark records compact paraphrases and answer keys, not the complete article. Evidence IDs refer to curated section excerpts in the local text corpus. They are suitable for retrieval comparisons, while they do not provide the article's PDF page numbers. Use uploaded PDFs to evaluate true page-level citation behavior.

