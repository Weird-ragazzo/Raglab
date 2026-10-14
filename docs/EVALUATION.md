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

## Real PDF evaluation

### Input and exact run order

Put 3-4 text-based research PDFs in `data/evaluation/pdfs/`. Scanned PDFs are unsupported because OCR is not included. Each basename (including `.pdf`) is the stable source ID stored in evidence and citations. Gemini is used first for benchmark generation; NVIDIA NIM is used if Gemini errors. Only use files you are authorized to process with those services.

Set `GEMINI_API_KEY` and `NVIDIA_API_KEY` in `.env`, then run the following from the repository root. NVIDIA is the fallback if Gemini is unavailable or rate-limited:

```bash
.venv/bin/python -m src.evaluation.generate_benchmark --pdf-dir data/evaluation/pdfs --per-pdf 12
# Read data/evaluation/real/review.md and correct records where needed; change verified_by from auto to human after review.
.venv/bin/python -m src.evaluation.real_runner --split dev --compare
.venv/bin/python -m src.evaluation.real_runner --split heldout --generate --judge
```

The generator samples page excerpts deterministically, requests structured JSON, and checks each quoted evidence span against its stated page after whitespace normalization. Gemini is primary; NVIDIA NIM is the fallback. Exact duplicate questions, missing quotes, invalid evidence, and reference answers repeated in their questions are rejected and counted in `metadata.json`. `benchmark_real.jsonl` is validated against `real_schema.json`; `metadata.json` records the random seed, requested and used models/providers, PDF names, split counts, rejection counts, and a content-derived dataset version. Output is not replaced unless `--force` is supplied.

Dev comparisons sweep the Cartesian product of chunk sizes 700/900/1200, overlaps 100/150/250, and top-K 3/5/8. `--compare` is rejected for heldout. Choose the configuration using dev, then specify that configuration on the single heldout command; do not use heldout to tune. Add `--limit N` or `--category factual` for a quick diagnostic run. `--repeat N` runs each selected case N times and records each attempt.

`--generate` calls the existing production answer function. Gemini is primary; NVIDIA NIM answers if Gemini errors. `--judge` requires `--generate` and requests structured, temperature-zero scores. The judge defaults to `gemini-2.5-flash`, choosing `gemini-2.5-pro` instead if that is the configured generator model; set `GEMINI_JUDGE_MODEL` or pass `--judge-model` to choose another Gemini model. If Gemini judging fails, the separately configured `NVIDIA_JUDGE_MODEL` is used. The actual provider and model are recorded per case. Calls retry transient rate-limit/server errors with exponential backoff; unrecoverable errors remain in case results rather than being turned into scores.

### Real-PDF metric definitions

- **Page Recall@K:** For answerable cases, the fraction of distinct expected `(PDF, page)` pairs represented in the first K retrieved chunks, macro-averaged across cases. Cross-document questions have multiple expected pages.
- **Page Precision@K:** Retrieved chunks whose PDF and page match any expected evidence page, divided by configured K. If fewer than K chunks exist, the denominator remains K.
- **Page MRR:** Reciprocal rank of the first retrieved chunk matching any expected evidence page, macro-averaged. A miss contributes zero.
- **Page hit rate:** Share of answerable cases with at least one expected evidence page in the top K.
- **Chunk quote containment:** A relevant chunk contains an entire evidence quote after whitespace normalization. Chunk Recall@K divides distinct contained evidence quotes by expected evidence quotes; chunk Precision@K divides matching chunks by K; chunk MRR uses the first matching chunk; chunk hit rate records any match. A quote split across chunk boundaries does not count as contained.
- **Retrieval latency:** Query embedding plus FAISS search time per case. Mean, p50, and p95 are reported; PDF extraction, corpus embedding/index construction, generation, and judging are excluded.
- **Citation validity:** Share of returned citations whose source, page, and chunk ID match a retrieved chunk. Cases with no citations do not contribute a citation denominator.
- **Correct-page citation rate:** Returned citations matching one of that question's expected evidence pages divided by all returned citations.
- **Abstention accuracy:** Share of successfully generated cases that abstain for unanswerable questions and do not abstain for answerable questions. False-refusal rate is the abstention share among answerable cases. Unanswerable hallucination rate is the non-abstention share among unanswerable cases.
- **End-to-end latency:** Query embedding, retrieval, and generation duration; judging is excluded. p50 and p95 are reported.
- **Reference-answer lexical coverage:** Existing token-overlap proxy for answerable cases. It is secondary and cannot recognize many valid paraphrases or establish correctness.
- **LLM judge:** Faithfulness, answer correctness, and relevance are independently scored 1-5 with one-line reasons. Faithfulness asks whether each claim is supported by retrieved passages; correctness compares with the reference; relevance checks whether the answer addresses the question. This is a model judgment, not ground truth.
- **Tokens and estimated cost:** Input/output token counts are included when returned by the provider. Cost remains null unless provider-specific input/output per-million prices are configured as `GEMINI_INPUT_COST_PER_MILLION`, `GEMINI_OUTPUT_COST_PER_MILLION`, `NVIDIA_INPUT_COST_PER_MILLION`, and `NVIDIA_OUTPUT_COST_PER_MILLION`.

Results include overall metrics plus breakdowns by question type and primary evidence PDF. Each run writes a timestamped JSON, updates `latest.json`, `latest.csv`, and `latest.md`, and appends a row to `history.csv` under `data/evaluation/results/real/`. Only `latest.md` is intended for Git; the other result data is ignored. The Streamlit Evaluation tab reads these reports. Run the optional regression gate with:

```bash
.venv/bin/python scripts/check_thresholds.py
```

Thresholds live in `data/evaluation/thresholds.json`. A missing metric fails the gate rather than being treated as a measured zero.

### Limitations

Questions and reference answers are LLM-generated and may be ambiguous, unrepresentative, or factually wrong. Quote matching verifies only that the quoted text occurs on the stated page; it does not prove the answer is supported, that the question is answerable, or that the quote is sufficient. Unanswerable labels and cross-document reasoning also require human review. Gemini judge scores can inherit model bias and are not independent ground truth. Three or four papers and a small per-paper sample are not statistically representative; report sample sizes and avoid broad quality claims. PDF text extraction can lose reading order, tables, equations, and multi-column structure. The real pipeline reuses RAGLab's dense MiniLM/FAISS retrieval and Gemini generation, so results do not measure other retrievers or OCR.

## Verified results

_To be completed only after a genuine heldout run on reviewed PDFs. No measured values are recorded here yet._
