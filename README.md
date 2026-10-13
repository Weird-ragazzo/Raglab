# RAGLab

RAGLab is a small Streamlit research-paper assistant. It extracts selectable PDF text, retrieves local passages with MiniLM and FAISS, and asks Gemini to answer from those passages with verifiable page citations. A separate CLI measures retrieval quality on a versioned benchmark.

## Features

- PDF upload safeguards for file type, size, page count, empty files, and scanned PDFs.
- One-based page provenance, configurable overlapping chunks, CPU embeddings, and isolated in-memory collections.
- Evidence-grounded Gemini answers with citation validation and an explicit abstention path.
- Inspectable retrieved passages and similarity scores.
- Retrieval-only evaluation with Recall@K, MRR, Precision@K, latency, and optional generation, citation, correctness-proxy, and abstention metrics.
- Pytest and GitHub Actions.

## Architecture

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). The app keeps uploaded data and indexes in session memory. Hosted restarts clear them. The MiniLM weights download on first use and need local disk/cache space.

## Setup

Python 3.10–3.12 is recommended. Create and activate a virtual environment, install dependencies, then configure a local secret:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

Set `GEMINI_API_KEY` in `.env`. Never commit `.env`; it is ignored. `GEMINI_MODEL` defaults to `gemini-3.5-flash-lite` and can be changed to a currently supported Gemini model. The embedding model is configurable through `EMBEDDING_MODEL`.

## Run

```bash
.venv/bin/streamlit run app.py
```

Upload text-based PDFs in the sidebar, build a collection, and ask questions in **Ask papers**. Increase the chunking settings before rebuilding the collection. Scanned PDFs need OCR performed before upload. Session uploads are temporary and are not persisted to disk.

## Evaluation

```bash
.venv/bin/python -m src.evaluation.runner --split dev --compare
```

This compares the configured baseline (900 characters, 150 overlap, top 5) with one-variable-at-a-time alternatives for chunk size, overlap, and top-K. Use `--sweep top_k` (or either chunking setting) to isolate one parameter. It runs retrieval only and writes actual results to `data/evaluation/results/`. Use `--split heldout` only after choosing settings on dev. Add `--generate` for Gemini generation metrics, which may consume API quota. The lexical reference-answer rubric is a transparent proxy, not a judge. See [docs/EVALUATION.md](docs/EVALUATION.md) and [docs/EXPERIMENT_LOG_TEMPLATE.md](docs/EXPERIMENT_LOG_TEMPLATE.md).

The starter benchmark has manually verified examples derived from a PLOS article available under a Creative Commons Attribution License. Provenance and the limitations of its HTML section references are documented in [docs/EVALUATION.md](docs/EVALUATION.md).

## Tests

```bash
.venv/bin/python -m pytest -q
```

Gemini calls are not part of normal tests.

## Streamlit Community Cloud

1. Push the project to a GitHub repository.
2. Create a Streamlit Community Cloud app pointing to `app.py`.
3. Add `GEMINI_API_KEY` under **App settings → Secrets**. Optionally set `GEMINI_MODEL` there too.
4. Keep uploaded papers temporary; Community Cloud restarts may remove runtime files and session state. No database or hosted vector store is required.

Free service and model quota availability are not guaranteed. Keep upload limits and CPU use in mind on constrained instances.

## Limitations

- Text extraction only; no OCR, table understanding, reranker, or hybrid sparse retrieval.
- In-memory indexes are session scoped and are lost on restart.
- Citation validity ensures references point to retrieved passages, not that a passage logically supports every claim.
- The starter benchmark is deliberately small. Results should not be generalized without a larger, held-out dataset.

## Screenshots

Capture the running app with a sample paper, a grounded answer, and its expanded evidence passages. Add only genuine screenshots under `docs/screenshots/`.

