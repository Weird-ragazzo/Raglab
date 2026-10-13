# Development log

## 2026-10-14

- Inspected a clean, effectively empty repository on `main`; retained the user's existing untracked `.env` without reading or printing its secret.
- Created a Python 3.12 project virtual environment and installed the official `google-genai` SDK. Verified a minimal request with `gemini-3.5-flash-lite`; a first attempt on Gemini 3.8 Flash returned a temporary 503.
- Implemented PDF extraction/chunking, local embeddings and FAISS retrieval, evidence-constrained Gemini generation, Streamlit UI, deterministic evaluation metrics/runner, docs, and CI.
- Pinned a CPU-only PyTorch build after pip initially selected a CUDA bundle. The first MiniLM weight fetch hit sandbox DNS restrictions; after the authorized download, benchmark runs use the cached model offline.
- Verification: 10 tests passed; `compileall` and `git diff --check` passed. Streamlit started successfully on localhost and was stopped after the startup check.
- Retrieval-only dev comparison (5 answerable questions): baseline (900/150/top-5) Recall@5 1.00, MRR 1.00, Precision@5 0.20; top-3 kept Recall@3 and MRR at 1.00 while Precision@3 was 0.333. Chunk-size and overlap variants produced the same retrieval ranks on this tiny, short-passage set. No generation metrics were run; the lexical correctness, citation, and abstention metrics remain available behind `--generate`.
- Held-out baseline run (1 answerable question): Recall@3 1.00, MRR 1.00, Precision@3 0.333. The held-out set also has one unanswerable question, excluded from retrieval rank metrics and reserved for generation abstention scoring.
- Next useful task: expand the manually labeled benchmark with more papers and longer evidence spans, then run the generation/citation evaluation and capture genuine UI screenshots.

