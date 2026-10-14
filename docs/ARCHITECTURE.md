# RAGLab architecture

```text
Streamlit upload
    -> PyMuPDF pages (source, one-based page)
    -> overlapping character chunks
    -> MiniLM CPU embeddings (normalized)
    -> per-collection FAISS IndexFlatIP
    -> top-K passages
     -> Gemini structured answer
         -> NVIDIA NIM fallback on Gemini errors or invalid citations
     -> source/page/chunk citation validation
```

`src/documents.py` handles PDF validation, extraction, and chunking. `src/retrieval.py` owns local embedding and cosine search. `src/generation.py` passes retrieved text as untrusted evidence, calls Gemini first, falls back to NVIDIA NIM when needed, and checks citations against actual retrievals. `src/nvidia.py` uses the standard-library HTTP client for NVIDIA's OpenAI-compatible endpoint. `src/evaluation/` runs retrieval and optional generation benchmarks without requiring Streamlit.

Collections and indexes are held in Streamlit session state. This keeps distinct uploads isolated and avoids persistent storage. The embedding model is cached per process. The first run downloads the model weights; subsequent runs reuse the cache when the host retains it.
