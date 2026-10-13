"""RAGLab: research paper Q&A with inspectable retrieval evidence."""

from __future__ import annotations

import hashlib
import time
import uuid

import streamlit as st

from src.config import settings
from src.documents import Chunk, chunk_pages, load_pdf_bytes
from src.generation import answer_question
from src.retrieval import VectorIndex, embed_texts

st.set_page_config(page_title="RAGLab", page_icon="◈", layout="wide")
st.markdown("""
<style>
.block-container {max-width: 1180px; padding-top: 2rem;}
.hero {padding: 1.7rem 2rem; border-radius: 18px; background: linear-gradient(120deg,#e3eee8,#f7f8f5 72%); margin-bottom:1.4rem;}
.hero h1 {letter-spacing:-.04em; margin:0; font-size:2.4rem;}
.hero p {margin:.5rem 0 0; color:#52625c; max-width:720px;}
.metric-note {color:#66756e; font-size:.85rem;}
</style>
<div class="hero"><h1>RAGLab <span style="color:#226f63">◈</span></h1>
<p>Ask research papers. Follow every answer back to the retrieved page and passage.</p></div>
""", unsafe_allow_html=True)

if "collections" not in st.session_state:
    st.session_state.collections = {}
if "active_collection" not in st.session_state:
    st.session_state.active_collection = None
if "messages" not in st.session_state:
    st.session_state.messages = {}

with st.sidebar:
    st.header("Document collection")
    collection_name = st.text_input("Collection name", placeholder="e.g. RAG survey papers")
    uploads = st.file_uploader("Add research PDFs", type=["pdf"], accept_multiple_files=True)
    if st.button("Build collection", type="primary", use_container_width=True):
        if not collection_name.strip():
            st.error("Enter a collection name first.")
        elif not uploads:
            st.error("Choose at least one PDF.")
        else:
            pages = []
            failures = []
            for upload in uploads:
                try:
                    pages.extend(load_pdf_bytes(
                        upload.getvalue(), upload.name,
                        settings.max_upload_mb * 1024 * 1024, settings.max_pdf_pages,
                    ))
                except Exception as exc:
                    failures.append(f"{upload.name}: {exc}")
            if failures:
                for failure in failures:
                    st.warning(failure)
            if pages:
                with st.spinner("Embedding extracted pages on CPU…"):
                    chunks = chunk_pages(
                        pages, chunk_size=st.session_state.get("chunk_size", settings.default_chunk_size),
                        overlap=st.session_state.get("chunk_overlap", settings.default_chunk_overlap),
                    )
                    vectors = embed_texts([chunk.text for chunk in chunks], settings.embedding_model)
                    index = VectorIndex(chunks, vectors)
                key = uuid.uuid4().hex[:8]
                st.session_state.collections[key] = {
                    "name": collection_name.strip(), "chunks": chunks, "index": index,
                    "files": sorted({page.source for page in pages}),
                    "digest": hashlib.sha256("".join(c.chunk_id for c in chunks).encode()).hexdigest()[:12],
                }
                st.session_state.active_collection = key
                st.session_state.messages[key] = []
                st.success(f"Indexed {len(chunks)} passages.")
    if st.session_state.collections:
        keys = list(st.session_state.collections)
        labels = {key: st.session_state.collections[key]["name"] for key in keys}
        active = st.selectbox("Active collection", keys,
                              format_func=lambda key: labels[key],
                              index=keys.index(st.session_state.active_collection)
                              if st.session_state.active_collection in keys else 0)
        st.session_state.active_collection = active
        if st.button("Remove active collection", use_container_width=True):
            del st.session_state.collections[active]
            st.session_state.messages.pop(active, None)
            st.session_state.active_collection = None
            st.rerun()
    st.divider()
    st.subheader("Retrieval settings")
    top_k = st.slider("Passages per answer", 1, 12, 5)
    chunk_size = st.slider("Chunk size (characters)", 300, 1800, settings.default_chunk_size, 100)
    chunk_overlap = st.slider("Chunk overlap", 0, 500, settings.default_chunk_overlap, 25)
    st.session_state.chunk_size = chunk_size
    st.session_state.chunk_overlap = chunk_overlap
    st.caption("Chunk settings apply when you build a collection.")

tab_chat, tab_eval, tab_about = st.tabs(["Ask papers", "Evaluation", "How it works"])
with tab_chat:
    active_key = st.session_state.active_collection
    if not active_key or active_key not in st.session_state.collections:
        st.info("Upload one or more text-based research PDFs in the sidebar to begin.")
    else:
        collection = st.session_state.collections[active_key]
        st.caption(f"{len(collection['files'])} papers · {collection['index'].size} passages · temporary session storage")
        for message in st.session_state.messages[active_key]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message.get("evidence"):
                    with st.expander("Inspect retrieved evidence"):
                        for item in message["evidence"]:
                            st.markdown(f"**{item['source']} · page {item['page']} · score {item['score']:.3f}**")
                            st.write(item["text"])
        question = st.chat_input("Ask a question about this collection")
        if question:
            st.session_state.messages[active_key].append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)
            with st.chat_message("assistant"):
                if collection["index"].size == 0:
                    st.warning("This collection has no searchable passages.")
                elif not settings.gemini_api_key:
                    st.error("Configure GEMINI_API_KEY to generate answers.")
                else:
                    try:
                        with st.spinner("Searching papers and checking evidence…"):
                            query = embed_texts([question], settings.embedding_model)[0]
                            hits = collection["index"].search(query, top_k)
                            started = time.perf_counter()
                            answer = answer_question(question, hits)
                            elapsed = (time.perf_counter() - started) * 1000
                        st.markdown(answer.answer)
                        if answer.citations:
                            citation_labels = ", ".join(
                                f"{citation['source']} · page {citation['page']}"
                                for citation in answer.citations
                            )
                            st.caption(f"Citations: {citation_labels}")
                        evidence = [{"source": hit.chunk.source, "page": hit.chunk.page,
                                     "text": hit.chunk.text, "score": hit.score}
                                    for hit in hits]
                        with st.expander("Inspect retrieved evidence"):
                            for item in evidence:
                                st.markdown(f"**{item['source']} · page {item['page']} · score {item['score']:.3f}**")
                                st.write(item["text"])
                        st.caption(f"Generation {answer.latency_ms:.0f} ms · {elapsed:.0f} ms total")
                        st.session_state.messages[active_key].append({
                            "role": "assistant", "content": answer.answer, "evidence": evidence,
                        })
                    except Exception as exc:
                        st.error(f"The answer could not be generated: {exc}")
            st.rerun()

with tab_eval:
    st.subheader("Reproducible retrieval benchmark")
    st.write("Run the versioned benchmark from a terminal. It reports deterministic retrieval metrics without calling Gemini by default.")
    st.code(".venv/bin/python -m src.evaluation.runner --split dev --compare")
    st.caption("Use --split heldout only after choosing settings on dev. Add --generate to measure Gemini answer, citation, and abstention metrics.")
    from pathlib import Path
    result_dir = Path("data/evaluation/results")
    results = sorted(result_dir.glob("*.json"), reverse=True) if result_dir.exists() else []
    if results:
        import json
        rows = []
        for path in results:
            try:
                report = json.loads(path.read_text())
                rows.append({"Run": path.name, "Split": report["dataset"]["split"], **report["metrics"]})
            except (OSError, ValueError, KeyError):
                continue
        if rows:
            st.dataframe(rows, use_container_width=True, hide_index=True)
    else:
        st.info("No measured results yet. Run the command above to create the first result file.")

with tab_about:
    st.subheader("Small, inspectable RAG")
    st.markdown("""
    **Extract** PDF text with PyMuPDF, preserving one-based page numbers. **Chunk and embed** locally with MiniLM on CPU. **Retrieve** by cosine similarity in an in-memory FAISS index. **Generate** with the configured Gemini model using only the retrieved passages. Answers expose their passages so you can verify them.

    Collections live only in this Streamlit session. Hosted restarts clear uploads and indexes. Scanned PDFs need OCR before upload; OCR is not included. Large papers may need a smaller page limit on constrained hosts.
    """)
    st.code(".venv/bin/streamlit run app.py")

