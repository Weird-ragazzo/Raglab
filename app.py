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
                elif not settings.gemini_api_key and not settings.nvidia_api_key:
                    st.error("Configure GEMINI_API_KEY or NVIDIA_API_KEY to generate answers.")
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
                        st.caption(
                            f"{answer.provider} · {answer.model_name or 'configured model'} · "
                            f"Generation {answer.latency_ms:.0f} ms · {elapsed:.0f} ms total"
                        )
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

    st.divider()
    st.subheader("Real PDF benchmark")
    real_dir = Path("data/evaluation/results/real")
    latest_path = real_dir / "latest.json"
    history_path = real_dir / "history.csv"
    if not latest_path.exists():
        st.info("Add text-based research PDFs to data/evaluation/pdfs, then generate and evaluate a benchmark.")
        st.code(
            ".venv/bin/python -m src.evaluation.generate_benchmark --pdf-dir data/evaluation/pdfs --per-pdf 12\n"
            ".venv/bin/python -m src.evaluation.real_runner --split dev --compare\n"
            ".venv/bin/python -m src.evaluation.real_runner --split heldout --generate --judge"
        )
    else:
        import csv
        import json

        latest = json.loads(latest_path.read_text(encoding="utf-8"))
        real_metrics = latest.get("metrics", {})
        metric_keys = [
            ("Recall@5", "recall_at_5", True),
            ("MRR", "mrr", True),
            ("Citation validity", "citation_validity", True),
            ("Abstention accuracy", "abstention_accuracy", True),
            ("Faithfulness", "judge_faithfulness", False),
            ("End-to-end p95 latency", "end_to_end_latency_ms_p95", False),
        ]
        metric_columns = st.columns(len(metric_keys))
        for column, (label, key, percentage) in zip(metric_columns, metric_keys):
            value = real_metrics.get(key)
            shown = "Not run" if value is None else f"{value:.1%}" if percentage else f"{value:.1f}"
            column.metric(label, shown)
        st.caption(
            f"Latest: {latest.get('run_id', 'unknown')} · {latest.get('dataset', {}).get('split', 'unknown')} · "
            f"{latest.get('dataset', {}).get('cases', 0)} cases · {latest.get('config', {})}"
        )

        history_rows = []
        if history_path.exists():
            with history_path.open(encoding="utf-8", newline="") as history_file:
                history_rows = list(csv.DictReader(history_file))
        if history_rows:
            st.markdown("**Run history**")
            st.dataframe(history_rows, use_container_width=True, hide_index=True)
            score_rows = []
            latency_rows = []
            for row in history_rows:
                score_row = {"run": row.get("run_id", "")}
                latency_row = {"run": row.get("run_id", "")}
                for key in ("recall_at_k", "mrr", "citation_validity", "abstention_accuracy", "judge_faithfulness"):
                    try:
                        score_row[key] = float(row[key]) if row.get(key) else None
                    except (TypeError, ValueError):
                        score_row[key] = None
                for key in ("retrieval_latency_ms_p95", "end_to_end_latency_ms_p95"):
                    try:
                        latency_row[key] = float(row[key]) if row.get(key) else None
                    except (TypeError, ValueError):
                        latency_row[key] = None
                score_rows.append(score_row)
                latency_rows.append(latency_row)
            st.markdown("**Quality metrics over runs**")
            st.line_chart(score_rows, x="run")
            st.markdown("**p95 latency over runs (ms)**")
            st.line_chart(latency_rows, x="run")
        config_runs = []
        for run_path in sorted(real_dir.glob("*.json"), reverse=True):
            if run_path.name == "latest.json":
                continue
            try:
                report = json.loads(run_path.read_text(encoding="utf-8"))
                if report.get("dataset", {}).get("split") == "dev":
                    config_runs.append({
                        "Run": report.get("run_id", run_path.stem),
                        "Configuration": report.get("config", {}).get("name"),
                        "Chunk size": report.get("config", {}).get("chunk_size"),
                        "Overlap": report.get("config", {}).get("chunk_overlap"),
                        "Top K": report.get("config", {}).get("top_k"),
                        "Recall@K": report.get("metrics", {}).get("recall_at_k"),
                        "MRR": report.get("metrics", {}).get("mrr"),
                        "Precision@K": report.get("metrics", {}).get("precision_at_k"),
                    })
            except (OSError, ValueError):
                continue
        if config_runs:
            st.markdown("**Dev configuration comparisons**")
            st.dataframe(config_runs, use_container_width=True, hide_index=True)

        type_breakdown = latest.get("breakdown", {}).get("question_type", {})
        if type_breakdown:
            st.markdown("**Recall by question type**")
            st.bar_chart([{"question_type": name, "recall_at_k": values.get("recall_at_k")}
                          for name, values in type_breakdown.items()], x="question_type", y="recall_at_k")

        cases = latest.get("cases", [])
        if cases:
            st.markdown("**Per-case explorer**")
            status_filter = st.selectbox("Case status", ["all", "failed", "passed"], key="real_case_status")
            filtered_cases = [case for case in cases if status_filter == "all"
                              or status_filter == "passed" and case.get("passed")
                              or status_filter == "failed" and not case.get("passed")]
            if filtered_cases:
                selected_id = st.selectbox(
                    "Question", range(len(filtered_cases)),
                    format_func=lambda index: f"{filtered_cases[index]['id']} · {filtered_cases[index]['question'][:100]}",
                    key="real_case_id",
                )
                selected_case = filtered_cases[selected_id]
                st.write(selected_case["question"])
                st.caption(f"Type: {selected_case.get('question_type')} · Answerable: {selected_case.get('answerable')} · Passed: {selected_case.get('passed')}")
                st.markdown("**Expected evidence**")
                if selected_case.get("evidence"):
                    for evidence in selected_case["evidence"]:
                        st.write(f"{evidence['pdf']} · page {evidence['page']}")
                        st.code(evidence.get("quote", ""))
                else:
                    st.write("No evidence; benchmark item is marked unanswerable.")
                st.markdown("**Reference answer**")
                st.write(selected_case.get("reference_answer") or "Not applicable")
                st.markdown("**Retrieved passages**")
                for passage in selected_case.get("retrieved", []):
                    with st.expander(f"{passage['pdf']} · page {passage['page']} · score {passage['score']:.3f}"):
                        st.write(passage["text"])
                st.markdown("**Generated answer**")
                st.write(selected_case.get("answer") or "Not generated")
                if selected_case.get("provider"):
                    st.caption(f"Generated by {selected_case['provider']} · {selected_case.get('model') or 'unknown model'}")
                if selected_case.get("error") or selected_case.get("judge_error"):
                    st.warning(selected_case.get("error") or selected_case.get("judge_error"))
                if selected_case.get("judge"):
                    st.markdown("**Judge scores and reasons**")
                    st.caption(
                        f"Judged by {selected_case.get('judge_provider', 'unknown')} · "
                        f"{selected_case.get('judge_model', 'unknown model')}"
                    )
                    for dimension in ("faithfulness", "correctness", "relevance"):
                        st.write(f"{dimension.title()}: {selected_case['judge'][dimension]}/5 · {selected_case['judge'][dimension + '_reason']}")
            else:
                st.info("No cases match this status filter.")

with tab_about:
    st.subheader("Small, inspectable RAG")
    st.markdown("""
    **Extract** PDF text with PyMuPDF, preserving one-based page numbers. **Chunk and embed** locally with MiniLM on CPU. **Retrieve** by cosine similarity in an in-memory FAISS index. **Generate** with the configured Gemini model using only the retrieved passages. Answers expose their passages so you can verify them.

    Collections live only in this Streamlit session. Hosted restarts clear uploads and indexes. Scanned PDFs need OCR before upload; OCR is not included. Large papers may need a smaller page limit on constrained hosts.
    """)
    st.code(".venv/bin/streamlit run app.py")

