from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.documents import Chunk, PageText
from src.evaluation import generate_benchmark as generator
from src.evaluation.generate_benchmark import (
    GeneratedBatch, GeneratedQuestion, Evidence, _verified_records, quote_is_on_page, stratified_split,
)
from src.evaluation.real_runner import _aggregate, _judge_answer, _persist, _retrieval_scores, check_thresholds, run_real
from src.generation import Answer, answer_question
from src.retrieval import SearchResult


@pytest.fixture
def tiny_pdf(tmp_path: Path) -> Path:
    fitz = pytest.importorskip("fitz")
    path = tmp_path / "tiny-paper.pdf"
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), "RAG systems retrieve evidence from documents.")
    document.save(path)
    document.close()
    return path


def test_quote_verification_normalizes_whitespace():
    assert quote_is_on_page("retrieve evidence from\ndocuments", "RAG systems retrieve evidence from documents.")
    assert not quote_is_on_page("retrieve unrelated evidence", "RAG systems retrieve evidence from documents.")
    assert not quote_is_on_page("  \n ", "RAG systems retrieve evidence from documents.")


def test_generator_requires_pdfs_with_actionable_message(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="Put 3-4 text-based research PDFs"):
        generator.generate_benchmark(tmp_path)


def test_verification_logs_quote_duplicate_and_trivial_rejections():
    batch = GeneratedBatch(questions=[
        GeneratedQuestion(question="What is retrieved?", reference_answer="Evidence is retrieved.",
                          answerable=True, question_type="factual",
                          evidence=[Evidence(pdf="paper.pdf", page=1, quote="RAG retrieves evidence.")]),
        GeneratedQuestion(question="What is retrieved?", reference_answer="Evidence is retrieved.",
                          answerable=True, question_type="factual",
                          evidence=[Evidence(pdf="paper.pdf", page=1, quote="RAG retrieves evidence.")]),
        GeneratedQuestion(question="Evidence is retrieved in RAG, right?", reference_answer="Evidence",
                          answerable=True, question_type="factual",
                          evidence=[Evidence(pdf="paper.pdf", page=1, quote="RAG retrieves evidence.")]),
        GeneratedQuestion(question="What does RAG retrieve?", reference_answer="Something",
                          answerable=True, question_type="factual",
                          evidence=[Evidence(pdf="paper.pdf", page=1, quote="not on the page")]),
    ])
    records, rejected = _verified_records(
        batch, {"paper.pdf": [PageText("paper.pdf", 1, "RAG retrieves evidence.")]},
        anchor_pdf="paper.pdf", other_pdf=None, next_id=1, seen_questions=set(),
    )
    assert len(records) == 1
    assert rejected["duplicate"] == 1
    assert rejected["trivial"] == 1
    assert rejected["quote"] == 1


def test_page_and_quote_retrieval_metrics_are_hand_computed():
    retrieved = [
        SearchResult(Chunk("a", "paper.pdf", 2, "Exact evidence quote appears here."), .9, 1),
        SearchResult(Chunk("b", "paper.pdf", 4, "Unrelated passage."), .7, 2),
    ]
    scores = _retrieval_scores(
        retrieved,
        [{"pdf": "paper.pdf", "page": 2, "quote": "Exact evidence quote appears here."}],
        2,
    )
    assert scores == {
        "page_recall": 1.0,
        "page_precision": 0.5,
        "page_mrr": 1.0,
        "page_hit_rate": 1.0,
        "chunk_recall": 1.0,
        "chunk_precision": 0.5,
        "chunk_mrr": 1.0,
        "chunk_hit_rate": 1.0,
    }


def test_split_is_deterministic_and_stratified():
    rows = [{"id": f"q{index}", "question_type": kind}
            for kind in ("factual", "numeric") for index in range(5)]
    first = stratified_split(rows, seed=42)
    assert first == stratified_split(rows, seed=42)
    for kind in ("factual", "numeric"):
        splits = [row["split"] for row in first if row["question_type"] == kind]
        assert splits.count("dev") == 3
        assert splits.count("heldout") == 2


def test_singleton_question_types_still_fill_heldout_split():
    rows = [{"id": f"q{index}", "question_type": f"kind-{index}"} for index in range(5)]
    assigned = stratified_split(rows, seed=42)
    assert sum(record["split"] == "heldout" for record in assigned) == 2


def test_unanswerable_cases_do_not_dilute_retrieval_aggregates():
    base = {
        "page_recall": 1.0, "page_precision": 0.2, "page_mrr": 1.0, "page_hit_rate": 1.0,
        "chunk_recall": 1.0, "chunk_precision": 0.2, "chunk_mrr": 1.0, "chunk_hit_rate": 1.0,
    }
    cases = [
        {**base, "answerable": True, "page_recall_at_5": 1.0},
        {**base, "answerable": False, "page_recall": None, "chunk_recall": None, "page_recall_at_5": None},
    ]
    metrics = _aggregate(cases, 5, generate=False, judge=False)
    assert metrics["recall_at_k"] == 1.0
    assert metrics["hit_rate"] == 1.0
    assert metrics["retrieval_question_count"] == 1


def test_mocked_judge_returns_structured_scores(monkeypatch):
    import src.evaluation.real_runner as runner

    monkeypatch.setattr(runner, "settings", SimpleNamespace(gemini_api_key="test-key"))

    class FakeModels:
        def generate_content(self, **kwargs):
            assert kwargs["model"] == "judge-model"
            assert kwargs["config"].temperature == 0
            return SimpleNamespace(text=json.dumps({"result": {
                "faithfulness": 5, "faithfulness_reason": "Claims match the passage.",
                "correctness": 4, "correctness_reason": "Matches the reference.",
                "relevance": 5, "relevance_reason": "Directly answers the question.",
            }}))

    hit = SearchResult(Chunk("a", "paper.pdf", 1, "Evidence text."), .9, 1)
    result, provider, model = _judge_answer(
        "Question?", "Reference", "Answer", [hit], "judge-model", SimpleNamespace(models=FakeModels())
    )
    assert result.faithfulness == 5
    assert result.correctness_reason == "Matches the reference."
    assert provider == "gemini"
    assert model == "judge-model"


def test_judge_falls_back_to_nvidia(monkeypatch):
    import src.evaluation.real_runner as runner
    import src.nvidia as nvidia
    from google import genai
    from src.nvidia import NvidiaResponse

    monkeypatch.setattr(runner, "settings", SimpleNamespace(
        gemini_api_key="gemini-test", nvidia_api_key="nvidia-test", nvidia_model="meta/generator",
        nvidia_judge_model="meta/judge",
    ))

    class BrokenModels:
        def generate_content(self, **kwargs):
            raise RuntimeError("Gemini service unavailable")

    class FakeGeminiClient:
        models = BrokenModels()

        def __init__(self, **kwargs):
            assert kwargs["api_key"] == "gemini-test"

    monkeypatch.setattr(genai, "Client", FakeGeminiClient)

    def fake_nvidia(prompt, *, model_name, temperature):
        assert model_name == "meta/judge"
        assert temperature == 0
        return NvidiaResponse(json.dumps({"result": {
            "faithfulness": 4, "faithfulness_reason": "Supported.",
            "correctness": 5, "correctness_reason": "Matches reference.",
            "relevance": 4, "relevance_reason": "On topic.",
        }}), model_name, 100, 30)

    monkeypatch.setattr(nvidia, "generate_json", fake_nvidia)
    hit = SearchResult(Chunk("a", "paper.pdf", 1, "Evidence text."), .9, 1)
    result, provider, model = _judge_answer("Question?", "Reference", "Answer", [hit], "gemini-judge")
    assert result.correctness == 5
    assert provider == "nvidia"
    assert model == "meta/judge"


def test_mocked_gemini_generates_verified_pdf_benchmark(monkeypatch, tmp_path: Path, tiny_pdf: Path):
    from google import genai

    monkeypatch.setattr(generator, "settings", SimpleNamespace(gemini_api_key="test-key", gemini_model="test-model"))
    monkeypatch.setattr(generator, "_model_name", lambda: "test-model")

    class FakeModels:
        def generate_content(self, **kwargs):
            assert kwargs["model"] == "test-model"
            assert kwargs["config"].response_mime_type == "application/json"
            return SimpleNamespace(text=json.dumps({"questions": [{
                "question": "What do RAG systems retrieve?",
                "reference_answer": "They retrieve evidence from documents.",
                "answerable": True,
                "question_type": "factual",
                "cross_document": False,
                "evidence": [{"pdf": tiny_pdf.name, "page": 1,
                              "quote": "RAG systems retrieve evidence from documents."}],
            }]}))

    class FakeClient:
        models = FakeModels()

        def __init__(self, **kwargs):
            assert kwargs["api_key"] == "test-key"

    monkeypatch.setattr(genai, "Client", FakeClient)
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    (pdf_dir / tiny_pdf.name).write_bytes(tiny_pdf.read_bytes())
    output_dir = tmp_path / "generated"
    result = generator.generate_benchmark(pdf_dir, 1, output_dir=output_dir, seed=5)
    record = json.loads((output_dir / "benchmark_real.jsonl").read_text().splitlines()[0])
    assert record["verified_by"] == "auto"
    assert record["evidence_page"] == 1
    assert result["generator_model"] == "test-model"
    assert "tiny-paper.pdf · page 1" in (output_dir / "review.md").read_text()


def test_benchmark_generation_falls_back_to_nvidia(monkeypatch):
    import src.evaluation.generate_benchmark as benchmark_generator
    from src.nvidia import NvidiaResponse

    monkeypatch.setattr(benchmark_generator, "settings", SimpleNamespace(
        gemini_api_key="gemini-test", nvidia_api_key="nvidia-test",
    ))

    class BrokenModels:
        def generate_content(self, **kwargs):
            raise RuntimeError("Gemini service unavailable")

    def fake_nvidia(prompt, *, temperature):
        assert "Page excerpts" in prompt
        assert temperature == 0.2
        text = json.dumps({"questions": [{
            "question": "What is the result?", "reference_answer": "A result.",
            "answerable": True, "question_type": "factual", "evidence": [],
        }]})
        return NvidiaResponse(text, "meta/test-model", 100, 50)

    monkeypatch.setattr(benchmark_generator, "nvidia_generate_json", fake_nvidia)
    batch, provider, model = benchmark_generator._generate_batch(
        "Page excerpts", "gemini-test", SimpleNamespace(models=BrokenModels())
    )
    assert batch.questions[0].question == "What is the result?"
    assert provider == "nvidia"
    assert model == "meta/test-model"


def test_answer_generation_falls_back_to_nvidia(monkeypatch):
    import src.generation as generation
    import src.nvidia as nvidia
    from google import genai
    from src.nvidia import NvidiaResponse

    monkeypatch.setattr(generation, "settings", SimpleNamespace(
        gemini_api_key="gemini-test", gemini_model="gemini-test",
        nvidia_api_key="nvidia-test", nvidia_model="meta/test-model",
    ))

    class BrokenModels:
        def generate_content(self, **kwargs):
            raise RuntimeError("Gemini rate limit 429")

    class FakeGeminiClient:
        models = BrokenModels()

        def __init__(self, **kwargs):
            assert kwargs["api_key"] == "gemini-test"

    monkeypatch.setattr(genai, "Client", FakeGeminiClient)

    def fake_nvidia(prompt, *, temperature):
        assert "Question:" in prompt
        assert temperature == 0.1
        return NvidiaResponse(
            '{"answer":"Supported by the passage.","citations":[{"chunk_id":"c1","source":"paper.pdf","page":1}]}',
            "meta/test-model", 20, 8,
        )

    monkeypatch.setattr(nvidia, "generate_json", fake_nvidia)
    hit = SearchResult(Chunk("c1", "paper.pdf", 1, "Evidence passage."), .9, 1)
    answer = answer_question("What is supported?", [hit])
    assert answer.provider == "nvidia"
    assert answer.model_name == "meta/test-model"
    assert answer.input_tokens == 20
    assert answer.citations[0]["chunk_id"] == "c1"


def test_answer_generation_supports_nvidia_without_gemini_key(monkeypatch):
    import src.generation as generation
    import src.nvidia as nvidia
    from src.nvidia import NvidiaResponse

    monkeypatch.setattr(generation, "settings", SimpleNamespace(
        gemini_api_key=None, nvidia_api_key="nvidia-test", nvidia_model="meta/test-model",
    ))
    monkeypatch.setattr(nvidia, "generate_json", lambda *args, **kwargs: NvidiaResponse(
        '{"answer":"Supported.","citations":[{"chunk_id":"c1","source":"paper.pdf","page":1}]}',
        "meta/test-model", 12, 3,
    ))
    hit = SearchResult(Chunk("c1", "paper.pdf", 1, "Evidence passage."), .9, 1)
    answer = answer_question("What is supported?", [hit])
    assert answer.provider == "nvidia"
    assert answer.answer == "Supported."


def test_nvidia_json_client_retries_timeout_and_parses_usage(monkeypatch):
    import src.nvidia as nvidia

    monkeypatch.setattr(nvidia, "settings", SimpleNamespace(
        nvidia_api_key="test-key", nvidia_model="nvidia/test", nvidia_base_url="https://example.test/chat",
    ))
    attempts = []

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self):
            return json.dumps({
                "model": "nvidia/test",
                "choices": [{"message": {"content": '{"ok":true}'}}],
                "usage": {"prompt_tokens": 15, "completion_tokens": 4},
            }).encode()

    def fake_urlopen(request, timeout):
        assert request.full_url == "https://example.test/chat"
        assert timeout == 120
        attempts.append(request)
        if len(attempts) == 1:
            raise TimeoutError("temporary timeout")
        return FakeResponse()

    monkeypatch.setattr(nvidia, "urlopen", fake_urlopen)
    monkeypatch.setattr(nvidia.time, "sleep", lambda _seconds: None)
    result = nvidia.generate_json("prompt")
    assert len(attempts) == 2
    assert result.text == '{"ok":true}'
    assert result.input_tokens == 15
    assert result.output_tokens == 4


def test_real_runner_uses_pdf_retrieval_and_scores_mocked_generation(monkeypatch, tmp_path: Path, tiny_pdf: Path):
    import numpy as np
    import src.evaluation.real_runner as runner

    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    (pdf_dir / tiny_pdf.name).write_bytes(tiny_pdf.read_bytes())
    dataset_path = tmp_path / "benchmark_real.jsonl"
    dataset_path.write_text(json.dumps({
        "id": "real-0001", "split": "dev", "question": "What do RAG systems retrieve?",
        "reference_answer": "evidence from documents", "answerable": True,
        "question_type": "factual", "evidence_pdf": tiny_pdf.name, "evidence_page": 1,
        "evidence_quote": "RAG systems retrieve evidence from documents.",
        "evidence": [{"pdf": tiny_pdf.name, "page": 1,
                      "quote": "RAG systems retrieve evidence from documents."}],
    }) + "\n")
    monkeypatch.setattr(runner, "embed_texts", lambda texts, model: np.tile([[1.0, 0.0]], (len(texts), 1)).astype("float32"))

    def answer(question, retrieved, **kwargs):
        assert kwargs == {"strict_citations": False}
        hit = retrieved[0]
        return Answer("RAG retrieves evidence from documents.", [{
            "source": hit.chunk.source, "page": hit.chunk.page, "chunk_id": hit.chunk.chunk_id,
        }], False, 4.0, 12, 5)

    result = run_real(
        {"name": "test", "chunk_size": 900, "chunk_overlap": 150, "top_k": 5},
        "dev", generate=True, dataset_path=dataset_path, pdf_dir=pdf_dir, answer_fn=answer,
    )
    assert result["metrics"]["recall_at_k"] == 1.0
    assert result["metrics"]["citation_validity"] == 1.0
    assert result["metrics"]["abstention_accuracy"] == 1.0
    assert result["cases"][0]["passed"]


def test_threshold_checker_fails_missing_or_low_values():
    thresholds = {"recall_at_5": 0.7, "citation_validity": 0.8}
    assert check_thresholds({"metrics": {"recall_at_5": 0.8, "citation_validity": 0.9}}, thresholds) == []
    failures = check_thresholds({"metrics": {"recall_at_5": 0.6, "citation_validity": None}}, thresholds)
    assert len(failures) == 2
    assert "below minimum" in failures[0]
    assert "not measured" in failures[1]


def test_persist_writes_json_csv_markdown_and_history(tmp_path: Path):
    result = {
        "dataset": {"path": "data/evaluation/real/benchmark_real.jsonl", "version": "v1", "split": "dev", "cases": 1},
        "config": {"name": "baseline", "chunk_size": 900, "chunk_overlap": 150, "top_k": 5},
        "metrics": {"recall_at_k": 1.0, "recall_at_5": 1.0, "mrr": 1.0, "precision_at_k": 0.2,
                    "hit_rate": 1.0, "failed_cases": 0, "case_count": 1},
        "breakdown": {"question_type": {}, "pdf": {}},
        "cases": [{"id": "q1", "question_type": "factual", "question": "Question?", "passed": True,
                   "retrieved_pages": ["paper.pdf:1"], "answer": "Answer"}],
        "runtime": {"created_at": "2026-10-14T00:00:00+00:00", "git_commit": "abc"},
    }
    saved = _persist(result, tmp_path)
    assert saved.exists()
    assert (tmp_path / "latest.json").exists()
    assert "question" in (tmp_path / "latest.csv").read_text()
    assert "Executive summary" in (tmp_path / "latest.md").read_text()
    assert len((tmp_path / "history.csv").read_text().splitlines()) == 2