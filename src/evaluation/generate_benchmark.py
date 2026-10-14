"""Generate and automatically verify a benchmark from local research PDFs."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from src.config import ROOT, settings
from src.documents import PageText, extract_pdf
from src.nvidia import generate_json as nvidia_generate_json

DEFAULT_PDF_DIR = ROOT / "data" / "evaluation" / "pdfs"
OUTPUT_DIR = ROOT / "data" / "evaluation" / "real"
QUESTION_TYPES = ("factual", "numeric", "definition", "method", "comparison", "multi_hop")
DEFAULT_SEED = 20261014


class Evidence(BaseModel):
    pdf: str
    page: int = Field(ge=1)
    quote: str = Field(min_length=1)


class GeneratedQuestion(BaseModel):
    question: str = Field(min_length=1)
    reference_answer: str = ""
    answerable: bool
    question_type: str
    cross_document: bool = False
    evidence: list[Evidence] = Field(default_factory=list)


class GeneratedBatch(BaseModel):
    questions: list[GeneratedQuestion]


class BenchmarkRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    split: str
    question: str = Field(min_length=1)
    reference_answer: str
    answerable: bool
    question_type: str
    cross_document: bool
    evidence_pdf: str | None
    evidence_page: int | None
    evidence_quote: str | None
    evidence: list[Evidence]
    verified_by: str

    @model_validator(mode="after")
    def validate_record(self) -> "BenchmarkRecord":
        if self.split not in {"dev", "heldout"}:
            raise ValueError("split must be dev or heldout")
        if self.question_type not in QUESTION_TYPES:
            raise ValueError("question_type is not supported")
        if self.verified_by not in {"auto", "human"}:
            raise ValueError("verified_by must be auto or human")
        if self.answerable != bool(self.evidence):
            raise ValueError("answerable records need evidence; unanswerable records must not have evidence")
        if self.evidence:
            primary = self.evidence[0]
            if (self.evidence_pdf, self.evidence_page, self.evidence_quote) != (primary.pdf, primary.page, primary.quote):
                raise ValueError("primary evidence fields must match the first evidence item")
        return self


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def quote_is_on_page(quote: str, page_text: str) -> bool:
    normalized_quote = normalize_whitespace(quote)
    return bool(normalized_quote) and normalized_quote in normalize_whitespace(page_text)


def stratified_split(records: list[dict[str, Any]], seed: int = DEFAULT_SEED) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(str(record["question_type"]), []).append(record)
    assigned: list[dict[str, Any]] = []
    for question_type in sorted(grouped):
        group = list(grouped[question_type])
        rng.shuffle(group)
        dev_count = round(len(group) * 0.6)
        if len(group) > 1:
            dev_count = min(len(group) - 1, max(1, dev_count))
        else:
            dev_count = 1
        assigned.extend({**record, "split": "dev" if index < dev_count else "heldout"}
                        for index, record in enumerate(group))
    if len(assigned) > 1:
        target_heldout = min(len(assigned) - 1, max(1, round(len(assigned) * 0.4)))
        current_heldout = sum(record["split"] == "heldout" for record in assigned)
        for record in assigned:
            if current_heldout >= target_heldout:
                break
            if record["split"] == "dev" and len(grouped[str(record["question_type"])]) == 1:
                record["split"] = "heldout"
                current_heldout += 1
    return assigned


def _sample_pages(pages: list[PageText], rng: random.Random, limit: int = 4) -> list[PageText]:
    if len(pages) <= limit:
        return pages
    selected = sorted(rng.sample(pages, limit), key=lambda page: page.page)
    return selected


def _model_name() -> str:
    import os

    return os.getenv("BENCHMARK_GENERATOR_MODEL", settings.gemini_model)


def _is_retryable(exc: Exception) -> bool:
    status = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    message = str(exc).lower()
    return status in {429, 500, 502, 503, 504} or any(
        token in message for token in ("429", "503", "resource_exhausted", "temporarily unavailable")
    )


def _generate_batch(
    prompt: str, model_name: str, client: Any | None = None
) -> tuple[GeneratedBatch, str, str]:
    gemini_error: Exception | None = None
    if settings.gemini_api_key:
        if client is None:
            from google import genai

            client = genai.Client(api_key=settings.gemini_api_key)
        from google.genai import types

        for attempt in range(5):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=GeneratedBatch,
                        temperature=0.2,
                    ),
                )
                return GeneratedBatch.model_validate_json(response.text or "{}"), "gemini", model_name
            except Exception as exc:
                gemini_error = exc
                if attempt == 4 or not _is_retryable(exc):
                    break
                time.sleep(min(2**attempt, 16))
    else:
        gemini_error = RuntimeError("GEMINI_API_KEY is not configured.")
    if getattr(settings, "nvidia_api_key", None):
        try:
            response = nvidia_generate_json(prompt, temperature=0.2)
            batch = GeneratedBatch.model_validate_json(response.text)
            return batch, "nvidia", response.model
        except Exception as nvidia_error:
            raise RuntimeError(
                f"Gemini benchmark generation failed ({gemini_error}); "
                f"NVIDIA fallback failed ({nvidia_error})."
            ) from nvidia_error
    raise RuntimeError(
        f"Gemini benchmark generation failed ({gemini_error}); "
        "NVIDIA_API_KEY is not configured for fallback."
    ) from gemini_error


def _prompt_for_pdf(
    anchor_pdf: str,
    sampled_pages: list[PageText],
    per_pdf: int,
    other_pdf: str | None,
    other_pages: list[PageText],
) -> str:
    answerable_count = round(per_pdf * 0.70)
    unanswerable_count = round(per_pdf * 0.15)
    cross_count = max(0, per_pdf - answerable_count - unanswerable_count) if other_pdf else 0
    answerable_count += per_pdf - answerable_count - unanswerable_count - cross_count
    context = [
        {"pdf": anchor_pdf, "page": page.page, "text": page.text[:9000]}
        for page in sampled_pages
    ]
    if other_pdf:
        context.extend({"pdf": other_pdf, "page": page.page, "text": page.text[:6000]}
                       for page in other_pages)
        return (
            "Create benchmark questions only from the supplied PDF page excerpts. Return exactly the requested "
            "counts. For answerable items, evidence must include a verbatim, contiguous quote copied from the "
            "stated page; page numbers are one-based. Questions must be useful and nontrivial; never put the "
            "reference answer verbatim inside its question. Include a concise reference_answer and one question_type "
            "from factual, numeric, definition, method, comparison, multi_hop. multi_hop questions must use only "
            "one PDF. For unanswerable items set answerable "
            "false, leave evidence empty and describe a plausible but unreported detail; these must not be answered "
            "by the provided excerpts. For each cross-document item set cross_document true, use a non-multi_hop "
            "question_type, and include exactly two evidence objects: one quote from the anchor PDF and one from "
            "the second PDF, each with its own exact page and verbatim quote. Do not use evidence from any other PDF. "
            "Do not invent PDF names, page numbers, quotes, or facts.\n"
            f"Generate {answerable_count} answerable single-PDF questions primarily about {anchor_pdf}, "
            f"{unanswerable_count} unanswerable questions, and {cross_count} cross-document questions.\n"
            f"For cross-document items, the only permitted pair is {anchor_pdf} and {other_pdf or '(none)'}.\n"
            f"Page excerpts JSON: {json.dumps(context, ensure_ascii=False)}"
    )


def _verified_records(
    batch: GeneratedBatch,
    pages_by_pdf: dict[str, list[PageText]],
    *,
    anchor_pdf: str,
    other_pdf: str | None,
    next_id: int,
    seen_questions: set[str],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    page_map = {(page.source, page.page): page.text for pages in pages_by_pdf.values() for page in pages}
    rejected = {"quote": 0, "duplicate": 0, "trivial": 0, "schema": 0, "cross_document": 0}
    records: list[dict[str, Any]] = []
    for generated in batch.questions:
        question = normalize_whitespace(generated.question)
        reference = normalize_whitespace(generated.reference_answer)
        question_key = question.casefold()
        if not question or question_key in seen_questions:
            rejected["duplicate"] += 1
            continue
        if (generated.answerable and not reference) or (reference and reference.casefold() in question.casefold()):
            rejected["trivial"] += 1
            continue
        if generated.question_type not in QUESTION_TYPES:
            rejected["schema"] += 1
            continue
        evidence = [item.model_dump() for item in generated.evidence]
        if generated.answerable:
            if not evidence or any(
                (item["pdf"], item["page"]) not in page_map
                or not quote_is_on_page(item["quote"], page_map[(item["pdf"], item["page"])])
                for item in evidence
            ):
                rejected["quote"] += 1
                continue
            evidence_pdfs = {item["pdf"] for item in evidence}
            if generated.cross_document and evidence_pdfs != {anchor_pdf, other_pdf}:
                rejected["cross_document"] += 1
                continue
            if generated.question_type == "multi_hop" and generated.cross_document:
                rejected["schema"] += 1
                continue
            if not generated.cross_document and {item["pdf"] for item in evidence} != {anchor_pdf}:
                rejected["schema"] += 1
                continue
            if generated.cross_document:
                evidence.sort(key=lambda item: item["pdf"] != anchor_pdf)
        elif evidence:
            rejected["schema"] += 1
            continue
        seen_questions.add(question_key)
        primary = evidence[0] if evidence else {"pdf": None, "page": None, "quote": None}
        records.append({
            "id": f"real-{next_id + len(records):04d}",
            "split": "",
            "question": question,
            "reference_answer": reference,
            "answerable": generated.answerable,
            "question_type": generated.question_type,
            "cross_document": generated.cross_document,
            "evidence_pdf": primary["pdf"],
            "evidence_page": primary["page"],
            "evidence_quote": primary["quote"],
            "evidence": evidence,
            "verified_by": "auto",
        })
    return records, rejected


def generate_benchmark(
    pdf_dir: Path = DEFAULT_PDF_DIR,
    per_pdf: int = 12,
    *,
    force: bool = False,
    seed: int = DEFAULT_SEED,
    client: Any | None = None,
    output_dir: Path = OUTPUT_DIR,
) -> dict[str, Any]:
    pdf_paths = sorted(pdf_dir.glob("*.pdf"), key=lambda path: path.name.casefold())
    if not pdf_paths:
        raise FileNotFoundError(
            f"No PDFs found in {pdf_dir}. Put 3-4 text-based research PDFs in that folder and retry."
        )
    if per_pdf < 1:
        raise ValueError("--per-pdf must be at least 1")
    benchmark_path = output_dir / "benchmark_real.jsonl"
    metadata_path = output_dir / "metadata.json"
    review_path = output_dir / "review.md"
    if not force and any(path.exists() for path in (benchmark_path, metadata_path, review_path)):
        raise FileExistsError("Real benchmark output already exists; pass --force to replace it.")
    pages_by_pdf: dict[str, list[PageText]] = {}
    for path in pdf_paths:
        pages_by_pdf[path.name] = extract_pdf(path.read_bytes(), path.name)
    model_name = _model_name()
    rng = random.Random(seed)
    all_records: list[dict[str, Any]] = []
    rejected_counts = {key: 0 for key in ("quote", "duplicate", "trivial", "schema", "cross_document")}
    seen_questions: set[str] = set()
    generator_models: set[str] = set()
    generator_providers: set[str] = set()
    names = list(pages_by_pdf)
    for index, name in enumerate(names):
        other_name = names[(index + 1) % len(names)] if len(names) > 1 else None
        sampled = _sample_pages(pages_by_pdf[name], rng)
        other_sampled = _sample_pages(pages_by_pdf[other_name], rng) if other_name else []
        prompt = _prompt_for_pdf(name, sampled, per_pdf, other_name, other_sampled)
        batch, provider, actual_model = _generate_batch(prompt, model_name, client)
        generator_providers.add(provider)
        generator_models.add(actual_model)
        records, rejected = _verified_records(
            batch, pages_by_pdf, anchor_pdf=name, other_pdf=other_name,
            next_id=len(all_records) + 1,
            seen_questions=seen_questions,
        )
        all_records.extend(records)
        for key, count in rejected.items():
            rejected_counts[key] += count
    assigned = stratified_split(all_records, seed)
    if not assigned:
        raise RuntimeError("No questions passed automatic verification; no benchmark files were written.")
    output_dir.mkdir(parents=True, exist_ok=True)
    schema_path = ROOT / "data" / "evaluation" / "real_schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    _validate_records(assigned, schema)
    benchmark_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in assigned), encoding="utf-8")
    metadata = {
        "seed": seed,
        "generator_model": model_name,
        "generator_models_used": sorted(generator_models),
        "generator_providers_used": sorted(generator_providers),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "pdfs": [path.name for path in pdf_paths],
        "per_pdf_requested": per_pdf,
        "per_pdf_accepted": {
            name: sum(record["evidence_pdf"] == name for record in assigned)
            for name in names
        },
        "record_count": len(assigned),
        "split_counts": {split: sum(row["split"] == split for row in assigned) for split in ("dev", "heldout")},
        "rejected_counts": rejected_counts,
        "dataset_version": hashlib.sha256(benchmark_path.read_bytes()).hexdigest()[:16],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    _write_review(review_path, assigned, metadata)
    return metadata


def _validate_records(records: list[dict[str, Any]], schema: dict[str, Any]) -> None:
    if set(schema["required"]) != set(BenchmarkRecord.model_fields):
        raise ValueError("real_schema.json fields do not match the benchmark validator")
    for record in records:
        BenchmarkRecord.model_validate(record)


def _write_review(path: Path, records: list[dict[str, Any]], metadata: dict[str, Any]) -> None:
    lines = ["# Generated benchmark review", "", f"Seed: `{metadata['seed']}`  ",
             f"Generator: `{metadata['generator_model']}`", ""]
    for record in records:
        lines.extend([f"## {record['id']} ({record['split']})", "",
                      f"**Question:** {record['question']}", "",
                      f"**Answerable:** {record['answerable']}  ",
                      f"**Type:** {record['question_type']}  ",
                      f"**Verified by:** {record['verified_by']}  ",
                      f"**Reference:** {record['reference_answer'] or '(none)' }", ""])
        if record["evidence"]:
            for item in record["evidence"]:
                lines.extend([f"**Evidence:** {item['pdf']} · page {item['page']}", "",
                              f"> {item['quote']}", ""])
        else:
            lines.extend(["**Evidence:** none (generated as unanswerable)", ""])
        lines.append("---")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf-dir", type=Path, default=DEFAULT_PDF_DIR)
    parser.add_argument("--per-pdf", type=int, default=12)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--force", action="store_true", help="Replace existing benchmark outputs")
    args = parser.parse_args()
    try:
        metadata = generate_benchmark(args.pdf_dir, args.per_pdf, force=args.force, seed=args.seed)
    except (FileNotFoundError, FileExistsError, ValueError, RuntimeError, ValidationError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
