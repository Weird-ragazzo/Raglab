"""PDF extraction and character-based chunking with page provenance."""

from __future__ import annotations

import hashlib
import io
import re
from dataclasses import asdict, dataclass
from typing import BinaryIO

import fitz


@dataclass(frozen=True)
class PageText:
    source: str
    page: int
    text: str


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    source: str
    page: int
    text: str

    def as_dict(self) -> dict[str, str | int]:
        return asdict(self)


class DocumentError(ValueError):
    """Invalid, unsafe, or unusable uploaded document."""


def extract_pdf(
    data: bytes | BinaryIO,
    filename: str,
    *,
    max_bytes: int = 25 * 1024 * 1024,
    max_pages: int = 500,
) -> list[PageText]:
    """Extract nonempty pages from a PDF, using one-based page numbers."""
    if not filename.lower().endswith(".pdf"):
        raise DocumentError("Upload a PDF file.")
    if hasattr(data, "read"):
        raw = data.read(max_bytes + 1)  # type: ignore[union-attr]
    else:
        raw = data
    if len(raw) > max_bytes:
        raise DocumentError(f"PDF exceeds the {max_bytes // (1024 * 1024)} MB limit.")
    try:
        pdf = fitz.open(stream=raw, filetype="pdf")
    except (fitz.FileDataError, ValueError) as exc:
        raise DocumentError("The uploaded file is not a readable PDF.") from exc
    with pdf:
        if pdf.page_count == 0:
            raise DocumentError("The PDF has no pages.")
        if pdf.page_count > max_pages:
            raise DocumentError(f"PDF exceeds the {max_pages}-page limit.")
        pages = [
            PageText(filename, number, text)
            for number, page in enumerate(pdf, start=1)
            if (text := re.sub(r"\s+", " ", page.get_text("text")).strip())
        ]
    if not pages:
        raise DocumentError(
            "No selectable text was found. This may be a scanned PDF; OCR is not enabled."
        )
    return pages


def chunk_pages(
    pages: list[PageText], *, chunk_size: int = 900, overlap: int = 150
) -> list[Chunk]:
    """Split pages independently so every chunk retains exact page provenance."""
    if chunk_size < 100:
        raise ValueError("chunk_size must be at least 100 characters")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be nonnegative and smaller than chunk_size")
    step = chunk_size - overlap
    chunks: list[Chunk] = []
    for page in pages:
        text = page.text.strip()
        if not text:
            continue
        for start in range(0, len(text), step):
            excerpt = text[start : start + chunk_size].strip()
            if excerpt:
                digest = hashlib.sha256(
                    f"{page.source}\0{page.page}\0{start}\0{excerpt}".encode()
                ).hexdigest()[:16]
                chunks.append(Chunk(digest, page.source, page.page, excerpt))
            if start + chunk_size >= len(text):
                break
    return chunks


def load_pdf_bytes(data: bytes, filename: str, max_bytes: int, max_pages: int) -> list[PageText]:
    return extract_pdf(io.BytesIO(data), filename, max_bytes=max_bytes, max_pages=max_pages)

