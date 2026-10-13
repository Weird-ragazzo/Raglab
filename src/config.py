"""Environment-backed configuration with safe secret handling."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")


def _api_key() -> str | None:
    key = os.getenv("GEMINI_API_KEY")
    if key:
        return key
    # Streamlit Community Cloud provides secrets through st.secrets, not env vars.
    try:
        import streamlit as st

        return str(st.secrets.get("GEMINI_API_KEY", "")) or None
    except Exception:
        return None


@dataclass(frozen=True)
class Settings:
    gemini_api_key: str | None = field(default_factory=_api_key)
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    max_upload_mb: int = int(os.getenv("MAX_UPLOAD_MB", "25"))
    max_pdf_pages: int = int(os.getenv("MAX_PDF_PAGES", "500"))
    default_chunk_size: int = int(os.getenv("CHUNK_SIZE", "900"))
    default_chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "150"))


settings = Settings()

