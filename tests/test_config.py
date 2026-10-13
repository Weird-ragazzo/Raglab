from src.config import ROOT, settings


def test_secret_is_optional_and_model_configured():
    assert settings.gemini_model
    assert settings.max_upload_mb > 0


def test_secret_example_is_blank_and_ignored_locally():
    example = (ROOT / ".env.example").read_text()
    assert "GEMINI_API_KEY=\n" in example
    ignore = (ROOT / ".gitignore").read_text().splitlines()
    assert ".env" in ignore
    assert ".venv/" in ignore

