"""Small OpenAI-compatible client for NVIDIA NIM chat completions."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from src.config import settings


@dataclass(frozen=True)
class NvidiaResponse:
    text: str
    model: str
    input_tokens: int | None
    output_tokens: int | None


def generate_json(prompt: str, *, model_name: str | None = None, temperature: float = 0.1) -> NvidiaResponse:
    if not settings.nvidia_api_key:
        raise RuntimeError("NVIDIA_API_KEY is not configured.")
    model = model_name or settings.nvidia_model
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": 8192,
        "response_format": {"type": "json_object"},
    }).encode("utf-8")
    request = Request(
        settings.nvidia_base_url,
        data=payload,
        headers={
            "Authorization": f"Bearer {settings.nvidia_api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )
    for attempt in range(4):
        try:
            with urlopen(request, timeout=120) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
            choices = response_payload.get("choices", [])
            if not choices or not isinstance(choices[0].get("message", {}).get("content"), str):
                raise RuntimeError("NVIDIA returned an empty or malformed chat completion.")
            usage = response_payload.get("usage", {})
            return NvidiaResponse(
                choices[0]["message"]["content"],
                str(response_payload.get("model") or model),
                usage.get("prompt_tokens"),
                usage.get("completion_tokens"),
            )
        except HTTPError as exc:
            retryable = exc.code in {429, 500, 502, 503, 504}
            if not retryable or attempt == 3:
                detail = exc.read().decode("utf-8", errors="replace")[:500]
                raise RuntimeError(f"NVIDIA API returned HTTP {exc.code}: {detail}") from exc
        except (URLError, TimeoutError) as exc:
            if attempt == 3:
                reason = getattr(exc, "reason", str(exc))
                raise RuntimeError(f"NVIDIA API request failed: {reason}") from exc
        if attempt < 3:
            time.sleep(min(2**attempt, 8))
    raise RuntimeError("NVIDIA API request failed after retries.")