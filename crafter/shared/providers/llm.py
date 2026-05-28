"""LLM/VLM providers — OpenAI-compatible gateway."""
from __future__ import annotations

import logging
import time
from typing import Optional

import requests

from ._config import LLMConfig
from .base import LLMProvider, LLMResponse

logger = logging.getLogger(__name__)


class OpenRouterLLM(LLMProvider):
    """OpenRouter OpenAI-compatible chat completion gateway.

    Supports both text and vision messages. Same wire format as OpenAI's
    /chat/completions, but routed through https://openrouter.ai/api/v1.
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        self.cfg = config or LLMConfig()
        if not self.cfg.api_key:
            raise RuntimeError(
                "OpenRouter LLM provider needs api_key (set OPENROUTER_API_KEY env "
                "or pass via LLMConfig).")

    def chat(
        self,
        messages: list,
        *,
        model: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: Optional[float] = None,
        label: Optional[str] = None,
    ) -> LLMResponse:
        model = model or self.cfg.default_model
        payload: dict = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if temperature is not None:
            payload["temperature"] = temperature

        last_err = None
        for attempt in range(self.cfg.max_retries + 1):
            t0 = time.time()
            try:
                r = requests.post(
                    f"{self.cfg.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.cfg.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=self.cfg.timeout_s,
                )
                if r.status_code in (502, 503, 504):
                    last_err = f"HTTP {r.status_code}: {r.text[:200]}"
                    if attempt < self.cfg.max_retries:
                        time.sleep(8.0 * (2 ** attempt))
                        continue
                r.raise_for_status()
                data = r.json()
                usage = data.get("usage") or {}
                elapsed = time.time() - t0
                return LLMResponse(
                    text=data["choices"][0]["message"]["content"],
                    model=model,
                    elapsed_s=round(elapsed, 2),
                    prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
                    completion_tokens=int(usage.get("completion_tokens", 0) or 0),
                    total_tokens=int(usage.get("total_tokens", 0) or 0),
                    raw=data,
                )
            except (requests.Timeout, requests.ConnectionError) as e:
                last_err = f"{type(e).__name__}: {e}"
                if attempt < self.cfg.max_retries:
                    time.sleep(8.0 * (2 ** attempt))
                    continue
            except requests.HTTPError:
                raise
        raise RuntimeError(
            f"OpenRouterLLM call failed after {self.cfg.max_retries + 1} attempts: {last_err}")


class OpenAILLM(LLMProvider):
    """OpenAI direct API (drop-in alternative to OpenRouter)."""

    def __init__(self, *, api_key: str, base_url: str = "https://api.openai.com/v1",
                 default_model: str = "gpt-4o", timeout_s: int = 600,
                 max_retries: int = 3):
        self._cfg = LLMConfig(
            api_key=api_key,
            base_url=base_url,
            default_model=default_model,
            timeout_s=timeout_s,
            max_retries=max_retries,
        )
        # Reuse OpenRouterLLM impl (same wire format)
        self._impl = OpenRouterLLM(self._cfg)

    def chat(self, messages, *, model=None, max_tokens=4096,
             temperature=None, label=None):
        return self._impl.chat(
            messages, model=model, max_tokens=max_tokens,
            temperature=temperature, label=label)


# ============================================================
# Default singleton (for facade-style imports)
# ============================================================

_DEFAULT_LLM: Optional[LLMProvider] = None


def get_default_llm() -> LLMProvider:
    """Return process-wide default LLM provider (lazy singleton)."""
    global _DEFAULT_LLM
    if _DEFAULT_LLM is None:
        _DEFAULT_LLM = OpenRouterLLM()
    return _DEFAULT_LLM


def set_default_llm(provider: LLMProvider) -> None:
    """Override the default provider (e.g. for testing or swap to OpenAI)."""
    global _DEFAULT_LLM
    _DEFAULT_LLM = provider
