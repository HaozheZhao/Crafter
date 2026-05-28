"""Multi-model API client using OpenAI-compatible interface."""
from __future__ import annotations

import base64
import io
import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from openai import OpenAI
from PIL import Image

logger = logging.getLogger(__name__)


# Models known to reject an explicit ``temperature`` request parameter when
# routed via the OpenRouter gateway (the gateway proxies to anthropic / openai
# upstreams, and certain newer reasoning models 400 on temperature).
_TEMPERATURE_REJECTING_PREFIXES = (
    "claude-opus-4",
    "anthropic/claude-4",
    "anthropic/claude-opus",
    "openai/o1",
    "openai/o3",
)


def _model_rejects_temperature(model: str) -> bool:
    m = (model or "").lower()
    return any(m.startswith(p) for p in _TEMPERATURE_REJECTING_PREFIXES)


class ModelRouter:
    """Routes requests to different VLM/LLM models via a single API endpoint."""

    def __init__(self, base_url: str, api_key: str, timeout: int = 300):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.timeout = timeout

    def chat(
        self,
        messages: list[dict],
        model: str,
        temperature: float = 0.2,
        max_tokens: int = 8000,
        reasoning_effort: str | None = None,
    ) -> str:
        """Send chat completion and return text content.

        Args:
            reasoning_effort: "xhigh", "high", "medium", "low" or None.
                When set, enables reasoning mode for deeper thinking.
        """
        kwargs = dict(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            timeout=self.timeout,
        )
        # Some gateway-routed models (claude-opus-4.x, certain "thinking" models)
        # reject an explicit ``temperature`` parameter outright with HTTP 400.
        # Silently drop it for those — the model uses its own sampling defaults.
        if not _model_rejects_temperature(model):
            kwargs["temperature"] = temperature
        if reasoning_effort:
            kwargs["extra_body"] = {
                "reasoning": {"effort": reasoning_effort, "summary": "detailed"}
            }
        resp = self.client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""

    def chat_with_image(
        self,
        prompt: str,
        image_path: str,
        model: str,
        temperature: float = 0.2,
        max_tokens: int = 8000,
        max_dim: int = 2048,
    ) -> str:
        """Send image + text prompt, return text response."""
        b64 = encode_image(image_path, max_dim=max_dim)
        messages = [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            {"type": "text", "text": prompt},
        ]}]
        return self.chat(messages, model=model, temperature=temperature, max_tokens=max_tokens)

    def chat_with_images(
        self,
        prompt: str,
        image_paths: list[str],
        model: str,
        temperature: float = 0.2,
        max_tokens: int = 8000,
        max_dim: int = 1200,
    ) -> str:
        """Send multiple images + prompt."""
        content = []
        for p in image_paths:
            b64 = encode_image(p, max_dim=max_dim)
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
        content.append({"type": "text", "text": prompt})
        return self.chat([{"role": "user", "content": content}], model=model,
                         temperature=temperature, max_tokens=max_tokens)

    def chat_parallel(
        self,
        messages: list[dict],
        models: list[str],
        temperature: float = 0.2,
        max_tokens: int = 4000,
    ) -> dict[str, str]:
        """Send same request to multiple models in parallel. Returns {model: response}."""
        results = {}
        with ThreadPoolExecutor(max_workers=len(models)) as pool:
            futures = {
                pool.submit(self.chat, messages, m, temperature, max_tokens): m
                for m in models
            }
            for future in as_completed(futures):
                model = futures[future]
                try:
                    results[model] = future.result()
                except Exception as e:
                    logger.warning(f"Model {model} failed: {e}")
                    results[model] = ""
        return results

    def chat_with_retry(
        self,
        messages: list[dict],
        model: str,
        max_retries: int = 3,
        delay: int = 10,
        **kwargs,
    ) -> str:
        """Chat with exponential backoff retry."""
        for attempt in range(max_retries):
            try:
                return self.chat(messages, model=model, **kwargs)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {e}. Retrying in {delay}s...")
                time.sleep(delay)
                delay *= 2
        return ""


def encode_image(path: str, max_dim: int = 2048, quality: int = 90) -> str:
    """Encode image to base64 JPEG, resizing if needed."""
    img = Image.open(path).convert("RGB")
    if max(img.size) > max_dim:
        ratio = max_dim / max(img.size)
        img = img.resize((int(img.width * ratio), int(img.height * ratio)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


def parse_json_response(text: str) -> dict | list:
    """Extract JSON from LLM response, handling markdown code blocks."""
    text = text.strip()
    # Extract from code block
    if "```" in text:
        blocks = re.findall(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
        if blocks:
            text = blocks[0].strip()
    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try json_repair
    try:
        import json_repair
        return json_repair.loads(text)
    except Exception:
        pass
    # Last resort: find first [ or { and parse from there
    for i, c in enumerate(text):
        if c in "[{":
            try:
                return json.loads(text[i:])
            except json.JSONDecodeError:
                continue
    raise ValueError(f"Cannot parse JSON from: {text[:200]}")
