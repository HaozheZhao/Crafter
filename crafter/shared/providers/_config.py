"""Centralized config + env-var schema for the editor providers.

Every environment variable the editor providers read is listed here.
All calls go through an OpenAI-compatible endpoint (OpenRouter by default).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

OPENROUTER_BASE = "https://openrouter.ai/api/v1"


def _env(key: str, default: Optional[str] = None) -> Optional[str]:
    v = os.environ.get(key)
    return v if v is not None and v != "" else default


def _yaml_model(slot: str, fallback: str) -> str:
    """Read ``models.<slot>`` from the active config yaml, falling back."""
    try:
        from crafter.shared.config import load_config
        return load_config().models.get(slot) or fallback
    except Exception:
        return fallback


@dataclass
class LLMConfig:
    """Chat / VLM provider config.

    Env vars:
      OPENROUTER_API_KEY        — API key
      OPENROUTER_API_BASE       — base URL (default OpenRouter)
      CRAFTER_EDITOR_LLM_MODEL  — default model
      CRAFTER_LLM_TIMEOUT       — per-request timeout seconds (default 600)
    """
    api_key: Optional[str] = field(default_factory=lambda: _env("OPENROUTER_API_KEY"))
    base_url: str = field(default_factory=lambda: _env("OPENROUTER_API_BASE", OPENROUTER_BASE) or OPENROUTER_BASE)
    default_model: str = field(default_factory=lambda: _env(
        "CRAFTER_EDITOR_LLM_MODEL") or _yaml_model("llm", "openai/gpt-5.5"))
    timeout_s: int = field(default_factory=lambda: int(_env("CRAFTER_LLM_TIMEOUT", "600") or "600"))
    max_retries: int = field(default_factory=lambda: int(_env("CRAFTER_LLM_MAX_RETRIES", "3") or "3"))


@dataclass
class ImageEditConfig:
    """Instructable image-edit provider config (gpt-image-2 via OpenRouter).

    The edit goes through the chat-completions multimodal route with
    ``modalities=["image", "text"]``.

    Env vars:
      OPENROUTER_API_KEY              — API key
      OPENROUTER_API_BASE            — base URL (default OpenRouter)
      CRAFTER_IMAGE_EDIT_MODEL       — model id (default openai/gpt-5.4-image-2)
      CRAFTER_IMAGE_EDIT_TIMEOUT     — per-request timeout seconds (default 420)
      CRAFTER_IMAGE_EDIT_MAX_RETRIES — retries on rate-limit / timeout (default 5)
    """
    api_key: Optional[str] = field(default_factory=lambda: _env("OPENROUTER_API_KEY"))
    base_url: str = field(default_factory=lambda: _env("OPENROUTER_API_BASE", OPENROUTER_BASE) or OPENROUTER_BASE)
    model: str = field(default_factory=lambda: _env(
        "CRAFTER_IMAGE_EDIT_MODEL") or _yaml_model("generator", "openai/gpt-5.4-image-2"))
    timeout_s: int = field(default_factory=lambda: int(_env("CRAFTER_IMAGE_EDIT_TIMEOUT", "420") or "420"))
    max_retries: int = field(default_factory=lambda: int(_env("CRAFTER_IMAGE_EDIT_MAX_RETRIES", "5") or "5"))


@dataclass
class SAM3Config:
    """SAM3 grounding-segmentation server config.

    Env vars:
      SAM3_SERVER_URL    — http://host:port of a running SAM3 server
      CRAFTER_SAM3_TIMEOUT — per-request timeout (default 300)
    """
    server_url: Optional[str] = field(default_factory=lambda: _env("SAM3_SERVER_URL"))
    timeout_s: int = field(default_factory=lambda: int(_env("CRAFTER_SAM3_TIMEOUT", "300") or "300"))


@dataclass
class RMBGConfig:
    """Background-removal service config (optional fallback path).

    Env vars:
      RMBG_SERVER_URL    — http://host:port of a running RMBG service
      CRAFTER_RMBG_TIMEOUT — per-request timeout (default 60)
    """
    server_url: str = field(default_factory=lambda: _env(
        "RMBG_SERVER_URL", "http://localhost:9101") or "http://localhost:9101")
    timeout_s: int = field(default_factory=lambda: int(_env("CRAFTER_RMBG_TIMEOUT", "60") or "60"))


@dataclass
class StageAConfig:
    """Extraction-phase options.

    Env vars:
      CRAFTER_EXTRACT_HARNESS    — '1' to use the iterative harness (default 0)
      CRAFTER_EXTRACT_ITER_MODEL — iteration model (default gemini-3.1-pro-preview)
      CRAFTER_EXTRACT_FINAL_MODEL— final model (default openai/gpt-5.5)
      CRAFTER_EXTRACT_MAX_ITERS  — max harness iters (default 3)
      CRAFTER_EXTRACT_REVISE_DIFF— '1' to use SEARCH/REPLACE patches (default 0)
    """
    use_harness: bool = field(default_factory=lambda: _env("CRAFTER_EXTRACT_HARNESS", "0") == "1")
    iter_model: str = field(default_factory=lambda: _env(
        "CRAFTER_EXTRACT_ITER_MODEL", "google/gemini-3.1-pro-preview") or "google/gemini-3.1-pro-preview")
    final_model: str = field(default_factory=lambda: _env(
        "CRAFTER_EXTRACT_FINAL_MODEL", "openai/gpt-5.5") or "openai/gpt-5.5")
    max_iters: int = field(default_factory=lambda: int(_env("CRAFTER_EXTRACT_MAX_ITERS", "3") or "3"))
    revise_diff: bool = field(default_factory=lambda: _env("CRAFTER_EXTRACT_REVISE_DIFF", "0") == "1")
    use_gpt_image2: bool = True  # set False for SAM3 + RMBG-only fallback


@dataclass
class StageBConfig:
    """Composition-phase options.

    Env vars:
      CRAFTER_COMPOSE_MODEL    — model for skeleton/refine/judge (default openai/gpt-5.5)
      CRAFTER_COMPOSE_MAX_ITER — refine loop iterations (default 4)
    """
    model: str = field(default_factory=lambda: _env(
        "CRAFTER_COMPOSE_MODEL", "openai/gpt-5.5") or "openai/gpt-5.5")
    max_iter: int = field(default_factory=lambda: int(_env("CRAFTER_COMPOSE_MAX_ITER", "4") or "4"))
