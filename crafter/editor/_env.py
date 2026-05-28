"""Central env-var resolver for the editor sub-package.

Single source of truth for the credentials and endpoint the editor pipeline
needs. Everything routes through an OpenAI-compatible endpoint (OpenRouter by
default).

    OPENROUTER_API_KEY    — API key for chat / VLM / image-edit calls
    OPENROUTER_API_BASE   — base URL (default https://openrouter.ai/api/v1)
    SAM3_SERVER_URL       — grounding server used by the processing phase
"""
from __future__ import annotations

import os

_DEFAULT_BASE = "https://openrouter.ai/api/v1"


def api_key() -> str:
    """Chat / VLM API key. Raises if unset."""
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY env var is required.")
    return key


def api_endpoint() -> str:
    """Chat / VLM base URL."""
    return os.environ.get("OPENROUTER_API_BASE") or _DEFAULT_BASE


def image_edit_api_key() -> str:
    """Image-edit API key (same provider as chat)."""
    return api_key()


def image_edit_endpoint() -> str:
    """Image-edit base URL (same provider as chat)."""
    return api_endpoint()


def sam3_url() -> str:
    """SAM3 server URL. Reads env or a discovery file. Raises if neither set."""
    from pathlib import Path
    url = os.environ.get("SAM3_SERVER_URL")
    if url:
        return url
    home = Path(os.environ.get("CRAFTER_HOME", os.getcwd()))
    discovery = home / ".sam3_server_url"
    if discovery.exists():
        return discovery.read_text().strip()
    raise RuntimeError(
        "SAM3_SERVER_URL env var is required "
        f"(or a discovery file at {discovery})."
    )


def rmbg_url() -> str | None:
    """Background-removal server URL, or None if not configured."""
    return os.environ.get("RMBG_SERVER_URL")
