"""GPT-Image backend stub (roadmap #6).

Implements the ImageGenBackend protocol surface but raises
NotImplementedError on actual calls. Lets pipeline code already declare
"my backend is GptImageBackend" and fail loudly until the real adapter
lands, instead of silently routing through OPENROUTER-Gemini.

When implementing for real:
- Wire to OpenAI's images.generate / images.edit endpoints (or whichever
  path is canonical at the time).
- Accept the same `reference_images: list[bytes]` arg; multi-image
  conditioning may need to encode them differently.
"""
from __future__ import annotations

from typing import Optional


class GptImageBackend:
    """Stub backend for OpenAI GPT-Image. Not yet implemented."""

    name = "gpt_image"

    def __init__(self, api_key: str = "", model: str = "gpt-image-1"):
        self.api_key = api_key
        self.model = model

    def generate(
        self,
        prompt: str,
        *,
        reference_images: list[bytes] = (),
        model: Optional[str] = None,
        max_retries: int = 3,
    ) -> Optional[bytes]:
        raise NotImplementedError(
            "GptImageBackend is a stub. Implement against the canonical "
            "OpenAI images endpoint when roadmap #6 lands."
        )
