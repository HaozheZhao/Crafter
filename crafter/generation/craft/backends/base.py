"""ImageGenBackend protocol.

A backend produces image bytes from a prompt + optional reference images
through a single chat-completions multimodal route.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, runtime_checkable


@dataclass
class GenerateResult:
    """Result of a backend.generate() call."""
    image_bytes: Optional[bytes]
    # The model id that was actually used (after any router-level fallback).
    model_used: str = ""
    # Backend-specific metadata: latency, retry count, route taken, etc.
    extra: Optional[dict] = None


@runtime_checkable
class ImageGenBackend(Protocol):
    """Protocol for image-generation backends.

    Pass `reference_images` (zero, one, or many) for multimodal context.
    Returns image bytes, or None on irrecoverable failure (caller decides
    retry policy at a higher level).
    """

    name: str

    def generate(
        self,
        prompt: str,
        *,
        reference_images: list[bytes] = ...,
        model: Optional[str] = None,
        max_retries: int = 3,
    ) -> Optional[bytes]:
        ...
