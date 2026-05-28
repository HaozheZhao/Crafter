"""Abstract provider interfaces.

A "provider" wraps an external service (LLM API, image-edit API, SAM3
server, RMBG service). All concrete implementations must subclass these
bases and implement the methods. Swap a provider by passing a custom
instance to PaperEditor():

    pe = PaperEditor(llm=MyCustomLLMProvider(), image_edit=...)
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


# ============================================================
# Data containers
# ============================================================

@dataclass
class LLMResponse:
    """Result of one LLM call."""
    text: str
    model: str
    elapsed_s: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    raw: Any = None  # provider-specific raw response (debug)


@dataclass
class ImageEditResponse:
    """Result of one image-edit call (gpt-image-2 etc.)."""
    status: str  # "ok" | error code
    out_path: Path
    elapsed_s: float
    error: Optional[str] = None


@dataclass
class SAM3Result:
    """One bounding box from SAM3 segmentation."""
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    prompt: str
    mask_b64: Optional[str] = None  # base64-encoded PNG, only if requested


# ============================================================
# Provider interfaces
# ============================================================

class LLMProvider(ABC):
    """Chat-completion provider for both pure-text and vision tasks.

    `messages` is the standard OpenAI chat-completion format:
        [{"role": "user", "content": "..."}]
        [{"role": "user", "content": [{"type": "text", "text": "..."},
                                       {"type": "image_url",
                                        "image_url": {"url": "data:image/jpeg;base64,..."}}]}]
    """

    @abstractmethod
    def chat(
        self,
        messages: list,
        *,
        model: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: Optional[float] = None,
        label: Optional[str] = None,
    ) -> LLMResponse:
        """Call the chat endpoint and return the LLM response."""
        ...


class ImageEditProvider(ABC):
    """Image-edit provider (e.g. gpt-image-2).

    Takes an input image and a text instruction, produces an edited image.
    """

    @abstractmethod
    def edit(
        self,
        orig_path: Path,
        prompt: str,
        out_path: Path,
        *,
        quality: str = "high",
    ) -> ImageEditResponse:
        """Edit the image according to `prompt`; write to out_path."""
        ...


class SAM3Provider(ABC):
    """SAM3 grounded-segmentation provider.

    Takes an image + a list of text prompts, returns one bounding box per
    prompt (or none if the prompt isn't found).
    """

    @abstractmethod
    def wait_ready(self, max_wait_s: int = 300, interval_s: int = 10) -> bool:
        """Block until the SAM3 server is ready (or timeout)."""
        ...

    @abstractmethod
    def segment_text(
        self,
        image_path: Path,
        prompts: list[str],
        min_score: float = 0.3,
        return_masks: bool = False,
    ) -> list[list[SAM3Result]]:
        """Per-prompt list of detected bboxes, sorted by score desc."""
        ...

    @abstractmethod
    def segment_bbox(
        self,
        image_path: Path,
        bboxes: list[tuple[float, float, float, float]],
    ) -> list[Optional[str]]:
        """Per-bbox base64-encoded mask PNG (or None if SAM3 fails)."""
        ...


class RMBGProvider(ABC):
    """Background-removal provider (e.g. self-hosted RMBG-2.0)."""

    @abstractmethod
    def remove_background(self, image_bytes: bytes) -> bytes:
        """Take an RGB PNG (bytes), return RGBA PNG with bg made transparent."""
        ...
