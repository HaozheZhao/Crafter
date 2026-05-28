"""Chat-completions image backend.

Wraps ``ModelRouter`` behind the ``ImageGenBackend`` protocol so callers can
swap in other vendors without changing pipeline code.
"""
from __future__ import annotations

import logging
from typing import Optional

from crafter.shared.model_router import ModelRouter

logger = logging.getLogger(__name__)


class ChatImageBackend:
    """ImageGenBackend backed by chat-completions multimodal image generation.

    Forwards to ModelRouter.generate_image.
    """

    name = "chat_image"

    def __init__(self, router: ModelRouter):
        self.router = router

    def generate(
        self,
        prompt: str,
        *,
        reference_images: list[bytes] = (),
        model: Optional[str] = None,
        max_retries: int = 3,
    ) -> Optional[bytes]:
        return self.router.generate_image(
            prompt=prompt,
            reference_images=list(reference_images),
            model=model,
            max_retries=max_retries,
        )
