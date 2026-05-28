"""CraftImageGenerator: reference-aware image generation.

Delegates image generation to a pluggable
`ImageGenBackend` (default = `ChatImageBackend` wrapping `ModelRouter`).
The indirection
exists so that callers can swap GPT-Image / Vertex / etc. without
touching the rest of the pipeline.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from crafter.generation.craft.backends import ImageGenBackend, ChatImageBackend
from crafter.shared.model_router import ModelRouter

logger = logging.getLogger(__name__)


class CraftImageGenerator:
    """Generates images with reference-image context.

    Args:
        router: ModelRouter. Always required so that downstream code
                (max_retries from config, etc.) keeps working.
        backend: optional ImageGenBackend. If None, a ChatImageBackend
                 is built from `router`.
    """

    def __init__(
        self,
        router: ModelRouter,
        backend: Optional[ImageGenBackend] = None,
    ):
        self.router = router
        self.backend: ImageGenBackend = backend or ChatImageBackend(router)

    def generate(
        self,
        prompt: str,
        reference_images: list[bytes],
        style_prefix: str = "",
        model: Optional[str] = None,
        debug_dir: Optional[str] = None,
    ) -> Optional[bytes]:
        """Generate an image using style prefix + prompt with reference images.

        Always routes through chat-completions multimodal — refer images
        attach as multimodal context. The refer-aware behavior is encoded
        in the prompt (via `figure_spec.build_edit_instruction` /
        `_REFER_IMAGE_HINTS`), not the backend route.

        Args:
            prompt: Figure-specific generation prompt.
            reference_images: Raw bytes of reference images for context.
            style_prefix: Style template to prepend to the prompt.
            model: Override generation model.
            debug_dir: If set, save debug JSON on failure.

        Returns:
            Generated image bytes, or None on failure.
        """
        full_prompt = prompt
        if style_prefix:
            full_prompt = style_prefix + "\n\n" + prompt

        logger.info(
            f"Generating image: {len(reference_images)} reference(s), "
            f"prompt length={len(full_prompt)}"
        )

        image_bytes = self.backend.generate(
            full_prompt,
            reference_images=reference_images,
            model=model,
            max_retries=self.router.config.max_retries_per_generation,
        )

        if image_bytes is None and debug_dir:
            debug_path = Path(debug_dir) / "last_failed_prompt.txt"
            debug_path.parent.mkdir(parents=True, exist_ok=True)
            debug_path.write_text(full_prompt, encoding="utf-8")
            logger.warning(f"Generation failed, prompt saved to {debug_path}")

        return image_bytes

    def generate_with_current_image(
        self,
        prompt: str,
        current_image: bytes,
        reference_images: list[bytes],
        style_prefix: str = "",
        model: Optional[str] = None,
    ) -> Optional[bytes]:
        """Refine an existing image by including it alongside references.

        Sends current image + reference images + refinement prompt.
        Useful for iterative refinement where the model sees its previous
        output. Always uses fresh-generation route (chat-completions
        multimodal); never edit-mode.
        """
        full_prompt = prompt
        if style_prefix:
            full_prompt = style_prefix + "\n\n" + prompt

        all_images = [current_image] + reference_images
        return self.backend.generate(
            full_prompt,
            reference_images=all_images,
            model=model,
            max_retries=self.router.config.max_retries_per_generation,
        )
