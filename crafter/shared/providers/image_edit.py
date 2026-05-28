"""Instructable image-edit provider (gpt-image-2 via an OpenAI-compatible API).

The edit is performed through the chat-completions multimodal route: the base
image and the instruction are sent together, and the edited image is returned
as a base64 data URL in the assistant message.
"""
from __future__ import annotations

import base64
import io
import logging
import time
from pathlib import Path
from typing import Optional

from openai import OpenAI
from PIL import Image

from ._config import ImageEditConfig
from .base import ImageEditProvider, ImageEditResponse

logger = logging.getLogger(__name__)


def _extract_image(response) -> Optional[bytes]:
    msg = response.choices[0].message
    imgs = getattr(msg, "images", None)
    if imgs:
        d = imgs[0]
        url = d.get("image_url", {}).get("url", "") if isinstance(d, dict) else getattr(d, "url", "")
        if url.startswith("data:"):
            return base64.b64decode(url.split(",", 1)[1])
    content = msg.content
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and part.get("type") == "image_url":
                url = part.get("image_url", {}).get("url", "")
                if url.startswith("data:"):
                    return base64.b64decode(url.split(",", 1)[1])
    return None


class GptImageEditor(ImageEditProvider):
    """Instructable image editor backed by gpt-image-2 over chat-completions."""

    def __init__(self, config: Optional[ImageEditConfig] = None):
        self.cfg = config or ImageEditConfig()
        if not self.cfg.api_key:
            raise RuntimeError(
                "Image-edit provider needs OPENROUTER_API_KEY in the environment.")
        self.client = OpenAI(api_key=self.cfg.api_key, base_url=self.cfg.base_url,
                             timeout=self.cfg.timeout_s)

    def edit(self, orig_path: Path, prompt: str, out_path: Path,
             *, quality: str = "high") -> ImageEditResponse:
        img = Image.open(orig_path).convert("RGB")
        W, H = img.size
        buf = io.BytesIO(); img.save(buf, format="PNG")
        png_bytes = buf.getvalue()

        # Infer the closest supported aspect ratio so the edit preserves shape.
        ar = W / max(1, H)
        options = [("21:9", 21/9), ("16:9", 16/9), ("3:2", 3/2), ("4:3", 4/3),
                    ("5:4", 5/4), ("1:1", 1.0), ("4:5", 4/5), ("3:4", 3/4),
                    ("2:3", 2/3), ("9:16", 9/16)]
        aspect_ratio = min(options, key=lambda kv: abs(kv[1] - ar))[0]

        # Direct-Azure fast path: if the four AZURE_OPENAI_* env vars are set,
        # the call bypasses OpenRouter and goes to the user's Azure deployment.
        from crafter.shared.model_router import (
            _azure_image_creds, _azure_call_edit, _azure_image_size,
            _is_gpt_image_model,
        )
        creds = _azure_image_creds() if _is_gpt_image_model(self.cfg.model) else None

        t0 = time.time()
        last_err = None
        if creds:
            out_bytes = _azure_call_edit(
                creds, png_bytes, prompt,
                size=_azure_image_size(aspect_ratio, "2K"),
                quality=quality, max_retries=self.cfg.max_retries,
            )
            if out_bytes:
                Image.open(io.BytesIO(out_bytes)).convert("RGB") \
                    .resize((W, H), Image.LANCZOS).save(out_path)
                return ImageEditResponse(
                    status="ok", out_path=out_path,
                    elapsed_s=round(time.time() - t0, 1))
            return ImageEditResponse(
                status="exhausted", out_path=out_path,
                elapsed_s=round(time.time() - t0, 1),
                error="azure edit failed")

        data_url = f"data:image/png;base64,{base64.b64encode(png_bytes).decode()}"
        messages = [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": data_url}},
            {"type": "text", "text": prompt},
        ]}]
        extra_body = {
            "modalities": ["image", "text"],
            "image_config": {"aspect_ratio": aspect_ratio, "image_size": "2K"},
        }
        for attempt in range(self.cfg.max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.cfg.model,
                    messages=messages,
                    max_tokens=4096,
                    extra_body=extra_body,
                )
                out_bytes = _extract_image(resp)
                if out_bytes:
                    Image.open(io.BytesIO(out_bytes)).convert("RGB") \
                        .resize((W, H), Image.LANCZOS).save(out_path)
                    return ImageEditResponse(
                        status="ok", out_path=out_path,
                        elapsed_s=round(time.time() - t0, 1))
                last_err = "no image in response"
            except Exception as e:
                last_err = f"{type(e).__name__}: {e}"
                logger.warning("[image_edit] attempt %d/%d failed: %s",
                               attempt + 1, self.cfg.max_retries, last_err)
            if attempt < self.cfg.max_retries - 1:
                time.sleep(10 + attempt * 15)
        return ImageEditResponse(
            status="exhausted", out_path=out_path,
            elapsed_s=round(time.time() - t0, 1),
            error=f"all {self.cfg.max_retries} attempts failed: {last_err}")


_DEFAULT: Optional[ImageEditProvider] = None


def get_default_image_edit() -> ImageEditProvider:
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = GptImageEditor()
    return _DEFAULT


def set_default_image_edit(provider: ImageEditProvider) -> None:
    global _DEFAULT
    _DEFAULT = provider
