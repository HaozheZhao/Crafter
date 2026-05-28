"""ModelRouter: routes each agent role to its model via an OpenAI-compatible API.

A single client (OpenRouter by default) serves every role; the role → model
mapping comes from the active config:

    planner / refiner   designer + prompt writer       (e.g. claude-opus)
    critic              multimodal verifier            (e.g. gemini-pro)
    quick               cheap classification / judging (e.g. gemini-flash-lite)
    generator           image generation               (e.g. nano-banana)

Image generation and editing both go through the chat-completions multimodal
route with ``modalities=["image", "text"]``; the generated image is returned as
a base64 data URL in the assistant message.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Optional

from typing import TYPE_CHECKING

from openai import OpenAI

if TYPE_CHECKING:
    from crafter.generation.core.config import CraftConfig

logger = logging.getLogger(__name__)



# Supported OpenRouter image_config.aspect_ratio enum (gpt-image-2 / nano-banana).
_ASPECT_RATIOS = [
    ("21:9", 21/9), ("16:9", 16/9), ("3:2", 3/2), ("4:3", 4/3),
    ("5:4", 5/4), ("1:1", 1.0), ("4:5", 4/5), ("3:4", 3/4),
    ("2:3", 2/3), ("9:16", 9/16),
]


# ── Optional direct-Azure path for gpt-image-2 ────────────────────────────
# When the four env vars below are set, gpt-image-2 calls bypass OpenRouter
# and go directly to the user's own Azure deployment, which supports arbitrary
# pixel-size requests (1024×512, 4K, etc.). Disabled by default; no Azure
# endpoint URL is hard-coded.

def _azure_image_creds() -> Optional[dict]:
    base = (os.environ.get("AZURE_OPENAI_ENDPOINT") or "").rstrip("/")
    key  = os.environ.get("AZURE_OPENAI_API_KEY") or ""
    dep  = os.environ.get("AZURE_OPENAI_DEPLOYMENT") or ""
    if not (base and key and dep):
        return None
    ver = os.environ.get("AZURE_OPENAI_API_VERSION") or "2025-04-01-preview"
    return {"endpoint": base, "key": key, "deployment": dep, "api_version": ver}


def _is_gpt_image_model(model: str) -> bool:
    m = (model or "").lower()
    return "gpt-image" in m or "gpt-5.4-image" in m


# Map (aspect_ratio, image_size) → an Azure pixel string accepted by
# gpt-image-2's ``/images/generations``. Override via $CRAFTER_AZURE_IMAGE_SIZE
# (e.g. "1024x512") for an exact pixel request.
_AZURE_SIZE_MAP = {
    ("21:9", "1K"): "1536x1024", ("21:9", "2K"): "1536x1024",
    ("16:9", "1K"): "1536x1024", ("16:9", "2K"): "1536x1024",
    ("3:2", "1K"):  "1536x1024", ("3:2", "2K"):  "1536x1024",
    ("4:3", "1K"):  "1536x1024", ("4:3", "2K"):  "1536x1024",
    ("5:4", "1K"):  "1024x1024", ("5:4", "2K"):  "1024x1024",
    ("1:1", "1K"):  "1024x1024", ("1:1", "2K"):  "1024x1024",
    ("4:5", "1K"):  "1024x1024", ("4:5", "2K"):  "1024x1024",
    ("3:4", "1K"):  "1024x1536", ("3:4", "2K"):  "1024x1536",
    ("2:3", "1K"):  "1024x1536", ("2:3", "2K"):  "1024x1536",
    ("9:16", "1K"): "1024x1536", ("9:16", "2K"): "1024x1536",
}


def _azure_image_size(aspect: str, size: str) -> str:
    override = os.environ.get("CRAFTER_AZURE_IMAGE_SIZE")
    return override or _AZURE_SIZE_MAP.get((aspect, size), "1024x1024")


def _azure_call_generate(creds: dict, prompt: str, size: str,
                          quality: str = "high", max_retries: int = 3) -> Optional[bytes]:
    import requests
    url = (f"{creds['endpoint']}/openai/deployments/{creds['deployment']}"
           f"/images/generations?api-version={creds['api_version']}")
    headers = {"api-key": creds["key"], "Content-Type": "application/json"}
    body = {"prompt": prompt, "size": size, "quality": quality}
    for attempt in range(max_retries):
        try:
            r = requests.post(url, headers=headers, json=body, timeout=300)
            if r.status_code == 200:
                return base64.b64decode(r.json()["data"][0]["b64_json"])
            logger.warning(f"[azure] generate {r.status_code}: {r.text[:200]}")
        except Exception as e:
            logger.warning(f"[azure] generate attempt {attempt+1} failed: {e}")
        if attempt < max_retries - 1:
            time.sleep(5)
    return None


def _azure_call_edit(creds: dict, image_bytes: bytes, prompt: str,
                      size: Optional[str] = None, quality: str = "high",
                      max_retries: int = 3) -> Optional[bytes]:
    import requests
    url = (f"{creds['endpoint']}/openai/deployments/{creds['deployment']}"
           f"/images/edits?api-version={creds['api_version']}")
    headers = {"api-key": creds["key"]}
    data = {"prompt": prompt, "quality": quality}
    if size:
        data["size"] = size
    files = {"image": ("img.png", image_bytes, "image/png")}
    for attempt in range(max_retries):
        try:
            r = requests.post(url, headers=headers, files=files, data=data, timeout=300)
            if r.status_code == 200:
                return base64.b64decode(r.json()["data"][0]["b64_json"])
            logger.warning(f"[azure] edit {r.status_code}: {r.text[:200]}")
        except Exception as e:
            logger.warning(f"[azure] edit attempt {attempt+1} failed: {e}")
        if attempt < max_retries - 1:
            time.sleep(5)
    return None


def _aspect_ratio_from_bytes(img_bytes: bytes) -> Optional[str]:
    """Return an OpenRouter ``aspect_ratio`` string matching ``img_bytes``.

    Snaps to the nearest value in the supported enum; returns None if PIL is
    unavailable or the bytes do not decode.
    """
    try:
        from PIL import Image
        import io
        w, h = Image.open(io.BytesIO(img_bytes)).size
    except Exception:
        return None
    ar = w / max(1, h)
    return min(_ASPECT_RATIOS, key=lambda kv: abs(kv[1] - ar))[0]


class ModelRouter:
    """Routes tasks to the configured models via an OpenAI-compatible API."""

    def __init__(self, config: "CraftConfig"):
        self.config = config
        self.client = OpenAI(
            api_key=config.api_key,
            base_url=config.api_base_url,
            timeout=config.api_timeout,
        )

    # ── Text / VLM roles ──────────────────────────────────────────────

    def plan(
        self,
        messages: list[dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: int = 8192,
    ) -> str:
        """Designer role: paper understanding and figure planning."""
        return self._chat(
            messages, model=self.config.planner_model,
            temperature=temperature, max_tokens=max_tokens,
        )

    def refine_prompt(
        self,
        messages: list[dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: int = 8192,
    ) -> str:
        """Reviser role: prompt engineering on the evolving specification."""
        return self._chat(
            messages, model=self.config.refiner_model,
            temperature=temperature, max_tokens=max_tokens,
        )

    def critique_image(
        self,
        image_path: str,
        prompt: str,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Verifier role: analyze one image and return a text critique."""
        model = model or self.config.critic_model
        img_b64 = self._encode_image(image_path)
        mime = self._get_mime(image_path)

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{img_b64}"}},
            ],
        })
        return self._chat(messages, model=model, temperature=0.3, max_tokens=4096)

    def critique_with_references(
        self,
        image_path: str,
        reference_paths: list[str],
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Critique an image alongside reference images for style comparison."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        content_parts = []
        for ref_path in reference_paths[:3]:
            if Path(ref_path).exists():
                ref_b64 = self._encode_image(ref_path)
                mime = self._get_mime(ref_path)
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{ref_b64}"},
                })
        img_b64 = self._encode_image(image_path)
        mime = self._get_mime(image_path)
        content_parts.append({
            "type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{img_b64}"},
        })
        content_parts.append({"type": "text", "text": prompt})

        messages.append({"role": "user", "content": content_parts})
        return self._chat(
            messages, model=self.config.critic_model,
            temperature=0.3, max_tokens=4096,
        )

    # ── Image generation / editing ────────────────────────────────────

    def generate_image(
        self,
        prompt: str,
        reference_images: list[bytes],
        model: Optional[str] = None,
        max_retries: int = 3,
        aspect_ratio: str = "16:9",
        image_size: str = "2K",
    ) -> Optional[bytes]:
        """Generate an image via chat-completions multimodal.

        Reference images (if any) are sent as a STYLE / context signal ahead
        of the prompt. ``aspect_ratio`` and ``image_size`` are forwarded as
        OpenRouter ``image_config`` hints — ignored by models that do not
        support them, honoured by gpt-image-2 / nano-banana.

        Returns the generated image bytes, or None on failure.
        """
        model = model or self.config.generator_model

        # Direct-Azure fast path (gpt-image-2 only, no refer images).
        creds = _azure_image_creds() if _is_gpt_image_model(model) else None
        if creds and not reference_images:
            return _azure_call_generate(
                creds, prompt, size=_azure_image_size(aspect_ratio, image_size),
                max_retries=max_retries,
            )

        content_parts = []
        for ref_bytes in reference_images:
            ref_b64 = base64.b64encode(ref_bytes).decode("utf-8")
            content_parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{ref_b64}"},
            })
        content_parts.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content_parts}]

        return self._image_call(
            messages, model, max_retries,
            label=f"generate ({len(reference_images)} refs)",
            aspect_ratio=aspect_ratio, image_size=image_size,
        )

    def generate_image_edit(
        self,
        image_bytes: bytes,
        prompt: str,
        model: Optional[str] = None,
        max_retries: int = 3,
        aspect_ratio: Optional[str] = None,
        image_size: str = "2K",
    ) -> Optional[bytes]:
        """Edit a single base image given an instruction.

        Sends the base image followed by the edit instruction through the
        chat-completions multimodal route. When ``aspect_ratio`` is omitted
        it is inferred from the base image so the edit preserves shape.
        """
        model = model or self.config.generator_model
        if aspect_ratio is None:
            aspect_ratio = _aspect_ratio_from_bytes(image_bytes) or "1:1"

        # Direct-Azure fast path (gpt-image-2 only).
        creds = _azure_image_creds() if _is_gpt_image_model(model) else None
        if creds:
            return _azure_call_edit(
                creds, image_bytes, prompt,
                size=_azure_image_size(aspect_ratio, image_size),
                max_retries=max_retries,
            )

        img_b64 = base64.b64encode(image_bytes).decode("utf-8")
        messages = [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
            {"type": "text", "text": prompt},
        ]}]
        return self._image_call(
            messages, model, max_retries, label="edit",
            aspect_ratio=aspect_ratio, image_size=image_size,
        )

    def _image_call(self, messages, model, max_retries, label="",
                    aspect_ratio: Optional[str] = None,
                    image_size: Optional[str] = None) -> Optional[bytes]:
        image_config = {}
        if aspect_ratio:
            image_config["aspect_ratio"] = aspect_ratio
        if image_size:
            image_config["image_size"] = image_size
        extra_body = {"modalities": ["image", "text"]}
        if image_config:
            extra_body["image_config"] = image_config

        for attempt in range(max_retries):
            try:
                logger.info(
                    f"Image {label} attempt {attempt + 1}/{max_retries}, "
                    f"model={model}, image_config={image_config or '-'}"
                )
                response = self.client.chat.completions.create(
                    model=model, messages=messages, max_tokens=4096,
                    extra_body=extra_body,
                )
                img_bytes = self._extract_image_from_response(response)
                if img_bytes:
                    logger.info(f"Image {label} ok ({len(img_bytes)} bytes)")
                    return img_bytes
                logger.warning(f"No image in response (attempt {attempt + 1})")
            except Exception as e:
                logger.warning(f"Image {label} attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(5)
        return None

    # ── Quick helpers ─────────────────────────────────────────────────

    def quick_task(
        self,
        messages: list[dict[str, Any]],
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> str:
        """Use the cheapest model for simple classification tasks."""
        return self._chat(
            messages, model=self.config.quick_model,
            temperature=temperature, max_tokens=max_tokens,
        )

    def chat_json(
        self,
        messages: list[dict[str, Any]],
        model: Optional[str] = None,
        temperature: float = 0.3,
    ) -> dict:
        """Send a chat request and parse the JSON response."""
        model = model or self.config.quick_model
        text = self._chat(messages, model=model, temperature=temperature).strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:])
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()
        return json.loads(text)

    # ── Internal ──────────────────────────────────────────────────────

    def _chat(
        self,
        messages: list[dict[str, Any]],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> str:
        """Core chat-completion call."""
        if getattr(self.config, "temperature_zero_mode", False):
            temperature = 0.0
        # Some reasoning-class models reject an explicit `temperature`; omit it.
        no_temp = any(tag in model.lower() for tag in (
            "claude-opus-4.7", "claude-opus-4.8", "claude-opus-5",
            "gpt-5.5", "gpt-5.6", "gpt-6", "o1-", "o3-",
        ))
        kwargs = dict(model=model, messages=messages, max_tokens=max_tokens)
        if not no_temp:
            kwargs["temperature"] = temperature
        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""

    def _extract_image_from_response(self, response) -> Optional[bytes]:
        """Pull image bytes out of a chat-completions response.

        Handles the assistant ``images`` field, ``content`` parts of type
        ``image_url``, and inline base64 data URLs.
        """
        msg = response.choices[0].message

        if hasattr(msg, "images") and msg.images:
            img_data = msg.images[0]
            if isinstance(img_data, dict):
                url = img_data.get("image_url", {}).get("url", "")
            else:
                url = getattr(img_data, "url", "") or ""
            if url.startswith("data:"):
                return base64.b64decode(url.split(",", 1)[1])

        content = msg.content
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    url = part.get("image_url", {}).get("url", "")
                    if url.startswith("data:"):
                        return base64.b64decode(url.split(",", 1)[1])
                elif hasattr(part, "type") and part.type == "image_url":
                    url = getattr(part, "image_url", {})
                    if isinstance(url, dict):
                        url = url.get("url", "")
                    elif hasattr(url, "url"):
                        url = url.url
                    if isinstance(url, str) and url.startswith("data:"):
                        return base64.b64decode(url.split(",", 1)[1])

        text = content if isinstance(content, str) else str(content or "")
        if "base64" in text and "data:image" in text:
            match = re.search(r"data:image/[^;]+;base64,([A-Za-z0-9+/=]+)", text)
            if match:
                return base64.b64decode(match.group(1))
        return None

    @staticmethod
    def _encode_image(path: str) -> str:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    @staticmethod
    def _get_mime(path: str) -> str:
        return {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
        }.get(Path(path).suffix.lower(), "image/png")
