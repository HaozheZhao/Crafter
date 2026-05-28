"""Image encoding for the VLM judge.

Scientific figures carry dense small text, so resolution is kept fairly high
(default max side 1280). RGBA is flattened onto white to match how figures are
viewed on a page.
"""
from __future__ import annotations
import base64, io
from functools import lru_cache
from PIL import Image


def _flatten(img: Image.Image) -> Image.Image:
    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[-1])
        return bg
    return img if img.mode == "RGB" else img.convert("RGB")


@lru_cache(maxsize=4096)
def encode_data_url(path: str, max_side: int = 1280, quality: int = 90) -> str:
    """Return a JPEG ``data:`` URL for ``path``, resized so max(w, h) <= max_side."""
    img = _flatten(Image.open(path))
    if max(img.size) > max_side:
        s = max_side / max(img.size)
        img = img.resize((max(1, int(img.width * s)), max(1, int(img.height * s))),
                         Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/jpeg;base64,{b64}"


def img_part(path: str, **kw) -> dict:
    return {"type": "image_url", "image_url": {"url": encode_data_url(str(path), **kw)}}
