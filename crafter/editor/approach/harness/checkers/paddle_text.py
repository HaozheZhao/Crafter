"""Paddle OCR — extract authoritative text list for skeleton prompt.

Single function: given an image path, return list of dicts:
  {text, x, y, w, h, font_size_estimate, color_hex, weight}

Used at skeleton time to seed the LLM with ground-truth text positions
+ content. The LLM still emits <text> SVG elements but anchored to
these bboxes (and may improve unicode rendering for subscripts/etc.).

Cached to disk so we don't pay the OCR cost on every harness re-run.
"""
from __future__ import annotations
import os

import json
import logging
import sys
from pathlib import Path

REPO = Path(os.environ.get("CRAFTER_HOME", str(Path(__file__).resolve().parents[2])))
sys.path.insert(0, str(REPO))

logger = logging.getLogger("checkers.paddle_text")


def extract(image_path: Path, cache_path: Path | None = None,
            min_score: float = 0.5) -> list[dict]:
    """Run PaddleOCR; return cleaned list. Optionally cache to JSON.

    Graceful fallback: if PaddleOCR isn't installed or fails to load,
    returns an empty list and caches it so the pipeline can proceed
    on LLM-only text reading. Downstream PADDLE_INJECT_THRESHOLD logic
    already handles empty/short paddle lists.
    """
    if cache_path and cache_path.exists():
        logger.info("  using cached Paddle text: %s", cache_path)
        return json.loads(cache_path.read_text())

    try:
        from PIL import Image
        from crafter.editor.raster_to_svg.agents.paddle_text_extractor import PaddleTextExtractor
    except ImportError as e:
        logger.warning("  PaddleOCR unavailable (%s) — skipping OCR. "
                       "Pipeline will use LLM-only text reading.", e)
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text("[]")
        return []
    try:
        w, h = Image.open(image_path).size
        ext = PaddleTextExtractor()
        out_dir = Path("/tmp/paddle_skeleton_tmp")
        out_dir.mkdir(parents=True, exist_ok=True)
        res = ext.analyze(str(image_path), (w, h), str(out_dir))
    except Exception as e:
        logger.warning("  PaddleOCR runtime error (%s) — skipping OCR. "
                       "Pipeline will use LLM-only text reading.", e)
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text("[]")
        return []

    items = []
    for it in res.texts:
        bb = it.bbox
        x1, y1, x2, y2 = float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])
        items.append({
            "text": it.content.strip(),
            "x": x1, "y": y1,
            "w": x2 - x1, "h": y2 - y1,
            "font_size_pt": getattr(it, "font_size_pt", None),
            "color_hex": getattr(it, "color_hex", "#000000"),
            "weight": getattr(it, "weight", "normal"),
            "italic": getattr(it, "italic", False),
            "rotation_deg": getattr(it, "rotation_deg", 0),
        })
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(items, indent=2))
        logger.info("  cached Paddle text → %s (%d entries)",
                    cache_path, len(items))
    return items


def format_for_prompt(items: list[dict]) -> str:
    """Render the Paddle list for SKELETON_PROMPT.

    If an item carries `style_hint` / `font_family` / `category` from
    text_recovery, surface them so the LLM picks the right unicode +
    font for the SVG.
    """
    lines = []
    for it in items:
        x = round(it["x"]); y = round(it["y"])
        w = round(it["w"]); h = round(it["h"])
        fs = it.get("font_size_pt") or max(8, round(h * 0.85))
        wt = it.get("weight", "normal")
        clr = it.get("color_hex", "#000000")
        rot = it.get("rotation_deg", 0)
        rot_s = f" rotation={rot}deg" if rot else ""
        # text_recovery hints (optional)
        hint_s = ""
        if it.get("style_hint") and it["style_hint"] != "normal":
            hint_s += f" style={it['style_hint']}"
        if it.get("font_family"):
            hint_s += f" font_family='{it['font_family']}'"
        if it.get("category") and it["category"] not in ("ascii", "other"):
            hint_s += f" cat={it['category']}"
        if it.get("original_paddle_text") and \
                it["original_paddle_text"] != it["text"]:
            hint_s += f" (recovered_from='{it['original_paddle_text']}')"
        lines.append(
            f'  text="{it["text"]}" bbox=({x},{y},{x+w},{y+h}) '
            f'font_size={fs} weight={wt} fill={clr}{rot_s}{hint_s}'
        )
    return "\n".join(lines)
