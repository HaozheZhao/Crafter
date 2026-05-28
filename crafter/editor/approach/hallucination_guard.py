"""Step A — batch hallucination guard.

After SAM3 produces a list of (icon_id, kind, desc, bbox, crop_path),
this module renders the original figure with NUMBERED bboxes overlaid
and asks gpt-5.5 to verify each numbered region in one batch call:

  • Does the region in the ORIGINAL actually depict an icon matching
    the description?
  • If not, mark it `hallucinated` so downstream Stage B drops it.

Output: per-id verdict {valid: bool, confidence: float, reason: str}.
Cached.

Why batch (not per-icon):
  • 1 VLM call ~30s vs N calls × 10s each (N typically 15-30).
  • Saves 5-10× wall-clock per image.
  • Slight downside: if the model makes one mistake it affects N items,
    but with explicit per-number verdict format the failure is local.

Design notes for the prompt:
  • Render bboxes with thick lime numbered labels so they are
    unambiguously locatable.
  • Demand strict JSON keyed by integer id.
  • Tell the model to be CONSERVATIVE — only flag hallucinated when
    confident. False positives hurt more than false negatives here.
"""
from __future__ import annotations
import os

import base64
import io
import json
import logging
import re
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REPO = Path(os.environ.get("CRAFTER_HOME", str(Path(__file__).resolve().parents[2])))

from crafter.editor.approach.iter_svg_fix import call_model  # noqa: E402

logger = logging.getLogger("hallucination_guard")


GUARD_PROMPT = """You are auditing automated icon detections on an
academic figure. The original is shown with NUMBERED green bboxes
overlaid. Each number corresponds to one detected region with a
description.

For EACH numbered bbox, decide:
  • valid       — the region in the ORIGINAL actually contains an icon
                   matching (or close to) the description.
  • invalid     — the region is empty, just background, contains pure
                   text, or shows something clearly unrelated to the
                   description (a "hallucinated" detection).

Be CONSERVATIVE: when in doubt, return valid=true. Only mark invalid
when you can clearly see that the bbox does NOT contain any icon
resembling the description (e.g. it's blank background, or just plain
text labels).

DESCRIPTIONS PER ID:
{DESCRIPTIONS}

Return STRICT JSON, exactly this shape, with one entry per id above:
{
  "verdicts": {
    "1": {"valid": true,  "confidence": 0.9, "reason": "matches"},
    "2": {"valid": false, "confidence": 0.85, "reason": "blank background"},
    "3": {"valid": true,  "confidence": 0.7, "reason": "small but visible icon"},
    ...
  }
}

GUARDRAILS:
  • Do NOT add extra keys.
  • Do NOT skip any numbered id (even if uncertain — return valid=true).
  • confidence is a float in [0, 1].
  • reason is at most 12 words.
"""


def _b64(img: Image.Image, max_dim: int = 1500) -> str:
    img = img.convert("RGB")
    if max(img.size) > max_dim:
        scale = max_dim / max(img.size)
        img = img.resize((int(img.width * scale), int(img.height * scale)),
                         Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return base64.b64encode(buf.getvalue()).decode()


def _font(sz: int) -> ImageFont.ImageFont:
    for p in [
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]:
        if Path(p).exists():
            return ImageFont.truetype(p, sz)
    return ImageFont.load_default()


def _draw_numbered(original: Path, items: list[dict],
                   out_path: Path) -> Image.Image:
    """Overlay each item's bbox + a numeric label. Saves to out_path
    and returns the PIL image."""
    img = Image.open(original).convert("RGB").copy()
    draw = ImageDraw.Draw(img, "RGBA")
    f = _font(22)
    for it in items:
        bb = it["bbox"]
        n = it["num"]
        x1, y1, x2, y2 = [int(round(v)) for v in bb]
        # Outer rect
        for w in range(4):
            draw.rectangle([x1 - w, y1 - w, x2 + w, y2 + w],
                           outline=(50, 220, 50, 255))
        # Numbered tag
        label = f"{n}"
        tb = draw.textbbox((0, 0), label, font=f)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        pad = 4
        tx, ty = x1, max(0, y1 - th - 2 * pad)
        draw.rectangle([tx, ty, tx + tw + 2 * pad, ty + th + 2 * pad],
                       fill=(20, 130, 20, 240))
        draw.text((tx + pad, ty + pad), label, fill=(255, 255, 255), font=f)
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(out_path)
    return img


def _format_descriptions(items: list[dict]) -> str:
    lines = []
    for it in items:
        n = it["num"]
        kind = it.get("kind", "?")
        desc = (it.get("desc") or it.get("simple_desc") or "")[:80]
        lines.append(f"  {n}. ({kind}) {desc}")
    return "\n".join(lines)


def _extract_json(text: str) -> dict | None:
    text = text.strip()
    m = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if m:
        text = m.group(1).strip()
    for s in range(len(text)):
        if text[s] != "{":
            continue
        depth = 0
        for e in range(s, len(text)):
            if text[e] == "{":
                depth += 1
            elif text[e] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[s:e + 1])
                    except json.JSONDecodeError:
                        break
    return None


def verify(
    original_png: Path,
    items: list[dict],
    out_dir: Path | None = None,
    cache_path: Path | None = None,
    overlay_path: Path | None = None,
    drop_threshold: float = 0.7,
    model: str = "openai/gpt-5.5",
) -> dict:
    """Verify each item is a real icon in the original.

    items: list of dicts, each must have:
      'id'    — caller's stable id
      'bbox'  — [x1,y1,x2,y2] in original-image coords
      'kind'  — short kind label
      'desc'  — natural-language description
    Items with bbox=None are skipped.

    Returns a dict:
      {
        "verdicts": {<caller_id>: {valid, confidence, reason}},
        "dropped": [<caller_id>, ...],
        "kept": [<caller_id>, ...],
        "overlay_path": str
      }
    """
    if cache_path and cache_path.exists():
        try:
            return json.loads(cache_path.read_text())
        except Exception:
            pass

    # Number the items 1..N for the prompt; map back at the end.
    numbered = []
    for i, it in enumerate(items, start=1):
        if not it.get("bbox"):
            continue
        numbered.append({**it, "num": i})

    if not numbered:
        return {"verdicts": {}, "dropped": [], "kept": [],
                "overlay_path": ""}

    overlay_dst = overlay_path or (out_dir / "halluc_numbered.png"
                                   if out_dir else None)
    img = _draw_numbered(original_png, numbered, overlay_dst)

    prompt = GUARD_PROMPT.replace("{DESCRIPTIONS}",
                                  _format_descriptions(numbered))
    msgs = [{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url":
                {"url": f"data:image/jpeg;base64,{_b64(img)}"}},
            {"type": "text", "text": prompt},
        ],
    }]
    try:
        resp = call_model(msgs, model=model, max_tokens=4000,
                          label="stage5_hallucination_guard")
    except Exception as e:
        logger.warning("hallucination_guard api failed: %s — keep all", e)
        verdicts = {it["id"]: {"valid": True, "confidence": 0.0,
                              "reason": "api error, keep"}
                    for it in numbered}
        return {"verdicts": verdicts, "dropped": [],
                "kept": [it["id"] for it in numbered],
                "overlay_path": str(overlay_dst) if overlay_dst else ""}

    parsed = _extract_json(resp) or {}
    raw = parsed.get("verdicts", {}) or {}

    verdicts: dict[str, dict] = {}
    for it in numbered:
        v = raw.get(str(it["num"])) or raw.get(it["num"]) or {}
        verdicts[it["id"]] = {
            "valid": bool(v.get("valid", True)),
            "confidence": float(v.get("confidence", 0.0)),
            "reason": str(v.get("reason", ""))[:80],
        }

    dropped = [iid for iid, v in verdicts.items()
               if (not v["valid"]) and v["confidence"] >= drop_threshold]
    kept = [iid for iid in verdicts if iid not in dropped]

    out = {"verdicts": verdicts,
           "dropped": dropped,
           "kept": kept,
           "overlay_path": str(overlay_dst) if overlay_dst else ""}

    if cache_path:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(out, indent=2))
        except Exception:
            pass

    return out
