"""Text-size critic — flag <text> elements whose font-size visibly
under- or over-shoots the character height in the original image.

Approach: run Paddle OCR on BOTH the original and the rendered
preview, match texts by content, and compare per-text bbox heights.
If the SVG render's bbox height is < 0.7× or > 1.3× the original's,
emit a `text_size_drift` fix item the refine prompt can act on.

Output items have the same shape as missing_text / text_overflow /
etc. so refine.py picks them up via the existing fix-list channel.
"""
from __future__ import annotations
import os

import logging
import re
import sys
from pathlib import Path
from typing import Optional

REPO = Path(os.environ.get("CRAFTER_HOME", str(Path(__file__).resolve().parents[2])))
sys.path.insert(0, str(REPO))

logger = logging.getLogger("checkers.text_size")


def _norm(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"[^\w%×→←↑↓+\-=/]", "", s)
    return s


def _bbox(entry: dict) -> Optional[tuple[float, float, float, float]]:
    """Normalise paddle entry into (x1, y1, x2, y2) or None."""
    bb = entry.get("bbox")
    if bb and len(bb) >= 4:
        return float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])
    if {"x", "y", "w", "h"} <= set(entry.keys()):
        x, y, w, h = (float(entry[k]) for k in ("x", "y", "w", "h"))
        return x, y, x + w, y + h
    return None


def _paddle_on_preview(preview_png: Path,
                       cache_path: Path | None = None) -> list[dict]:
    """Reuse missing_text's paddle helper logic — same call surface."""
    import json as _json
    if cache_path and cache_path.exists():
        try:
            return _json.loads(cache_path.read_text())
        except Exception:
            pass
    try:
        from PIL import Image
        from crafter.editor.raster_to_svg.agents.paddle_text_extractor import PaddleTextExtractor
        w, h = Image.open(preview_png).size
        ext = PaddleTextExtractor()
        out_dir = Path("/tmp/paddle_text_size_tmp")
        out_dir.mkdir(parents=True, exist_ok=True)
        res = ext.analyze(str(preview_png), (w, h), str(out_dir))
        out = []
        for it in res.texts:
            bb = it.bbox
            out.append({
                "text": it.content.strip(),
                "bbox": [float(bb[0]), float(bb[1]),
                         float(bb[2]), float(bb[3])],
            })
        if cache_path:
            try:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(_json.dumps(out, indent=2))
            except Exception:
                pass
        return out
    except Exception as e:
        logger.warning("paddle_on_preview failed: %s", e)
        return []


def check(
    preview_png: Path,
    paddle_original: list[dict],
    cache_path: Path | None = None,
    min_chars: int = 3,
    drift_low: float = 0.70,
    drift_high: float = 1.40,
    max_items: int = 10,
) -> list[dict]:
    """Return list of text_size_drift fix items.

    For each original-image text we can match in the rendered preview,
    compare height ratios:
      - render_height < drift_low * orig_height  → 'text_too_small'
      - render_height > drift_high * orig_height → 'text_too_large'

    Returns up to `max_items` of the worst drifters, sorted by severity.
    """
    fixes: list[dict] = []
    if not paddle_original:
        return fixes
    try:
        on_preview = _paddle_on_preview(preview_png, cache_path)
    except Exception as e:
        logger.warning("text_size paddle failed: %s", e)
        return fixes
    if not on_preview:
        return fixes

    # Build preview lookup: normalised content → list of bboxes
    preview_by_norm: dict[str, list[tuple]] = {}
    for it in on_preview:
        n = _norm(it.get("text", ""))
        if not n:
            continue
        bb = _bbox(it)
        if bb is None:
            continue
        preview_by_norm.setdefault(n, []).append(bb)

    seen_preview: set[int] = set()  # mark consumed preview entries

    for orig in paddle_original:
        txt = (orig.get("text") or "").strip()
        if len(txt) < min_chars:
            continue
        n = _norm(txt)
        if not n:
            continue
        orig_bb = _bbox(orig)
        if orig_bb is None:
            continue
        orig_h = orig_bb[3] - orig_bb[1]
        if orig_h < 4:  # too small to measure reliably
            continue

        # Fuzzy match: exact, contains, or contained
        cand_bboxes: list[tuple[float, float, float, float]] = []
        for pn, bbs in preview_by_norm.items():
            if (n == pn or n in pn or pn in n) and pn:
                # only consume each preview entry once
                for j, bb in enumerate(bbs):
                    key = id(bb)
                    if key in seen_preview:
                        continue
                    cand_bboxes.append(bb)
                    seen_preview.add(key)
                    break
                if cand_bboxes:
                    break
        if not cand_bboxes:
            continue

        rend_bb = cand_bboxes[0]
        rend_h = rend_bb[3] - rend_bb[1]
        if rend_h < 1:
            continue

        ratio = rend_h / orig_h
        if drift_low <= ratio <= drift_high:
            continue

        kind = "text_too_small" if ratio < drift_low else "text_too_large"
        # Severity: how far the ratio is from 1.0
        if ratio < 0.5 or ratio > 2.0:
            sev = "high"
        elif ratio < 0.6 or ratio > 1.6:
            sev = "medium"
        else:
            sev = "low"
        bb_round = [round(v) for v in orig_bb]
        rend_round = [round(v) for v in rend_bb]
        if kind == "text_too_small":
            advice = (f"text '{txt[:50]}' renders TOO SMALL — at original "
                      f"bbox {bb_round} the char height is ~{orig_h:.0f}px "
                      f"but the SVG renders it at only ~{rend_h:.0f}px "
                      f"(ratio {ratio:.2f}). INCREASE its font-size to fit "
                      f"the original character height (target font-size ≈ "
                      f"{orig_h * 0.85:.0f}px).")
        else:
            advice = (f"text '{txt[:50]}' renders TOO LARGE — at original "
                      f"bbox {bb_round} the char height is ~{orig_h:.0f}px "
                      f"but the SVG renders it at ~{rend_h:.0f}px "
                      f"(ratio {ratio:.2f}). DECREASE its font-size to "
                      f"match the original (target font-size ≈ "
                      f"{orig_h * 0.85:.0f}px).")
        fixes.append({
            "kind": kind,
            "text": txt[:50],
            "expected_bbox": bb_round,
            "render_bbox": rend_round,
            "ratio": round(ratio, 2),
            "severity": sev,
            "fix": advice,
        })

    fixes.sort(key=lambda f: ({"high": 0, "medium": 1, "low": 2}[f["severity"]],
                              abs(1.0 - f["ratio"]) * -1))  # most-deviated first
    return fixes[:max_items]
