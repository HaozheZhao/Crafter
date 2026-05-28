"""Missing-text checker — diff Paddle OCR results between original and
preview. Anything that Paddle reads on the original but not on the
preview (after fuzzy normalisation) is flagged as missing.

Reuses the cached paddle_texts on the original (from paddle_text.py)
and runs a fresh Paddle on the preview each call. Caches the preview
result alongside the iter's preview PNG to avoid re-running Paddle.
"""
from __future__ import annotations
import os

import json
import logging
import re
import sys
from pathlib import Path

REPO = Path(os.environ.get("CRAFTER_HOME", str(Path(__file__).resolve().parents[2])))
sys.path.insert(0, str(REPO))

logger = logging.getLogger("checkers.missing_text")


def _norm(s: str) -> str:
    """Normalise for fuzzy compare: lower, strip whitespace and most
    punctuation, keep alphanumerics + a few symbols that carry meaning."""
    s = (s or "").lower()
    s = re.sub(r"[^\w%×→←↑↓+\-=/]", "", s)
    return s


def _paddle_on_preview(preview_png: Path,
                       cache_path: Path | None = None) -> list[dict]:
    if cache_path and cache_path.exists():
        try:
            return json.loads(cache_path.read_text())
        except Exception:
            pass
    from PIL import Image
    from crafter.editor.raster_to_svg.agents.paddle_text_extractor import PaddleTextExtractor
    w, h = Image.open(preview_png).size
    ext = PaddleTextExtractor()
    out_dir = Path("/tmp/paddle_missing_tmp")
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
            cache_path.write_text(json.dumps(out, indent=2))
        except Exception:
            pass
    return out


def _bbox_of_paddle_entry(t: dict) -> list[float]:
    """Paddle entries can be either {bbox:[x1,y1,x2,y2], ...} (new) or
    {x, y, w, h, ...} (harness cache format). Normalise."""
    if t.get("bbox"):
        bb = t["bbox"]
        return [float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])]
    if {"x", "y", "w", "h"} <= set(t.keys()):
        x, y, w, h = float(t["x"]), float(t["y"]), float(t["w"]), float(t["h"])
        return [x, y, x + w, y + h]
    return [0.0, 0.0, 0.0, 0.0]


def check(
    preview_png: Path,
    paddle_original: list[dict],
    cache_path: Path | None = None,
    min_chars: int = 2,
) -> list[dict]:
    """Return list of missing-text fix items.

    Each fix item:
      kind="missing_text", text=str, expected_bbox=[..],
      severity= 'high'|'medium', fix=imperative sentence
    """
    fixes: list[dict] = []
    if not paddle_original:
        return fixes
    try:
        on_preview = _paddle_on_preview(preview_png, cache_path)
    except Exception as e:
        logger.warning("missing_text paddle on preview failed: %s", e)
        return fixes

    preview_set = {_norm(t["text"]) for t in on_preview if t["text"]}

    for orig in paddle_original:
        txt = (orig.get("text") or "").strip()
        if len(txt) < min_chars:
            continue
        n = _norm(txt)
        if not n:
            continue
        # Fuzzy match: try exact, contains in either direction
        found = False
        for p in preview_set:
            if n == p or n in p or p in n:
                found = True
                break
        if found:
            continue

        bb = _bbox_of_paddle_entry(orig)
        # Severity: longer text is harder to miss → higher severity
        sev = "high" if len(txt) >= 6 else "medium"
        bb_round = [round(float(v)) for v in bb]
        fixes.append({
            "kind": "missing_text",
            "text": txt[:60],
            "expected_bbox": bb_round,
            "severity": sev,
            "fix": (f"text '{txt[:60]}' from the original is missing in "
                    f"the preview. Add a <text> with this exact content "
                    f"NEAR original-image bbox {bb_round} (this is a "
                    f"visual anchor — verify exact position by looking at "
                    f"Image #1). Choose a font-size matching the bbox "
                    f"height."),
        })

    # Sort: high severity first, then by length
    fixes.sort(key=lambda f: (f["severity"] != "high", -len(f["text"])))
    return fixes[:12]
