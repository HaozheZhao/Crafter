"""Text overflow / collision checker for the refine loop.

Given the rendered preview PNG of the current SVG, run Paddle OCR to
get actually-rendered text bboxes. Compare against:
  (a) the SVG <text> declared bbox (font-size × len heuristic)
  (b) other text bboxes — detect overlaps

Output: list of fix items the refine prompt can consume.
"""
from __future__ import annotations
import os

import logging
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

REPO = Path(os.environ.get("CRAFTER_HOME", str(Path(__file__).resolve().parents[2])))
sys.path.insert(0, str(REPO))

NS = "{http://www.w3.org/2000/svg}"
logger = logging.getLogger("checkers.text_overflow")


def _svg_text_bboxes(svg_path: Path) -> list[dict]:
    """Pull declared <text> bboxes from the SVG (heuristic by font-size)."""
    try:
        root = ET.parse(svg_path).getroot()
    except ET.ParseError:
        return []
    out = []
    for el in root.iter(NS + "text"):
        txt = "".join(el.itertext()).strip()
        if not txt:
            continue
        try:
            x = float(el.get("x", 0))
            y = float(el.get("y", 0))
        except (TypeError, ValueError):
            continue
        try:
            fs = float(el.get("font-size", 12))
        except (TypeError, ValueError):
            fs = 12
        anchor = el.get("text-anchor", "start")
        approx_w = fs * 0.55 * len(txt)
        approx_h = fs * 1.2
        if anchor == "middle":
            x1 = x - approx_w / 2
        elif anchor == "end":
            x1 = x - approx_w
        else:
            x1 = x
        y1 = y - approx_h * 0.85   # SVG y is baseline
        out.append({
            "text": txt, "x": x, "y": y,
            "declared_bbox": [x1, y1, x1 + approx_w, y1 + approx_h],
            "font_size": fs,
        })
    return out


def _paddle_on_preview(preview_png: Path) -> list[dict]:
    from PIL import Image
    from crafter.editor.raster_to_svg.agents.paddle_text_extractor import PaddleTextExtractor
    w, h = Image.open(preview_png).size
    ext = PaddleTextExtractor()
    out_dir = Path("/tmp/paddle_overflow_tmp")
    out_dir.mkdir(parents=True, exist_ok=True)
    res = ext.analyze(str(preview_png), (w, h), str(out_dir))
    out = []
    for it in res.texts:
        bb = it.bbox
        out.append({
            "text": it.content.strip(),
            "bbox": [float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])],
        })
    return out


def _bbox_overflow(decl: list[float], rendered: list[float],
                   margin: int = 5) -> bool:
    """Return True if rendered extends beyond declared by > margin px."""
    return (rendered[2] > decl[2] + margin or
            rendered[3] > decl[3] + margin or
            rendered[0] < decl[0] - margin or
            rendered[1] < decl[1] - margin)


def _bbox_overlap(a: list[float], b: list[float]) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0] or
                a[3] <= b[1] or b[3] <= a[1])


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", s.lower())


def check(svg_path: Path, preview_png: Path) -> list[dict]:
    """Return list of fix items (each: kind, text, fix)."""
    fixes = []
    declared = _svg_text_bboxes(svg_path)
    try:
        rendered = _paddle_on_preview(preview_png)
    except Exception as e:
        logger.warning("  paddle on preview failed: %s", e)
        return fixes

    # Match declared <text> with rendered Paddle bbox by content
    used = set()
    for d in declared:
        nd = _norm(d["text"])
        match = None
        for i, r in enumerate(rendered):
            if i in used:
                continue
            nr = _norm(r["text"])
            if nd == nr or nd in nr or nr in nd:
                match = (i, r)
                break
        if match is None:
            continue
        used.add(match[0])
        if _bbox_overflow(d["declared_bbox"], match[1]["bbox"]):
            fixes.append({
                "kind": "text_overflow",
                "text": d["text"],
                "current_font_size": d["font_size"],
                "declared_bbox": [round(v) for v in d["declared_bbox"]],
                "rendered_bbox": [round(v) for v in match[1]["bbox"]],
                "fix": (f"text '{d['text']}' overflows: rendered bbox extends "
                        f"beyond declared. Reduce font-size from "
                        f"{d['font_size']:.0f} to fit, OR break into multi-line "
                        f"with <tspan dy='1em'>"),
            })

    # Pairwise overlap check on rendered bboxes
    for i, a in enumerate(rendered):
        for j, b in enumerate(rendered[i + 1:], start=i + 1):
            if _bbox_overlap(a["bbox"], b["bbox"]):
                fixes.append({
                    "kind": "text_collision",
                    "texts": [a["text"], b["text"]],
                    "bboxes": [[round(v) for v in a["bbox"]],
                               [round(v) for v in b["bbox"]]],
                    "fix": (f"text '{a['text']}' and '{b['text']}' overlap "
                            f"in the rendered preview — move one of them or "
                            f"shrink both font-sizes"),
                })

    return fixes
