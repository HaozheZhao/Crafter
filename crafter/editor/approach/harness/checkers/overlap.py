"""Box-overlap checker — detects rect / image / g elements that
partially overlap each other (IoU > threshold) and are NOT in a
legitimate nested relationship (one fully contained inside the other).

Pure programmatic; no API calls.

Heuristics:
  * Only consider rect / image / g elements with positive area.
  * Skip elements < 20px on either side (decorative).
  * Skip if one bbox fully contains the other (panel-child relation).
  * Skip if IoU < 0.10 (incidental near-touch).
  * Skip pairs that share the same labels.json placement_bbox (the
    harness placed them there on purpose).

Output: list of fix items, one per offending pair, asking the refine
LLM to move the smaller element off the bigger one.
"""
from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from pathlib import Path
import json

NS = "{http://www.w3.org/2000/svg}"
logger = logging.getLogger("checkers.overlap")

IOU_THRESHOLD = 0.10
MIN_DIM_PX = 20


def _bbox_of(el: ET.Element) -> tuple[float, float, float, float] | None:
    tag = el.tag.replace(NS, "")
    try:
        if tag in ("rect", "image"):
            x = float(el.get("x", 0)); y = float(el.get("y", 0))
            w = float(el.get("width", 0)); h = float(el.get("height", 0))
            if w <= 0 or h <= 0:
                return None
            return (x, y, x + w, y + h)
        if tag == "circle":
            cx = float(el.get("cx", 0)); cy = float(el.get("cy", 0))
            r = float(el.get("r", 0))
            if r <= 0:
                return None
            return (cx - r, cy - r, cx + r, cy + r)
    except (TypeError, ValueError):
        return None
    return None


def _group_bbox(g: ET.Element) -> tuple[float, float, float, float] | None:
    bbs = []
    for child in g.iter():
        bb = _bbox_of(child)
        if bb:
            bbs.append(bb)
    if not bbs:
        return None
    return (min(b[0] for b in bbs), min(b[1] for b in bbs),
            max(b[2] for b in bbs), max(b[3] for b in bbs))


def _area(bb: tuple[float, float, float, float]) -> float:
    return max(0, bb[2] - bb[0]) * max(0, bb[3] - bb[1])


def _iou(a: tuple, b: tuple) -> float:
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    union = _area(a) + _area(b) - inter
    return inter / union if union > 0 else 0.0


def _contains(outer: tuple, inner: tuple, slack: float = 4.0) -> bool:
    """True if outer fully contains inner (with slack px tolerance)."""
    return (outer[0] <= inner[0] + slack and
            outer[1] <= inner[1] + slack and
            outer[2] >= inner[2] - slack and
            outer[3] >= inner[3] - slack)


def _is_decorative(bb: tuple) -> bool:
    return (bb[2] - bb[0]) < MIN_DIM_PX or (bb[3] - bb[1]) < MIN_DIM_PX


def _short(el: ET.Element) -> str:
    tag = el.tag.replace(NS, "")
    elid = el.get("id", "")
    cls = el.get("class", "")
    if elid:
        return f"<{tag} id='{elid}'>"
    if cls:
        return f"<{tag} class='{cls}'>"
    return f"<{tag}>"


def check(svg_path: Path, labels_json: Path | None = None) -> list[dict]:
    fixes: list[dict] = []
    try:
        root = ET.parse(svg_path).getroot()
    except ET.ParseError:
        return fixes

    # Collect candidate elements.
    items = []
    for el in root.iter():
        tag = el.tag.replace(NS, "")
        if tag == "g":
            bb = _group_bbox(el)
            # only consider <g> with id (likely a labelled group)
            if bb and el.get("id"):
                items.append((el, bb, tag))
        elif tag in ("rect", "image"):
            bb = _bbox_of(el)
            if bb:
                items.append((el, bb, tag))

    # Skip decorative tiny shapes
    items = [(e, b, t) for e, b, t in items if not _is_decorative(b)]

    # Pairwise overlap
    for i in range(len(items)):
        ea, ba, ta = items[i]
        for j in range(i + 1, len(items)):
            eb, bb, tb = items[j]
            if _contains(ba, bb) or _contains(bb, ba):
                continue  # legitimate nesting
            iou = _iou(ba, bb)
            if iou < IOU_THRESHOLD:
                continue
            # Decide which to move (smaller area)
            if _area(ba) <= _area(bb):
                small_e, small_bb = ea, ba
                big_e, big_bb = eb, bb
            else:
                small_e, small_bb = eb, bb
                big_e, big_bb = ea, ba
            fixes.append({
                "kind": "overlap",
                "iou": round(iou, 2),
                "small": _short(small_e),
                "small_bbox": [round(v) for v in small_bb],
                "big": _short(big_e),
                "big_bbox": [round(v) for v in big_bb],
                "fix": (f"{_short(small_e)} overlaps {_short(big_e)} "
                        f"with IoU {iou:.2f}. Move {_short(small_e)} "
                        f"to nearby empty space (offset 30+ px) so the "
                        f"two no longer cover each other."),
            })

    # Cap to top-N by IoU (worst first)
    fixes.sort(key=lambda f: -f["iou"])
    return fixes[:8]
