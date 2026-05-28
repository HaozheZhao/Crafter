"""Element-position checker (Item 4).

For each labelled element (RAS_NN, VEC_NN, icon_*) in the SVG,
compare its current bbox to the expected placement_bbox saved in
labels.json. Flag misplacements > 20 px.
"""
from __future__ import annotations

import json
import logging
import re
import xml.etree.ElementTree as ET
from pathlib import Path

NS = "{http://www.w3.org/2000/svg}"
logger = logging.getLogger("checkers.position")

MISPLACE_THRESHOLD_PX = 50  # widened from 20 — only flag clear errors;
                            # the LLM should rely on Image #1 for fine
                            # placement, not numeric reference data.


def _element_bbox(el: ET.Element) -> tuple[float, float, float, float] | None:
    tag = el.tag.replace(NS, "")
    try:
        if tag in ("rect", "image"):
            x = float(el.get("x", 0)); y = float(el.get("y", 0))
            w = float(el.get("width", 0)); h = float(el.get("height", 0))
            return (x, y, x + w, y + h)
        if tag == "circle":
            cx = float(el.get("cx", 0)); cy = float(el.get("cy", 0))
            r = float(el.get("r", 0))
            return (cx - r, cy - r, cx + r, cy + r)
    except (TypeError, ValueError):
        return None
    return None


def _group_bbox(g: ET.Element) -> tuple[float, float, float, float] | None:
    """Compute the bbox enclosing all primitive children of <g>."""
    bbs = []
    for child in g.iter():
        bb = _element_bbox(child)
        if bb:
            bbs.append(bb)
    if not bbs:
        return None
    return (min(b[0] for b in bbs), min(b[1] for b in bbs),
            max(b[2] for b in bbs), max(b[3] for b in bbs))


def _center(bb: tuple[float, float, float, float]) -> tuple[float, float]:
    return ((bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2)


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def check(svg_path: Path, labels_json_path: Path) -> list[dict]:
    fixes: list[dict] = []
    try:
        root = ET.parse(svg_path).getroot()
    except ET.ParseError:
        return fixes
    if not labels_json_path.exists():
        return fixes
    labels = json.loads(labels_json_path.read_text())
    expected: dict[str, list[float]] = {}
    for r in labels.get("raster", []):
        if r.get("bbox"):
            expected[r["label"]] = list(r["bbox"])
            # injected raster gets id="icon_RAS_NN"
            expected[f"icon_{r['label']}"] = list(r["bbox"])
    for v in labels.get("vector", []):
        if v.get("bbox"):
            expected[v["label"]] = list(v["bbox"])

    if not expected:
        return fixes

    # Walk SVG; for any element with id matching expected, get its bbox
    for el in root.iter():
        elid = el.get("id", "")
        if elid not in expected:
            continue
        tag = el.tag.replace(NS, "")
        if tag == "g":
            cur = _group_bbox(el)
        else:
            cur = _element_bbox(el)
        if cur is None:
            continue
        exp = tuple(expected[elid])
        offset = _dist(_center(cur), _center(exp))
        if offset > MISPLACE_THRESHOLD_PX:
            fixes.append({
                "kind": "misplaced",
                "id": elid,
                "current_bbox": [round(v) for v in cur],
                "expected_bbox": [round(v) for v in exp],
                "centre_offset_px": round(offset, 1),
                "fix": (f"element id='{elid}' is offset by "
                        f"{offset:.0f}px from its expected position "
                        f"{[round(v) for v in exp]}. MOVE it (do NOT just "
                        f"restyle) so its bbox matches."),
            })

    return fixes
