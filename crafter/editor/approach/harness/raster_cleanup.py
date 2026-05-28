"""raster_cleanup — remove vector primitives that overlap <image> bboxes.

This is a defensive cleanup: even when the prompts forbid drawing
vectors over raster images, LLMs sometimes do anyway. We scan the SVG
after inject_raster + after each accepted refine iter, and delete
overlapping vector primitives (rect / path / circle / ellipse / line
/ polyline / polygon) that would cause "double-drawing" or mid-layer
backgrounds beneath the raster icon.

Rules applied:
  • The full-canvas background <rect> at (0,0,W,H) is preserved.
  • A vector with bbox overlapping an <image> bbox by IoU ≥ 0.5 is
    removed. Smaller overlaps (e.g. nearby labels) are left alone.
  • Stripped vectors are reported (count + ids) for logging.
"""
from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from pathlib import Path

NS = "{http://www.w3.org/2000/svg}"
logger = logging.getLogger("harness.raster_cleanup")

# Same overlap math as checkers/overlap.py
def _bbox_of_rect(el):
    try:
        x = float(el.get("x", 0)); y = float(el.get("y", 0))
        w = float(el.get("width", 0)); h = float(el.get("height", 0))
        if w <= 0 or h <= 0:
            return None
        return (x, y, x + w, y + h)
    except (TypeError, ValueError):
        return None


def _bbox_of_circle(el):
    try:
        cx = float(el.get("cx", 0)); cy = float(el.get("cy", 0))
        r = float(el.get("r", 0))
        if r <= 0:
            return None
        return (cx - r, cy - r, cx + r, cy + r)
    except (TypeError, ValueError):
        return None


def _bbox_of_ellipse(el):
    try:
        cx = float(el.get("cx", 0)); cy = float(el.get("cy", 0))
        rx = float(el.get("rx", 0)); ry = float(el.get("ry", 0))
        if rx <= 0 or ry <= 0:
            return None
        return (cx - rx, cy - ry, cx + rx, cy + ry)
    except (TypeError, ValueError):
        return None


def _bbox_of_line(el):
    try:
        x1 = float(el.get("x1", 0)); y1 = float(el.get("y1", 0))
        x2 = float(el.get("x2", 0)); y2 = float(el.get("y2", 0))
        return (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
    except (TypeError, ValueError):
        return None


def _parse_path_d(d):
    """Approximate bbox from path d attribute by collecting numeric
    coordinates. Crude but works for most LLM-generated paths."""
    if not d:
        return None
    nums = re.findall(r"-?\d+(?:\.\d+)?", d)
    if len(nums) < 4:
        return None
    nums = [float(n) for n in nums]
    xs = nums[0::2]
    ys = nums[1::2]
    if not xs or not ys:
        return None
    return (min(xs), min(ys), max(xs), max(ys))


def _parse_polypoints(s):
    if not s:
        return None
    nums = re.findall(r"-?\d+(?:\.\d+)?", s)
    if len(nums) < 4:
        return None
    nums = [float(n) for n in nums]
    xs = nums[0::2]; ys = nums[1::2]
    return (min(xs), min(ys), max(xs), max(ys))


def _bbox_of(el):
    tag = el.tag.replace(NS, "")
    if tag == "rect":
        return _bbox_of_rect(el)
    if tag == "image":
        return _bbox_of_rect(el)
    if tag == "circle":
        return _bbox_of_circle(el)
    if tag == "ellipse":
        return _bbox_of_ellipse(el)
    if tag == "line":
        return _bbox_of_line(el)
    if tag == "path":
        return _parse_path_d(el.get("d", ""))
    if tag in ("polyline", "polygon"):
        return _parse_polypoints(el.get("points", ""))
    return None


def _iou(a, b):
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    aw, ah = a[2] - a[0], a[3] - a[1]
    bw, bh = b[2] - b[0], b[3] - b[1]
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _bbox_inside(inner, outer, slack=2.0):
    return (outer[0] <= inner[0] + slack and
            outer[1] <= inner[1] + slack and
            outer[2] >= inner[2] - slack and
            outer[3] >= inner[3] - slack)


def _is_canvas_background(el, w, h, slack=8):
    """Skip the full-canvas background rect (it's legitimately under
    everything, including raster images)."""
    bb = _bbox_of(el)
    if not bb:
        return False
    return (bb[0] <= slack and bb[1] <= slack and
            bb[2] >= w - slack and bb[3] >= h - slack)


CLEANUP_TAGS = {"rect", "circle", "ellipse", "line", "path",
                "polyline", "polygon"}


def cleanup_overlapping_vectors(
    svg_text: str,
    canvas_w: float,
    canvas_h: float,
    iou_threshold: float = 0.5,
    contains_threshold: float = 1.01,  # disabled by default
) -> tuple[str, dict]:
    """Remove vector primitives whose bbox overlaps any <image> bbox by
    IoU ≥ iou_threshold (= "near-identical bbox" — that's the
    background/halo case, safe to delete) OR fully contained with
    contains_frac ≥ contains_threshold.

    DEFAULT BEHAVIOUR (contains_threshold=1.01 = effectively disabled):
      ONLY remove vectors whose bbox is near-identical to an image
      bbox. This is the "灰底" / "redundant background rect" case.
      Small decorative vectors INSIDE a large image bbox (e.g. graph
      nodes drawn over a layered panel image, or histogram bars over
      a chart image) are PRESERVED — those are legitimate overlays
      the original figure intended.

    To be more aggressive (delete decorations too), pass a lower
    contains_threshold like 0.7. Empirically that destroys img5/img10
    style figures.

    Preserves: <image>, <text>, <use>, <g> wrappers, full-canvas
    background <rect>, and vectors that don't overlap any image.

    Returns (new_svg_text, info_dict).
    """
    try:
        # Parse with namespace-aware ET (svg lives in xhtml NS)
        root = ET.fromstring(svg_text)
    except ET.ParseError as e:
        logger.warning("cleanup parse failed: %s", e)
        return svg_text, {"error": str(e), "removed": 0}

    # Collect all <image> bboxes
    image_bboxes = []
    for el in root.iter(NS + "image"):
        bb = _bbox_of_rect(el)
        if bb:
            image_bboxes.append(bb)

    if not image_bboxes:
        return svg_text, {"removed": 0, "reason": "no images"}

    # Walk parents to find vector primitives to remove
    removed_count = 0
    removed_details = []
    # ET doesn't give parent refs; use iterator with manual parent map
    parent_map = {child: parent for parent in root.iter() for child in parent}
    to_remove = []
    for el in root.iter():
        tag = el.tag.replace(NS, "")
        if tag not in CLEANUP_TAGS:
            continue
        # Skip canvas background
        if _is_canvas_background(el, canvas_w, canvas_h):
            continue
        bb = _bbox_of(el)
        if not bb:
            continue
        # Check against every image bbox
        for img_bb in image_bboxes:
            iou = _iou(bb, img_bb)
            inside = _bbox_inside(bb, img_bb)
            # vector area inside image bbox
            vec_area = (bb[2] - bb[0]) * (bb[3] - bb[1])
            if vec_area > 0:
                ix1 = max(bb[0], img_bb[0]); iy1 = max(bb[1], img_bb[1])
                ix2 = min(bb[2], img_bb[2]); iy2 = min(bb[3], img_bb[3])
                inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                contains_frac = inter / vec_area
            else:
                contains_frac = 0
            if iou >= iou_threshold or contains_frac >= contains_threshold:
                to_remove.append((el, tag, bb, iou, contains_frac))
                break

    for el, tag, bb, iou, contains in to_remove:
        parent = parent_map.get(el)
        if parent is not None:
            parent.remove(el)
            removed_count += 1
            removed_details.append({
                "tag": tag,
                "bbox": [round(v) for v in bb],
                "iou": round(iou, 2),
                "contains": round(contains, 2),
                "id": el.get("id", ""),
            })

    if removed_count == 0:
        return svg_text, {"removed": 0, "details": []}

    # Serialise back. ET puts ns prefix; we want clean output.
    # Strategy: use tostring then strip the ns0: prefix.
    # First register the ns to suppress prefixes.
    ET.register_namespace("", "http://www.w3.org/2000/svg")
    new_svg = ET.tostring(root, encoding="unicode")
    return new_svg, {"removed": removed_count, "details": removed_details}
