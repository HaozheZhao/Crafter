"""SVG arrow checker.

Validates two failure modes (Item 3a from user feedback):
  - Oversized arrowhead — markerWidth/markerHeight too large vs arrow body
  - Dangling/tangled — arrow endpoints land in mid-air, not near any
    target element bbox.
"""
from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from pathlib import Path

NS = "{http://www.w3.org/2000/svg}"
logger = logging.getLogger("checkers.arrow")

MAX_ARROWHEAD_PX = 20
MAX_ARROWHEAD_FRAC = 0.30          # > 30% of arrow body length is too big
ENDPOINT_NEIGHBOUR_PX = 12         # endpoint must be ≤ this px from a bbox


def _markers(root: ET.Element) -> dict[str, dict]:
    """Collect <marker id="..."> definitions from <defs>."""
    markers = {}
    for m in root.iter(NS + "marker"):
        mid = m.get("id", "")
        if not mid:
            continue
        try:
            mw = float(m.get("markerWidth", 6))
            mh = float(m.get("markerHeight", 6))
        except (TypeError, ValueError):
            mw = mh = 6
        markers[mid] = {"w": mw, "h": mh}
    return markers


def _line_endpoints(el: ET.Element) -> list[tuple[float, float]] | None:
    tag = el.tag.replace(NS, "")
    g = el.get
    try:
        if tag == "line":
            return [(float(g("x1", 0)), float(g("y1", 0))),
                    (float(g("x2", 0)), float(g("y2", 0)))]
        if tag == "polyline" or tag == "polygon":
            pts = g("points", "").replace(",", " ").split()
            nums = [float(p) for p in pts if p]
            if len(nums) >= 4:
                return [(nums[0], nums[1]), (nums[-2], nums[-1])]
        if tag == "path":
            d = g("d", "")
            nums = [float(n) for n in re.findall(r"-?\d+(?:\.\d+)?", d)]
            if len(nums) >= 4:
                return [(nums[0], nums[1]), (nums[-2], nums[-1])]
    except (TypeError, ValueError):
        return None
    return None


def _length(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


def _element_bboxes(root: ET.Element) -> list[tuple[str, list[float]]]:
    """Return [(id_or_tag, [x1,y1,x2,y2]), ...] for elements an arrow could attach to."""
    out = []
    for el in root.iter():
        tag = el.tag.replace(NS, "")
        if tag in ("rect", "image", "circle", "ellipse"):
            try:
                if tag in ("rect", "image"):
                    x = float(el.get("x", 0)); y = float(el.get("y", 0))
                    w = float(el.get("width", 0)); h = float(el.get("height", 0))
                    bb = [x, y, x + w, y + h]
                elif tag == "circle":
                    cx = float(el.get("cx", 0)); cy = float(el.get("cy", 0))
                    r = float(el.get("r", 0))
                    bb = [cx - r, cy - r, cx + r, cy + r]
                else:
                    cx = float(el.get("cx", 0)); cy = float(el.get("cy", 0))
                    rx = float(el.get("rx", 0)); ry = float(el.get("ry", 0))
                    bb = [cx - rx, cy - ry, cx + rx, cy + ry]
                if bb[2] <= bb[0] or bb[3] <= bb[1]:
                    continue
                out.append((el.get("id", tag), bb))
            except (TypeError, ValueError):
                continue
    return out


def _near_any(point: tuple[float, float],
              bboxes: list[tuple[str, list[float]]],
              threshold: float = ENDPOINT_NEIGHBOUR_PX) -> str | None:
    px, py = point
    for elid, bb in bboxes:
        # distance from point to bbox (0 if inside)
        dx = max(bb[0] - px, 0, px - bb[2])
        dy = max(bb[1] - py, 0, py - bb[3])
        if (dx ** 2 + dy ** 2) ** 0.5 <= threshold:
            return elid
    return None


def check(svg_path: Path) -> list[dict]:
    fixes: list[dict] = []
    try:
        root = ET.parse(svg_path).getroot()
    except ET.ParseError:
        return fixes

    markers = _markers(root)
    elements = _element_bboxes(root)

    for el in root.iter():
        tag = el.tag.replace(NS, "")
        if tag not in ("line", "polyline", "polygon", "path"):
            continue
        marker_end = el.get("marker-end", "") or el.get("marker-start", "")
        if not marker_end:
            continue
        ends = _line_endpoints(el)
        if not ends or len(ends) < 2:
            continue

        body_len = _length(ends[0], ends[1])
        # Resolve marker by url(#id)
        m = re.match(r"url\(#([^)]+)\)", marker_end.strip())
        if m and m.group(1) in markers:
            mw = markers[m.group(1)]["w"]
            mh = markers[m.group(1)]["h"]
            head_size = max(mw, mh)
            # OVERSIZED check
            if head_size > MAX_ARROWHEAD_PX or \
                    (body_len > 0 and head_size / body_len > MAX_ARROWHEAD_FRAC):
                fixes.append({
                    "kind": "oversized_arrow",
                    "marker_id": m.group(1),
                    "head_w": mw, "head_h": mh,
                    "body_length_px": round(body_len, 1),
                    "fix": (f"<marker id='{m.group(1)}'> has width={mw} "
                            f"height={mh}, too large vs arrow body "
                            f"({body_len:.0f}px). Reduce to "
                            f"markerWidth=8 markerHeight=6"),
                })

        # DANGLING / TANGLED check — both endpoints in mid-air
        a = _near_any(ends[0], elements)
        b = _near_any(ends[1], elements)
        if a is None and b is None:
            elid = el.get("id", "(no id)")
            fixes.append({
                "kind": "dangling_arrow",
                "id": elid,
                "endpoints": [list(ends[0]), list(ends[1])],
                "fix": (f"arrow {elid} endpoints both land in mid-air "
                        f"({ends[0]}, {ends[1]}); attach each end to a "
                        f"nearby element or remove the arrow"),
            })

    return fixes
