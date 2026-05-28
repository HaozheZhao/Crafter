"""Vector style drift checker (Item 5).

Flat academic figures rarely have feDropShadow / radial gradients /
3D-style shading. If the LLM adds these to vector elements,
the overall style drifts. Detect and report for removal.

Rule: any of feDropShadow / linearGradient / radialGradient /
filter= / feGaussianBlur in the SVG → flag.
"""
from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from pathlib import Path

NS = "{http://www.w3.org/2000/svg}"
logger = logging.getLogger("checkers.style")

DRIFT_TAGS = {
    "feDropShadow", "linearGradient", "radialGradient",
    "feGaussianBlur", "feSpecularLighting", "feDiffuseLighting",
    "feMorphology", "feConvolveMatrix",
}


def check(svg_path: Path) -> list[dict]:
    fixes: list[dict] = []
    try:
        text = svg_path.read_text(encoding="utf-8")
        root = ET.parse(svg_path).getroot()
    except ET.ParseError:
        return fixes

    drifts: dict[str, int] = {t: 0 for t in DRIFT_TAGS}
    for tag in DRIFT_TAGS:
        # Count via text since drift tags are inside <defs>
        drifts[tag] = len(re.findall(rf"<{re.escape(tag)}[\s>]", text))

    # Count usages of filter= attribute (likely shadow / blur references)
    filter_attrs = len(re.findall(r'\bfilter\s*=\s*"', text))

    drifts_present = {k: v for k, v in drifts.items() if v > 0}
    if drifts_present or filter_attrs > 0:
        summary = ", ".join(f"{k}×{v}" for k, v in drifts_present.items())
        if filter_attrs:
            summary += f", filter= attr ×{filter_attrs}"
        fixes.append({
            "kind": "style_drift",
            "drift_elements": drifts_present,
            "filter_attrs": filter_attrs,
            "fix": (f"style drift detected ({summary}). "
                    "REMOVE all feDropShadow, linearGradient, radialGradient, "
                    "feGaussianBlur, and any filter= attributes on "
                    "structural shapes. Academic figures use FLAT colours."),
        })

    return fixes
