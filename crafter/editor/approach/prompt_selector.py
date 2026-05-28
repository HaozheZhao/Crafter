"""Step A — prompt selector + extraction config tuner.

Given a style profile (from style_analyzer.py), returns:
  • a SAM3 min_score floor that fits the figure complexity
  • a max_offset multiplier for the placement filter
  • additional captions to push into the segment_text fallback chain
    (style-specific decorative elements that the default caption set
    would miss)
  • a `prompt_addendum` string to append to whatever caption is sent
    to SAM3 (frames the search task — e.g. "schematic icon, flat
    coloured shape, no text").

Pure logic. No API calls. ~150 lines, mostly tables of style→config.
"""
from __future__ import annotations

from typing import Any

# ----------------------------------------------------------------- defaults
DEFAULT_CONFIG: dict[str, Any] = {
    "sam3_min_score": 0.20,
    "max_offset_mult": 1.0,
    "extra_captions": [],
    "prompt_addendum": "",
    "drop_unanchored": False,
    "notes": "default config",
}

# ----------------------------------------------------------------- per-style
STYLE_TUNING: dict[str, dict[str, Any]] = {
    "academic_pipeline": {
        "sam3_min_score": 0.25,
        "max_offset_mult": 1.0,
        "extra_captions": [],
        "prompt_addendum": (
            " (academic pipeline figure: clean schematic icon, "
            "flat fill, no decorative shading)"),
        "drop_unanchored": False,
        "notes": "tightest min_score; placement anchor reliable",
    },
    "academic_arch": {
        "sam3_min_score": 0.22,
        "max_offset_mult": 1.2,
        "extra_captions": ["stacked rectangular block",
                           "encoder block", "decoder block"],
        "prompt_addendum": (
            " (academic architecture figure: layered blocks, "
            "sometimes with thin coloured borders)"),
        "drop_unanchored": False,
        "notes": "stacked-block aware; allow slightly larger offset",
    },
    "academic_plot": {
        "sam3_min_score": 0.30,
        "max_offset_mult": 0.8,
        "extra_captions": [],
        "prompt_addendum": (
            " (plot/chart: prefer graph elements over symbols)"),
        "drop_unanchored": True,
        "notes": "plot-dominated — drop loose extracted icons",
    },
    "infographic": {
        "sam3_min_score": 0.18,
        "max_offset_mult": 1.4,
        "extra_captions": ["decorative illustration",
                           "small pictogram",
                           "stylised icon",
                           "abstract shape"],
        "prompt_addendum": (
            " (infographic-style: rich illustration, often colourful, "
            "may have soft shading)"),
        "drop_unanchored": False,
        "notes": "loose threshold; broaden caption set",
    },
    "poster": {
        "sam3_min_score": 0.20,
        "max_offset_mult": 1.5,
        "extra_captions": ["section icon", "stylised illustration",
                           "logo"],
        "prompt_addendum": (
            " (poster: large-scale illustrations + section icons; "
            "expect varied iconography across panels)"),
        "drop_unanchored": False,
        "notes": "looser placement (large canvas)",
    },
    "table": {
        "sam3_min_score": 0.35,
        "max_offset_mult": 0.7,
        "extra_captions": [],
        "prompt_addendum": (
            " (table-dominant figure: most content is text in cells; "
            "real icons are rare)"),
        "drop_unanchored": True,
        "notes": "minimal icon expectation",
    },
}

# ----------------------------------------------------------------- modifiers
def _apply_complexity(cfg: dict, complexity: str) -> dict:
    """complex figures need looser thresholds, simple ones tighter."""
    if complexity == "complex":
        cfg["sam3_min_score"] = max(0.12, cfg["sam3_min_score"] - 0.05)
        cfg["max_offset_mult"] *= 1.15
    elif complexity == "simple":
        cfg["sam3_min_score"] = min(0.40, cfg["sam3_min_score"] + 0.05)
        cfg["max_offset_mult"] *= 0.9
    return cfg


def _apply_density(cfg: dict, density: str) -> dict:
    """dense figures = candidates collide more, tighten offset to keep
    the right one."""
    if density == "dense":
        cfg["max_offset_mult"] *= 0.85
    elif density == "sparse":
        cfg["max_offset_mult"] *= 1.15
    return cfg


def _apply_special(cfg: dict, features: list[str]) -> dict:
    if "math" in features:
        cfg["extra_captions"].append("mathematical equation block")
        cfg["prompt_addendum"] += " Math equations may appear as text."
    if "code" in features:
        cfg["extra_captions"].append("code snippet box")
        cfg["prompt_addendum"] += " Code blocks may use monospace."
    if "callout_arrow" in features:
        cfg["extra_captions"].append("callout arrow with label")
    return cfg


def select(profile: dict) -> dict:
    """Return an extraction config tuned to the figure's profile."""
    style = profile.get("style", "academic_pipeline")
    base = STYLE_TUNING.get(style, DEFAULT_CONFIG).copy()
    base["extra_captions"] = list(base.get("extra_captions", []))
    base = _apply_complexity(base, profile.get("complexity", "medium"))
    base = _apply_density(base, profile.get("icon_density", "medium"))
    base = _apply_special(base, profile.get("special_features", []) or [])
    base["style_profile"] = profile
    base["sam3_min_score"] = round(float(base["sam3_min_score"]), 3)
    base["max_offset_mult"] = round(float(base["max_offset_mult"]), 3)
    return base
