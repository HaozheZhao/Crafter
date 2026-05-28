"""Minimal post-classifier override.

Two narrow overrides on top of the per-icon VLM classifier
(``vector_codeable.json``):
  1. Vectorable bars (single bars / few-bar series, NOT data-driven
     histograms or bar charts).
  2. A series of circles forming a token row / neuron diagram.

Everything else: trust the VLM classifier.

This file does NOT touch crop bboxes or placement logic — only the
raster/vector decision.
"""
from __future__ import annotations

import re
from typing import Optional, Tuple

# Vectorable bar kinds — true simple bar primitives only.
# color_strip / color_sequence / color_patch / single_swatch are not
# here: they carry colour as meaningful content and stay raster.
# histogram / bar_chart also excluded: data-driven, need raster.
KIND_BAR_VECTOR = {
    "vertical_bar", "horizontal_bar",
    "output_bars",            # img4 ic15 — flat orange rounded bars
    "bars",
}

# Circle-series kinds. Tokens / neurons drawn as circle rows.
KIND_CIRCLE_SERIES_VECTOR = {
    "token_row", "token_pair",
    "neuron_row", "neuron_pair", "neurons",
}

# Description patterns that confirm circle-series (as a sanity check
# on top of kind — only override if kind matches AND desc mentions
# circles / dots / tokens).
CIRCLE_SERIES_DESC = re.compile(
    r"\b(circles?|dots?|tokens?|nodes?|spheres?)\b",
    re.IGNORECASE,
)

# Stack-of-rounded-blocks kinds. Vector unless an icon is mentioned
# inside the description.
KIND_STACK_VECTOR_IF_PLAIN = {
    "layer_stack", "stack_tile",
}

# Iconography keywords — if any of these appear in the description we
# DO NOT override KIND_STACK_VECTOR_IF_PLAIN (treat as raster instead).
ICON_CONTENT_KEYWORDS = {
    "brain", "logo", "photo", "image", "thumbnail",
    "heatmap", "histogram", "scatter", "matrix",
    "snowflake", "clock", "axes", "axis", "star",
    "icon", "drawing", "render",
    "filmstrip", "frame", "video",
    "letter", "digit", "glyph",
    "face", "person", "animal",
}


def classify_override(
    kind: str,
    simple_desc: str,
    detailed_desc: str,
) -> Optional[Tuple[bool, str]]:
    """Return (vector_codeable, reason) override, or None to keep VLM verdict."""
    k = (kind or "").strip().lower()
    desc = f"{simple_desc} {detailed_desc}"
    desc_l = desc.lower()

    if k in KIND_BAR_VECTOR:
        return (True, f"override: kind '{k}' is a vectorable bar / strip / patch")

    if k in KIND_CIRCLE_SERIES_VECTOR and CIRCLE_SERIES_DESC.search(desc):
        return (True, f"override: kind '{k}' is a circle/token/neuron "
                f"series (desc mentions circles/tokens)")

    if k in KIND_STACK_VECTOR_IF_PLAIN:
        if any(w in desc_l for w in ICON_CONTENT_KEYWORDS):
            return None  # has icon content → keep raster
        return (True, f"override: kind '{k}' is a stack of plain blocks "
                f"(no iconography keywords in desc)")

    return None
