"""Step A — image style analyzer.

One gpt-5.5 vision call per image. Returns a structured profile that
the prompt_selector uses to tune extraction behaviour. Cached to a JSON
file alongside the per-image runs/ directory so it never re-runs unless
the cache is deleted.

Output schema (returned as a dict):
{
  "style": "academic_pipeline" | "academic_arch" | "academic_plot" |
           "infographic" | "poster" | "table",
  "complexity": "simple" | "medium" | "complex",
  "estimated_num_icons": int,
  "estimated_num_text_blocks": int,
  "icon_density": "sparse" | "medium" | "dense",
  "color_palette": "muted" | "vivid" | "monochrome",
  "layout_direction": "ltr" | "ttb" | "grid" | "radial",
  "special_features": ["math", "code", "rotated_text", "small_caps", ...],
  "extraction_hints": str   // 1-2 sentences telling the next stage what
                            // is unusual about this figure
}

Design notes for the prompt:
  - Demand the JSON exactly. No prose, no markdown.
  - Give explicit definitions of each style so the model picks reliably.
  - Cap special_features and extraction_hints to keep the output small.
"""
from __future__ import annotations
import os

import base64
import io
import json
import logging
import re
import sys
from pathlib import Path

from PIL import Image

REPO = Path(os.environ.get("CRAFTER_HOME", str(Path(__file__).resolve().parents[2])))

from crafter.editor.approach.iter_svg_fix import call_model  # noqa: E402

logger = logging.getLogger("style_analyzer")


STYLE_DEFINITIONS = """STYLE LABELS (pick exactly ONE):
  • academic_pipeline — left-to-right or top-to-bottom flow of stages
    or modules, mostly schematic boxes + arrows + small icons
    (typical methods figure of an arXiv paper).
  • academic_arch    — layered architecture diagram (e.g. NN
    architectures with stacked blocks, encoder/decoder).
  • academic_plot    — a chart / plot dominated by axes, bars, lines
    (less iconography, more data marks).
  • infographic      — rich illustrations + diverse icons + decorative
    layout (lillog / Olah / hf_blog style).
  • poster           — large title text + multi-section grid + icons
    + reference list (conference poster).
  • table            — dominant tabular grid (rows + columns + cells).
"""

ANALYZER_PROMPT = """You are analysing one academic figure to plan an
SVG re-creation. Look at the image carefully and return a strict JSON
profile. Be concise and accurate — downstream tools rely on these
labels.

""" + STYLE_DEFINITIONS + """

OTHER FIELDS:
  • complexity:
      simple  = ≤ 6 icons / ≤ 1 panel
      medium  = 7-20 icons / 2-4 panels
      complex = > 20 icons OR > 4 panels OR dense visual content.
  • estimated_num_icons:        integer estimate of distinct icons /
                                 raster-style elements.
  • estimated_num_text_blocks:  integer estimate of distinct text
                                 labels (titles + body + small).
  • icon_density: how packed icons are.
      sparse  = lots of whitespace between icons
      medium  = some breathing room
      dense   = icons crowd each other
  • color_palette:
      muted        = pastels / desaturated greys
      vivid        = saturated / high-contrast
      monochrome   = mostly one hue + greys
  • layout_direction:
      ltr   = left-to-right primary flow
      ttb   = top-to-bottom primary flow
      grid  = independent cells in a 2-D grid
      radial= centre-out radial arrangement
  • special_features (string array, max 4):
      "math"          if equations or LaTeX-style symbols are visible
      "code"          if code snippets are visible
      "rotated_text"  if any axis label or sidebar text is rotated
      "small_caps"    if SMALL-CAPS or all-caps section headers
      "subscripts"    if Q_K_V style sub/super-scripts
      "table_inset"   if a table is embedded inside the figure
      "callout_arrow" if labelled arrows pointing into a sub-region
  • extraction_hints (1-2 sentences):
      ANY non-obvious thing the next extractor should know — unusual
      icon shapes, atypical layout, mixed-script text, etc.
      Empty string if nothing notable.
  • font_hints (object):
      Per-region font/style guidance for the SVG renderer.
        title_style:    "bold_serif" | "bold_sans" | "regular_sans" | "regular_serif" | null
        body_style:     same options or null
        math_present:   true if any italic-Greek / LaTeX-style math is visible
        cursive_present:true if any handwritten or calligraphic text is visible
        body_font_family_guess: best-guess family name (e.g. "Helvetica",
                                "Times", "Computer Modern", "DejaVu Sans") or null
        emphasis_uses:  ["italic_for_var_names", "bold_for_section_titles", ...]
                        empty list if not detectable.

Return STRICT JSON only:
{
  "style": "academic_pipeline",
  "complexity": "medium",
  "estimated_num_icons": 14,
  "estimated_num_text_blocks": 30,
  "icon_density": "medium",
  "color_palette": "muted",
  "layout_direction": "ltr",
  "special_features": ["rotated_text", "subscripts"],
  "extraction_hints": "Some panels use a thin dashed outline; treat them as logical group boundaries, not decorative.",
  "font_hints": {
    "title_style": "bold_sans",
    "body_style": "regular_sans",
    "math_present": true,
    "cursive_present": false,
    "body_font_family_guess": "Helvetica",
    "emphasis_uses": ["italic_for_var_names"]
  }
}
"""


def _b64(path: Path, max_dim: int = 1500) -> str:
    img = Image.open(path).convert("RGB")
    if max(img.size) > max_dim:
        scale = max_dim / max(img.size)
        img = img.resize((int(img.width * scale), int(img.height * scale)),
                         Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88)
    return base64.b64encode(buf.getvalue()).decode()


def _extract_json(text: str) -> dict | None:
    text = text.strip()
    m = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if m:
        text = m.group(1).strip()
    for s in range(len(text)):
        if text[s] != "{":
            continue
        depth = 0
        for e in range(s, len(text)):
            if text[e] == "{":
                depth += 1
            elif text[e] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[s:e + 1])
                    except json.JSONDecodeError:
                        break
    return None


VALID_STYLES = {"academic_pipeline", "academic_arch", "academic_plot",
                "infographic", "poster", "table"}
VALID_COMPLEXITY = {"simple", "medium", "complex"}
VALID_DENSITY = {"sparse", "medium", "dense"}
VALID_PALETTE = {"muted", "vivid", "monochrome"}
VALID_DIRECTION = {"ltr", "ttb", "grid", "radial"}


VALID_TITLE_STYLE = {"bold_serif", "bold_sans", "regular_sans",
                     "regular_serif"}


def _normalise_font(fh: dict | None) -> dict:
    f = fh if isinstance(fh, dict) else {}
    return {
        "title_style": f.get("title_style")
                  if f.get("title_style") in VALID_TITLE_STYLE else None,
        "body_style": f.get("body_style")
                  if f.get("body_style") in VALID_TITLE_STYLE else None,
        "math_present": bool(f.get("math_present", False)),
        "cursive_present": bool(f.get("cursive_present", False)),
        "body_font_family_guess": str(f["body_font_family_guess"])[:60]
                  if f.get("body_font_family_guess") else None,
        "emphasis_uses": [str(x)[:40] for x in
                          (f.get("emphasis_uses") or [])][:4],
    }


def _normalise(profile: dict) -> dict:
    """Ensure all expected fields exist with sensible defaults / valid values."""
    p = profile if isinstance(profile, dict) else {}
    out = {
        "style": p.get("style", "academic_pipeline")
                  if p.get("style") in VALID_STYLES else "academic_pipeline",
        "complexity": p.get("complexity", "medium")
                  if p.get("complexity") in VALID_COMPLEXITY else "medium",
        "estimated_num_icons": int(p.get("estimated_num_icons", 0) or 0),
        "estimated_num_text_blocks": int(p.get("estimated_num_text_blocks", 0) or 0),
        "icon_density": p.get("icon_density", "medium")
                  if p.get("icon_density") in VALID_DENSITY else "medium",
        "color_palette": p.get("color_palette", "muted")
                  if p.get("color_palette") in VALID_PALETTE else "muted",
        "layout_direction": p.get("layout_direction", "ltr")
                  if p.get("layout_direction") in VALID_DIRECTION else "ltr",
        "special_features": [str(x)[:30] for x in
                             (p.get("special_features") or [])][:4],
        "extraction_hints": str(p.get("extraction_hints", ""))[:300],
        "font_hints": _normalise_font(p.get("font_hints")),
    }
    return out


def analyze(
    original_png: Path,
    cache_path: Path | None = None,
    model: str = "openai/gpt-5.5",
) -> dict:
    """Analyse the figure style. Returns a normalised dict.

    If cache_path is provided and exists, returns cached result.
    Otherwise calls the VLM and writes cache.
    """
    if cache_path and cache_path.exists():
        try:
            return json.loads(cache_path.read_text())
        except Exception:
            pass

    msgs = [{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url":
                {"url": f"data:image/jpeg;base64,{_b64(original_png)}"}},
            {"type": "text", "text": ANALYZER_PROMPT},
        ],
    }]
    try:
        resp = call_model(msgs, model=model, max_tokens=1500,
                          label="stage6_style_analyzer")
    except Exception as e:
        logger.warning("style analyzer api failed: %s — using defaults", e)
        return _normalise({})

    parsed = _extract_json(resp) or {}
    profile = _normalise(parsed)

    if cache_path:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(profile, indent=2))
        except Exception:
            pass

    return profile
