"""RolePlanner: generates a per-role structural blueprint that augments
`craft_input.description` BEFORE variant generation.

Why: `scripts/gen_mentor_poster.py` (the proven V13 poster recipe) only
works because the human author hand-wrote a 50-line poster blueprint
into `description` — explicit TOP BANNER (title + authors +
affiliations), Row 1-5 sections, BOTTOM STRIP. The R4 generic role
preamble is too abstract to fill the same gap; outputs miss the title
banner, miss the infographic's casual callout structure, etc.

This agent runs BEFORE variant generation and inserts a role-specific blueprint
extracted from `paper_text + caption`. It's an LLM call (router.plan)
producing a structured plan, then formatted into a description-style
string. Activates for role ∈ {"poster", "infographic"} and any
non-academic free-form role; academic / "" skip planning to preserve R1.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Optional

from crafter.shared.model_router import ModelRouter

logger = logging.getLogger(__name__)


POSTER_PLANNER_SYSTEM = """\
You are a senior academic poster designer. Given the paper's text + caption,
produce a STRUCTURAL BLUEPRINT for an academic conference poster. The
blueprint will be fed to an image-gen model that draws the poster.

Your blueprint MUST include, in this order:

1. PAPER TITLE — the exact title from the paper (or your best inference if
   missing). Render-emphasis: "render the TITLE in LARGE BOLD TYPOGRAPHY
   across the top banner".
2. AUTHOR LIST — comma-separated authors with affiliation superscripts.
3. AFFILIATIONS — institution names matching the superscripts.
4. CONFERENCE BADGE — the venue / year (e.g. "CVPR 2024", "NeurIPS 2024",
   or "ACL 2024" — infer from context); placed in a top-right corner.
5. ROW-BY-ROW BODY LAYOUT — 3-5 rows, each labelled "Row N: <Section name>"
   covering Motivation / Background, Method / Approach, Results /
   Experiments, Ablation, Conclusion. Per row state which subset of the
   paper's content goes there + suggested layout (single column,
   2-column, full-width, with figures vs tables).
6. BOTTOM STRIP — key takeaway one-liner OR project URL OR QR code stub.

Output STRICT JSON:
{
  "title": "<exact paper title>",
  "authors": "<author list verbatim>",
  "affiliations": "<institution list>",
  "conference": "<venue + year>",
  "rows": [
    {"row": 1, "section": "Motivation", "content_summary": "...",
     "layout_hint": "2 columns: problem on left, our solution on right"},
    {"row": 2, ...}, ...
  ],
  "bottom_strip": "<one-liner or project URL>"
}
"""


INFOGRAPHIC_PLANNER_SYSTEM = """\
You are a magazine-explainer designer (think HuggingFace blog,
Distill, Quanta Magazine). Given the paper's text + caption, produce a
STRUCTURAL BLUEPRINT for a casual EXPLAINER infographic — NOT a paper
figure.

Your blueprint MUST identify:

1. STEP COUNT — how many numbered steps the explainer needs (typical 3-7).
2. PER-STEP CONTENT — what each step shows. Each step gets a short
   friendly label + one-sentence body. Steps connected by expressive
   arrows.
3. VISUAL METAPHOR — pick one concrete metaphor / analogy that ties the
   whole explainer together (e.g. "GPUs as factory workers passing
   tokens via conveyor belts", "attention as a librarian routing books").
4. CALLOUT STYLE — where to place orange-circle numbered callouts, how
   they connect to step blocks.
5. LAYOUT FLOW — horizontal (left-to-right wide) vs vertical
   (top-to-bottom tall), based on the steps.

Output STRICT JSON:
{
  "title_label": "<friendly headline, NOT a paper title>",
  "metaphor": "<the central visual analogy>",
  "steps": [
    {"step": 1, "label": "<short>", "body": "<one sentence>"},
    {"step": 2, ...}, ...
  ],
  "layout_flow": "horizontal" | "vertical",
  "color_palette": "<palette description, e.g. orange + teal + cream>"
}
"""


GENERIC_PLANNER_SYSTEM = """\
You are a creative figure designer. Given a free-form role descriptor +
the paper's text + caption, produce a STRUCTURAL BLUEPRINT for a figure
in that role.

Output STRICT JSON:
{
  "format_summary": "<one sentence describing the format/medium>",
  "visual_conventions": ["<convention 1>", "<convention 2>", ...],
  "layout_components": [
    {"name": "<component>", "content": "<what to put>",
     "placement": "<where>"}
  ],
  "palette_typography": "<palette + typography hints>",
  "what_to_avoid": ["<academic conventions to avoid>", ...]
}
"""


def _parse_json(text: str) -> Optional[dict]:
    """Best-effort JSON extraction from an LLM response."""
    text = (text or "").strip()
    if text.startswith("```"):
        # strip code fence
        m = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
        if m: text = m.group(1).strip()
    try:
        return json.loads(text)
    except Exception:
        # try to locate the outermost JSON object
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try: return json.loads(m.group(0))
            except Exception: pass
    return None


def _format_poster_blueprint(plan: dict) -> str:
    title = plan.get("title", "(unknown title)")
    authors = plan.get("authors", "")
    affs = plan.get("affiliations", "")
    venue = plan.get("conference", "")
    rows = plan.get("rows") or []
    bottom = plan.get("bottom_strip", "")
    parts = [
        "POSTER LAYOUT BLUEPRINT (must follow):",
        "",
        f"  TOP BANNER (full width, the FIRST and LARGEST text on the canvas):",
        f"    *** PAPER TITLE: \"{title}\"",
        f"        Render this exact title TEXT in the LARGEST TYPOGRAPHY of",
        f"        the entire poster — bold, centered, prominent, READABLE",
        f"        FROM ACROSS THE ROOM. The title is the single most",
        f"        visible visual element of the poster.",
        f"        This title is the PAPER TITLE — NOT a body section label.",
        f"        Do NOT replace it with a Method / Motivation / Results",
        f"        header — those are body section headers, much smaller.",
    ]
    if authors:
        parts.append(f"    - AUTHORS: {authors}")
    if affs:
        parts.append(f"    - AFFILIATIONS: {affs}")
    if venue:
        parts.append(f"    - CONFERENCE BADGE: {venue} (corner, secondary size)")
    parts.append("")
    parts.append("  BODY ROWS (top-to-bottom):")
    for r in rows:
        rn = r.get("row", "?")
        section = r.get("section", "?")
        summary = r.get("content_summary", "")
        layout = r.get("layout_hint", "")
        parts.append(f"    Row {rn} — {section}:")
        if summary: parts.append(f"      content: {summary}")
        if layout:  parts.append(f"      layout: {layout}")
    if bottom:
        parts.append("")
        parts.append(f"  BOTTOM STRIP (full width, at bottom):")
        parts.append(f"    {bottom}")
    return "\n".join(parts)


def _format_infographic_blueprint(plan: dict) -> str:
    title_label = plan.get("title_label", "")
    metaphor = plan.get("metaphor", "")
    steps = plan.get("steps") or []
    flow = plan.get("layout_flow", "horizontal")
    palette = plan.get("color_palette", "")
    parts = ["INFOGRAPHIC LAYOUT BLUEPRINT (must follow):", ""]
    if title_label:
        parts.append(f"  TITLE LABEL (friendly, NOT paper title): \"{title_label}\"")
    if metaphor:
        parts.append(f"  CENTRAL METAPHOR: {metaphor}")
    if palette:
        parts.append(f"  COLOR PALETTE: {palette}")
    parts.append(f"  LAYOUT FLOW: {flow} (steps arranged in this direction)")
    parts.append("")
    parts.append(f"  STEPS ({len(steps)} total — each gets an ORANGE CIRCLE callout with the number, plus a short friendly label):")
    for s in steps:
        sn = s.get("step", "?")
        lab = s.get("label", "")
        body = s.get("body", "")
        parts.append(f"    {sn}. {lab} — {body}")
    parts.append("")
    parts.append("  Render expressive curved arrows (red / colored, thick stroke) connecting consecutive steps.")
    parts.append("  Use illustrative icons or cartoonish drawings for each step. NOT thin black academic arrows.")
    return "\n".join(parts)


def _format_generic_blueprint(plan: dict, role: str) -> str:
    fmt = plan.get("format_summary", "")
    conventions = plan.get("visual_conventions") or []
    components = plan.get("layout_components") or []
    palette = plan.get("palette_typography", "")
    avoid = plan.get("what_to_avoid") or []
    parts = [f"ROLE-SPECIFIC BLUEPRINT for \"{role}\" (must follow):", ""]
    if fmt: parts.append(f"  FORMAT: {fmt}")
    if conventions:
        parts.append("  VISUAL CONVENTIONS:")
        for c in conventions: parts.append(f"    - {c}")
    if components:
        parts.append("  LAYOUT COMPONENTS:")
        for c in components:
            parts.append(f"    - {c.get('name','?')}: {c.get('content','')} ({c.get('placement','')})")
    if palette: parts.append(f"  PALETTE / TYPOGRAPHY: {palette}")
    if avoid:
        parts.append("  AVOID (do NOT use these academic conventions):")
        for a in avoid: parts.append(f"    - {a}")
    return "\n".join(parts)


class RolePlanner:
    """LLM-driven structural blueprint generator for non-academic roles.

    The visual gap on poster / infographic T2I is structural (missing
    title banner, missing casual callouts) and stems from the description
    not carrying a layout blueprint. This agent generates that blueprint
    automatically from paper_text + caption.
    """

    def __init__(self, router: ModelRouter):
        self.router = router

    def plan(self, paper_text: str, caption: str, role: str) -> str:
        """Return a blueprint string to APPEND to the description, or "" if
        no planning needed (academic / empty role).
        """
        role_norm = (role or "").lower().strip()
        if role_norm in ("", "academic"):
            return ""

        if role_norm == "poster":
            sys_prompt = POSTER_PLANNER_SYSTEM
            formatter = _format_poster_blueprint
        elif role_norm == "infographic":
            sys_prompt = INFOGRAPHIC_PLANNER_SYSTEM
            formatter = _format_infographic_blueprint
        else:
            sys_prompt = GENERIC_PLANNER_SYSTEM
            formatter = lambda p: _format_generic_blueprint(p, role)

        user_prompt = (
            f"Paper caption / figure description: {caption[:400]}\n\n"
            f"Paper text (truncated): {paper_text[:6000]}\n\n"
            f"Role descriptor: \"{role}\"\n\n"
            f"Output STRICT JSON per the schema in the system prompt."
        )

        try:
            raw = self.router.plan(
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.4,
                max_tokens=2000,
            )
        except Exception as e:
            logger.warning(f"RolePlanner LLM call failed for role={role!r}: {e}")
            return ""

        plan = _parse_json(raw)
        if not plan:
            logger.warning(f"RolePlanner could not parse JSON for role={role!r}; raw[:200]={raw[:200]!r}")
            return ""

        return formatter(plan)
