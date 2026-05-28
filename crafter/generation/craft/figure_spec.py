"""EvolvingFigureSpec — single source of truth that all agents edit.

Agents coordinate through a shared evolving spec rather than producing
independent prompt blocks that get concatenated and conflict.

Architecture:
  - VG writes spec.required_elements (the "what to draw" list)
  - planner writes spec.layout (the "how to arrange" structure)
  - stylist writes spec.style (the "what it looks like")
  - SE/critic READ critique and EDIT existing spec fields (not append free text)
  - region-fixer writes spec.region_fixes (applied via PIL composite, NOT in prompt)
  - render_image_gen_prompt(spec, paper) produces single coherent prompt
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class VisualElement:
    """A required visual element (icon / photo / drawing / formula).
    Originally set by VisualGrounder; can be edited by SE."""
    id: str                     # e.g. "elem_0", stable for SE to reference
    name: str                   # "hippocampus brain icon"
    realization: str            # "anatomical brain illustration in soft purple"
    placement: str              # "top-right corner as analogy panel"
    must_have: bool = True      # SE may demote to False; never deletes
    size_hint: str = ""         # SE may set: "small" | "medium" | "large"
    notes: list[str] = field(default_factory=list)  # SE corrections


@dataclass
class PanelSpec:
    """A single panel in a multi-panel layout."""
    id: str                     # "a", "b", "c"
    title: str                  # "Multi-Directional Supervision"
    position: str               # "top-left", "top-right", etc
    components: list[str]       # short names of contained elements


@dataclass
class PanelLayout:
    """Overall figure layout structure."""
    panel_count: int = 1
    arrangement: str = "single"  # "single" | "horizontal" | "vertical" | "2x2 grid"
    panels: list[PanelSpec] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)  # SE may add layout hints


@dataclass
class StyleSpec:
    """Visual style (colors, typography, what to avoid)."""
    palette: str = "muted pastel"
    typography: str = "sans-serif (Arial/Helvetica)"
    background: str = "pure white"
    avoid: list[str] = field(default_factory=list)  # SE may add: "drop shadows on text", etc.


@dataclass
class RegionFix:
    """RR2's output. NOT included in prompt — applied via PIL composite
    after image generation."""
    bbox: tuple[float, float, float, float]   # normalized [0,1]
    new_crop_path: str                          # path to regenerated crop
    issue: str
    fix_instruction: str
    verified: bool = False                       # whether judge.verify_fix accepted


@dataclass
class SpecEdit:
    """Records a single edit operation for provenance."""
    agent: str                  # "VG" | "planner" | "stylist" | "SE" | "RR2"
    iteration: int              # refinement iteration index
    action: str                 # "set" | "resize" | "demote" | "add_note" | "add_avoid" | "add_region_fix"
    target: str                 # field path: "elements[elem_0].size_hint"
    value: str                  # new value
    reason: str = ""            # for SE: which critique observation drove this


@dataclass
class EvolvingFigureSpec:
    """The single source of truth that all agents edit.

    Agents call methods like spec.add_required_element(), spec.demote_element(),
    spec.add_style_avoid() etc. instead of appending free text to a prompt.
    Final prompt is rendered from this spec — no concatenation conflicts."""

    required_elements: list[VisualElement] = field(default_factory=list)
    layout: PanelLayout = field(default_factory=PanelLayout)
    style: StyleSpec = field(default_factory=StyleSpec)
    region_fixes: list[RegionFix] = field(default_factory=list)
    history: list[SpecEdit] = field(default_factory=list)

    # Reference-image field. When set, the spec render emits a single
    # sentence describing how the chat-completions backend should treat
    # the attached reference image. Empty string = fresh generation
    # (default; preserves V13 behavior).
    #   "preserve_partial" — refer image is the partial figure; fill the
    #                        masked region while preserving the rest.
    #   "use_elements"     — refer image is a layout of visual elements
    #                        the figure must include.
    #   "refine_sketch"    — refer image is a rough spatial structure to
    #                        refine into a polished version.
    refer_image_role: str = ""

    # ── Setters (for VG, planner, stylist — initial population) ──

    def set_elements(self, elements: list[VisualElement], by_agent: str = "VG") -> None:
        self.required_elements = elements
        self.history.append(SpecEdit(
            agent=by_agent, iteration=0, action="set",
            target="required_elements", value=f"{len(elements)} elements",
        ))

    def set_layout(self, layout: PanelLayout, by_agent: str = "planner") -> None:
        self.layout = layout
        self.history.append(SpecEdit(
            agent=by_agent, iteration=0, action="set",
            target="layout", value=layout.arrangement,
        ))

    def set_style(self, style: StyleSpec, by_agent: str = "stylist") -> None:
        self.style = style
        self.history.append(SpecEdit(
            agent=by_agent, iteration=0, action="set",
            target="style", value=style.palette,
        ))

    # ── Editors (for SE — structured corrections only) ──

    def resize_element(self, elem_id: str, size_hint: str, iter: int, reason: str) -> bool:
        for e in self.required_elements:
            if e.id == elem_id:
                e.size_hint = size_hint
                self.history.append(SpecEdit(
                    agent="SE", iteration=iter, action="resize",
                    target=f"elements[{elem_id}].size_hint", value=size_hint, reason=reason,
                ))
                return True
        return False

    def add_element_note(self, elem_id: str, note: str, iter: int, reason: str) -> bool:
        for e in self.required_elements:
            if e.id == elem_id:
                if note not in e.notes:
                    e.notes.append(note)
                self.history.append(SpecEdit(
                    agent="SE", iteration=iter, action="add_note",
                    target=f"elements[{elem_id}].notes", value=note[:80], reason=reason,
                ))
                return True
        return False

    def demote_element(self, elem_id: str, iter: int, reason: str) -> bool:
        """SE may mark an element as optional if critique clearly says it
        crowds the layout. Cannot delete (preserves VG's intent)."""
        for e in self.required_elements:
            if e.id == elem_id:
                e.must_have = False
                self.history.append(SpecEdit(
                    agent="SE", iteration=iter, action="demote",
                    target=f"elements[{elem_id}].must_have", value="false", reason=reason,
                ))
                return True
        return False

    def add_layout_note(self, note: str, iter: int, reason: str) -> None:
        if note not in self.layout.notes:
            self.layout.notes.append(note)
        self.history.append(SpecEdit(
            agent="SE", iteration=iter, action="add_note",
            target="layout.notes", value=note[:80], reason=reason,
        ))

    def add_style_avoid(self, item: str, iter: int, reason: str) -> None:
        if item not in self.style.avoid:
            self.style.avoid.append(item)
        self.history.append(SpecEdit(
            agent="SE", iteration=iter, action="add_avoid",
            target="style.avoid", value=item[:80], reason=reason,
        ))

    # ── RR2 integration ──

    def add_region_fix(self, fix: RegionFix, iter: int) -> None:
        self.region_fixes.append(fix)
        self.history.append(SpecEdit(
            agent="RR2", iteration=iter, action="add_region_fix",
            target="region_fixes", value=f"{fix.bbox} {fix.issue[:40]}",
        ))

    # ── Serialization ──

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=str)


# ─────────────── prompt rendering (single source of truth) ───────────────

# This is the ONLY place that builds the final image-gen prompt. It pulls
# everything from `spec`, so there is no opportunity for two agents to write
# contradictory instructions into different prompt blocks.
#
# Renders as a SINGLE MANDATORY block so the spec architecture and the
# compact image-gen prompt coexist without prompt-internal conflicts.


_POSTER_PREAMBLE = (
    "[ROLE: research poster — NOT a single academic figure]\n"
    "Render the output as a print-quality publication-format research "
    "POSTER. Target visual fidelity: high-DPI scientific poster, sharp "
    "vector-clean typography, dense information layout, the kind you "
    "would actually pin to a CVPR / NeurIPS / ICCV poster board.\n"
    "  - TOP BANNER STRIP (full width, ~10-15% of canvas height):\n"
    "      *** THE PAPER TITLE GOES HERE — render at THE LARGEST TYPOGRAPHY\n"
    "      OF THE WHOLE POSTER, prominently centered, BOLD, razor-crisp\n"
    "      anti-aliased glyphs (NOT blurry, NOT pixelated). The paper\n"
    "      title is the single most visible visual element — readable\n"
    "      from across the room. Section headers in the body must be at\n"
    "      most HALF the size of the title.\n"
    "      *** Below the title in the banner: author list + affiliations.\n"
    "      *** LOGOS — REQUIRED, but use STYLIZED PLACEHOLDER MARKS, not\n"
    "      attempts at real brand logos: at the banner's left corner an\n"
    "      abstract university crest (stylized shield / interlocking\n"
    "      circles / geometric flag) paired with a 2-4 letter mock\n"
    "      institution acronym in slab type; at the banner's right corner\n"
    "      a stylized conference badge (clean geometric mark + 4-letter\n"
    "      venue acronym like 'CVPR' or 'NeurIPS' + year). The crest and\n"
    "      badge must be visually present and crisp — empty corners read\n"
    "      as 'not a real poster'. Do NOT attempt to render any real\n"
    "      university wordmark or Microsoft/Google/Meta logo glyphs.\n"
    "      *** DO NOT promote a section header (Motivation / Method /\n"
    "      Results / etc) into the title slot — those are body section\n"
    "      labels, not the paper title.\n"
    "  - 3-4 column body grid filling the rest of the canvas. Typical "
    "section ordering: Motivation / Background → Approach / Method → "
    "Experiments / Results → Conclusion / Project page.\n"
    "  - Each column / panel has its own colored header bar with the "
    "section name (smaller than the paper title). Use complementary "
    "section colors (e.g. teal-blue + orange + purple) consistent across "
    "panels.\n"
    "  - Large readable typography (legible from 2 metres); generous "
    "whitespace between panels; visible dividers. All body text must be "
    "sharp and legible — NO blurred lorem-ipsum-looking smudges of "
    "text-shaped pixels.\n"
    "  - Include actual result figures, ablation snapshots, comparison "
    "tables INSIDE the panels — NOT just text descriptions. Figure axes "
    "and table cells must be crisp.\n"
    "  - Bottom-right corner: QR code (clean black-and-white square grid) "
    "or project URL is a strong signal of the poster format.\n"
    "  - Aspect ratio: WIDE landscape (~16:9 to 4:3); the canvas should "
    "look like a printed conference poster, not a single figure. Output "
    "should feel high-resolution: hairline borders, anti-aliased curves, "
    "no soft / hazy / dreamy filter — sharp print, not artistic blur.\n"
    "Do NOT collapse this to a single multi-panel academic figure with "
    "panels (a)/(b)/(c). The output IS a poster."
)


_INFOGRAPHIC_PREAMBLE = (
    "[ROLE: magazine / blog INFOGRAPHIC — NOT an academic figure]\n"
    "Render the output as a casual explainer-style INFOGRAPHIC:\n"
    "  - Reads like a HuggingFace blog post / Distill article, NOT a "
    "paper figure. Audience: someone curious but not deep in the field.\n"
    "  - Major step-markers as ROUND coloured callouts (orange / red "
    "circles) numbered 1, 2, 3 ... placed adjacent to the corresponding "
    "step blocks.\n"
    "  - Expressive, attention-grabbing arrows (curved, red, thicker "
    "strokes) — NOT the thin black straight arrows of academic figures.\n"
    "  - Large friendly labels (often handwritten-feeling sans serif), "
    "with descriptive phrases instead of equations or jargon.\n"
    "  - Use illustrative icons / cartoonish drawings to depict abstract "
    "concepts; soft gradients; bold-but-flat color blocks.\n"
    "  - Reading flow can be horizontal (left-to-right wide canvas) OR "
    "vertical (top-to-bottom tall canvas), depending on what the steps "
    "describe.\n"
    "Do NOT use academic conventions: no clean black-on-white method-"
    "pipeline figure, no panel labels (a)/(b)/(c)."
)


def _role_preamble(role: str) -> str:
    """Role-conditional structural preamble.

    Hardcoded blocks for "poster" / "infographic"; "" and "academic" emit
    no preamble (the default narrative handles those). Any other role
    string falls through to a generic "use this role descriptor verbatim"
    block — supports free-form intents like "1980s retro infographic" or
    "patent diagram with numbered callouts" without requiring a code
    change per role.
    """
    role_norm = (role or "academic").lower()
    if role_norm == "poster":
        return _POSTER_PREAMBLE
    if role_norm == "infographic":
        return _INFOGRAPHIC_PREAMBLE
    if role_norm in ("", "academic"):
        return ""
    # Free-form role descriptor passed straight to the model verbatim.
    return (
        f"[ROLE: {role}]\n"
        f"Render the output to match this role descriptor exactly: "
        f"the format, medium, visual conventions, and overall style "
        f"of a \"{role}\" — NOT a clean academic paper figure (unless the "
        f"role itself is academic).\n"
        f"  - Read the role descriptor literally; if it implies a specific "
        f"format (poster / infographic / sketch / cover art / patent / "
        f"slide / mock-up / etc.), produce that format.\n"
        f"  - If the role implies a specific aesthetic ('1980s retro', "
        f"'hand-drawn', 'cyberpunk', 'children's book', 'minimalist', "
        f"'photorealistic', etc.), use that aesthetic — palette, "
        f"typography, line weight, decoration all match.\n"
        f"  - Do NOT default to academic conventions. The role is what "
        f"the user asked for; honor it."
    )


# ── Edit-mode instruction builder ────────────────────────────────────────

def build_edit_instruction(task: str, style: str, description: str,
                              spec: Optional["EvolvingFigureSpec"] = None) -> str:
    """Produce the leading instruction the /v1/images/edits endpoint sees
    for a CraftBench edit-task sample.

    Branches on (task, style). Poster inpaint emits a more directive fill
    instruction (rich multi-panel content for ~25%-of-canvas regions); all
    inpaint wording uses "MAIN content preserved with minor cohesion
    adjustments acceptable" rather than pixel-faithful preservation, matching
    how a real designer fills in an inpaint region.

    If `spec` is provided, evolved layout notes + style-avoid items from
    prior critique are appended so structured spec edits actually reach
    the generator.
    """
    task = (task or "").lower()
    style = (style or "").lower()

    if task == "inpaint":
        if style == "poster":
            head = (
                "POSTER INPAINT EDIT — The attached image is a conference "
                "research POSTER. ONE full column (≈25% of the canvas) is "
                "blank/white. Fill that column with multiple sub-sections — "
                "typically Experiments / Results / Ablation / Conclusion / "
                "Project page — each with:\n"
                "  - a colored section header bar matching the styling of "
                "the existing columns (same banner color palette, same fonts);\n"
                "  - body content: actual result figures, comparison images, "
                "ablation panels, tables, or QR / project URL stub — NOT just "
                "a blank panel with a title;\n"
                "  - typography legible at 2 metres, generous whitespace.\n"
                "Keep the MAIN content of the rest of the poster (banner / "
                "title / authors / left columns) preserved — minor adjustments "
                "at column boundaries for visual cohesion are acceptable, but "
                "do NOT redraw or restyle the existing columns."
            )
        else:
            head = (
                "INPAINT EDIT — The attached image has a blank / white region. "
                "Fill ONLY that region with content matching the description; "
                "keep the MAIN content of the rest of the image (layout, "
                "components, labels, colors) preserved. Minor adjustments at "
                "the boundary for visual cohesion are acceptable, but do NOT "
                "redraw or restyle the un-masked region."
            )
    elif task == "keyelems":
        # The provided elements are MUST-INCLUDE inputs, NOT a hard limit
        # — phrasing must not be misread as "use ONLY these elements".
        head = (
            "ELEMENT-BUILD EDIT — The attached image shows specific visual "
            "elements (icons, charts, photos, illustrations) the user wants "
            "INCLUDED in the final figure, at their intended positions on a "
            "blank canvas.\n"
            "  - You MUST include each of those provided elements in the "
            "output, at roughly the positions shown.\n"
            "  - You ARE FREE to ADD any additional content the figure needs "
            "to be coherent and publication-quality: panels, arrows, "
            "section labels, supplementary diagrams, captions, headers, body "
            "text, MORE icons / charts, decoration. Add what the figure "
            "design calls for.\n"
            "  - The provided elements are MUST-INCLUDE inputs, NOT a hard "
            "limit on what may appear. Add freely."
        )
    elif task == "sketch":
        # Sketch is a LAYOUT REFERENCE only; model must REBUILD every
        # visual element in publication style, NOT copy or lightly retouch.
        head = (
            "SKETCH-REFINE EDIT — The attached image is a ROUGH SKETCH the "
            "user provided as a LAYOUT REFERENCE. The sketch may be "
            "hand-drawn, an SVG draft, an AI rough, or a low-quality draft. "
            "It conveys ONLY: spatial layout / panel arrangement, which "
            "boxes / components / labels go where, how arrows connect them, "
            "the reading order.\n"
            "It is NOT the target output style. It is NOT the output. You "
            "MUST RE-IMAGINE every visual element from scratch in clean "
            "publication style — real polished icons, real labels, clean "
            "typography, proper colors. The sketch's lines, scribbles, "
            "hand-drawn marks, informal shapes, draft-quality artifacts — "
            "DO NOT preserve ANY of those.\n"
            "DO NOT copy the sketch. DO NOT lightly retouch it. DO NOT "
            "trace its lines. The output should look as if a designer was "
            "given the sketch as a layout brief and produced a polished "
            "figure from scratch.\n"
            "Preserve ONLY: panel positions, arrow connections, reading "
            "order. Rebuild EVERYTHING ELSE in publication style."
        )
    else:
        head = ("EDIT — apply this instruction to the attached image, "
                "preserving everything else.")

    out = f"{head}\n\n{description}"

    # Append evolved layout notes + style-avoids if a spec is given so
    # structured edits from skill_evolver.evolve_spec() reach the generator.
    # For keyelems, style.avoid is suppressed: negative instructions cause
    # the generator to drop user-provided elements that look like avoided
    # things, violating the "MUST INCLUDE these elements" mandate.
    if spec is not None:
        try:
            evolved_notes = [n.text for n in getattr(spec.layout, "notes", []) if getattr(n, "text", "")]
            evolved_avoids = list(getattr(spec.style, "avoid", []) or [])
        except Exception:
            evolved_notes, evolved_avoids = [], []

        if task == "keyelems":
            evolved_avoids = []

        if evolved_notes:
            block = "\n".join(f"  - {t}" for t in evolved_notes[:8])
            out += f"\n\nEvolved layout guidance from prior critique:\n{block}"
        if evolved_avoids:
            out += "\n\nAvoid: " + ", ".join(evolved_avoids[:8])

    return out


# Reference-image rendering: refer hints are placed at the LEADING edge of the
# rendered prompt (not appended at the tail). Empirically, prepending the refer hint
# shows that prepending "respect
# what's visible in the attached reference image" before the description
# anchors the chat-completions model on the refer; appending it after a
# 1500-char figure narrative (Stage C iter 1's mistake) lets the model
# anchor on the description instead and regenerate from scratch.
#
# Each hint is a single short paragraph. Extending into per-task multi-
# sentence prose would be "task-by-task prompt hacking" and violate the
# architecture-driven constraint.
_REFER_IMAGE_HINTS = {
    "preserve_partial": (
        "[Reference image attached — the user's pre-existing PARTIAL figure. "
        "ONE region has been left blank/masked. The final output must "
        "preserve every pixel of the unmasked region exactly as shown, and "
        "only fill the blank/masked region with content matching the brief "
        "below.]"
    ),
    "use_elements": (
        "[Reference image attached — specific visual elements (icons, "
        "charts, photos, illustrations) the user requires in the final "
        "figure. Use those exact elements at roughly their shown "
        "positions; add the surrounding structure (panel borders, arrows, "
        "text labels) needed to make the figure coherent.]"
    ),
    "refine_sketch": (
        "[Reference image attached — the user's rough sketch of the "
        "intended figure structure. Preserve the spatial layout (where "
        "boxes are, how they connect, reading order, proportions). Refine "
        "the details: clean shapes, real labels in place of placeholder "
        "text, polished typography. The output should be visibly the same "
        "composition as the sketch but in publication quality, NOT a "
        "pixel-perfect retracing of the sketch.]"
    ),
    # Generic fallback for unknown / arbitrary edit tasks with a refer
    # image whose role (inpaint mask vs. element layout vs. structural
    # sketch) cannot be classified. Caller may override `refer_image_role`
    # directly when the precise role is known.
    "generic_edit": (
        "[Reference image attached — the user wants to apply an edit to "
        "this image. Preserve the visual structure that the description "
        "does NOT ask to change (panel arrangement, key components, "
        "overall composition); apply only the edits the description "
        "specifies. The output should be coherent and publication-quality.]"
    ),
}


def render_image_gen_prompt(
    spec: EvolvingFigureSpec, paper_text: str, caption: str,
    intent_preamble: str = "",
    role: str = "academic",
) -> str:
    """Render the figure spec into a single image-generation prompt.

    Integrated narrative format: reads like a senior visual designer
    briefing a junior artist — caption + visual elements woven into
    placement description + style as positive guidance + iteration
    learnings inline. The `role` parameter injects a role-conditional
    structural preamble; role="poster" emits a multi-column poster brief,
    role="infographic" emits a casual explainer brief, role="academic"
    (default) emits no preamble. Used by the chat-completions multimodal
    path; the /v1/images/edits path uses `build_edit_instruction` instead.
    """
    role_block = _role_preamble(role)
    # When refer_image_role is set, the refer label leads the prompt
    # (before role / intent / caption) so the multimodal model attends
    # to the reference image before reading the figure narrative.
    refer_lead = (
        _REFER_IMAGE_HINTS.get(spec.refer_image_role, "")
        if spec.refer_image_role else ""
    )

    if not spec.required_elements:
        # No spec elements — still apply role/intent preamble to caption
        head = []
        if refer_lead:
            head.append(refer_lead)
        if role_block:
            head.append(role_block)
        if intent_preamble:
            head.append(intent_preamble)
        head.append(caption)
        return "\n\n".join(head)

    must = [e for e in spec.required_elements if e.must_have]
    nice = [e for e in spec.required_elements if not e.must_have]

    parts = []

    # Refer-image label leads everything else when refer_image_role is
    # set (unified-mode anchoring). Empty when refer_image_role is "".
    if refer_lead:
        parts.append(refer_lead)
        parts.append("")

    # Role-conditional preamble goes FIRST because the model attends
    # most to leading content. Empty string for academic role.
    if role_block:
        parts.append(role_block)
        parts.append("")

    # Intent preamble (role + style + density) parametrises the template
    # across academic / poster / infographic / etc. without branching.
    if intent_preamble:
        parts.append(intent_preamble)
        parts.append("")

    # Opening: caption establishes WHAT the figure is
    parts.append(f"Figure brief: {caption[:300]}")

    # Visual elements woven into a description, not a list
    if must:
        elements_phrases = []
        for e in must:
            # Skip article — element names like "Hippocampus brain icon"
            # already read naturally. "a brain icon" is OK; "a thumbnails"
            # is awkward; just use the name verbatim.
            phrase = f"{e.name} ({e.realization})"
            if e.placement and e.placement != "as visual element":
                phrase += f" placed {e.placement}"
            if e.size_hint:
                phrase += f", sized {e.size_hint}"
            if e.notes:
                phrase += " — " + "; ".join(e.notes)
            elements_phrases.append(phrase)

        if len(elements_phrases) == 1:
            parts.append(f"\nKey visual element to render concretely: {elements_phrases[0]}.")
        else:
            parts.append(
                f"\nKey visual elements to render concretely (as actual icons / "
                f"drawings / photos, NOT as text labels in boxes):"
            )
            for p in elements_phrases:
                parts.append(f"  • {p}")

    # Optional elements — soft inclusion language
    if nice:
        nice_names = ", ".join(e.name.lower() for e in nice)
        parts.append(
            f"\nIf space allows you may include: {nice_names}. Skip them if "
            f"they would crowd the layout."
        )

    # Style as POSITIVE guidance (not "avoid X")
    style_phrases = []
    if spec.style.palette:
        style_phrases.append(f"palette: {spec.style.palette}")
    if spec.style.typography:
        style_phrases.append(f"typography: {spec.style.typography}")
    if spec.style.background:
        style_phrases.append(f"background: {spec.style.background}")

    if style_phrases:
        parts.append(f"\nStyle: {'; '.join(style_phrases)}.")

    # AVOID items — only if SE explicitly added them, phrase positively
    if spec.style.avoid:
        # Convert "drop shadows on text" → "without drop shadows on text"
        avoid_clause = ", ".join(spec.style.avoid)
        parts.append(f"For polish: avoid {avoid_clause}.")

    # Layout notes — inline with placement context
    if spec.layout.notes:
        layout_clause = "; ".join(spec.layout.notes)
        parts.append(f"\nLayout note: {layout_clause}.")

    # Anti-hallucination tail: ground every label, variable, and module
    # name in the caption. A short clause only — enumerating specific
    # examples primes the generator to invent similar ones.
    parts.append(
        "\nGround every label, variable name, layer count, and module name "
        "in the caption above. Do not add specific architecture variants, "
        "named techniques, or numerical hyperparameters that are not stated "
        "in the caption — when unsure, use a generic descriptive label."
    )

    return "\n".join(parts)


def render_image_gen_prompt_verbose(
    spec: EvolvingFigureSpec, paper_text: str, caption: str,
) -> str:
    """Multi-section render alternative (default off).

    Use only when you need spec to be visibly structured (e.g. inspecting
    extracted layout / required elements). For production prompt
    construction, use `render_image_gen_prompt()` above — it is more
    compact and scores higher on referenced VLM judges.
    """
    parts = []
    parts.append(f"## FIGURE TO GENERATE\n{caption[:300]}")

    if spec.layout.panel_count > 0:
        parts.append("\n## LAYOUT")
        parts.append(f"- arrangement: {spec.layout.arrangement}")
        if spec.layout.panel_count > 1:
            parts.append(f"- {spec.layout.panel_count} panels:")
            for p in spec.layout.panels:
                parts.append(f"  - panel ({p.id}) at {p.position}: {p.title}")
        for n in spec.layout.notes:
            parts.append(f"- {n}")

    if spec.required_elements:
        must = [e for e in spec.required_elements if e.must_have]
        nice = [e for e in spec.required_elements if not e.must_have]
        if must:
            parts.append("\n## REQUIRED VISUAL ELEMENTS (must all appear)")
            for e in must:
                line = f"- {e.name}: render as {e.realization}, placed at {e.placement}"
                if e.size_hint:
                    line += f"; size: {e.size_hint}"
                parts.append(line)
                for note in e.notes:
                    parts.append(f"  ◦ note: {note}")
        if nice:
            parts.append("\n## OPTIONAL VISUAL ELEMENTS (include if space allows)")
            for e in nice:
                parts.append(f"- {e.name}: {e.realization}")

    parts.append("\n## STYLE")
    parts.append(f"- palette: {spec.style.palette}")
    parts.append(f"- typography: {spec.style.typography}")
    parts.append(f"- background: {spec.style.background}")
    if spec.style.avoid:
        parts.append("- AVOID:")
        for a in spec.style.avoid:
            parts.append(f"  - {a}")

    parts.append(f"\n## PAPER CONTEXT (for grounding)\n{paper_text[:3500]}")

    return "\n".join(parts)
