"""VisualMetaphorTranslator: translate abstract / math / role-noun captions
into concrete visual narratives anchored to a named reference aesthetic.

The translator addresses three failure modes:
  (A) caller-specified non-default aesthetic silently overridden by
      PromptRefiner's academic boilerplate
  (B) captions with abstract relational nouns or LaTeX/math fail to render
  (C) PromptRefiner prompt blowup ≥ 6 KB causing image-gen 502s

Conservative gate (§4.3 of proposal): the translator runs only when the
default `NeurIPS academic + short plain caption` case does NOT hold, to
avoid regressing the academic-figure baseline.

Single LLM call (Opus / Gemini-pro depending on planner_model) returns a
visual_caption + structural_constraints + aesthetic_anchor. Downstream
PromptRefiner consumes the translated form.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from crafter.shared.model_router import ModelRouter

logger = logging.getLogger(__name__)


@dataclass
class VisualTranslation:
    """Output of VisualMetaphorTranslator.translate()."""

    visual_caption: str = ""        # 800-1500 chars — visual narrative
    structural_constraints: str = ""  # ≤400 chars — layout/color/typography
    aesthetic_anchor: str = "NeurIPS academic"
    visual_metaphors: dict = field(default_factory=dict)


_AESTHETIC_ANCHORS = {
    # ── academic / technical baselines ─────────────────────────────────
    "academic paper figure": (
        "publication-quality clean technical figure, restrained 3-5 color "
        "palette, plain sans-serif typography, crisp lines, rounded boxes "
        "with text labels, conventional arrows, generous whitespace"
    ),
    "blog-style explainer": (
        "casual editorial illustration, warm friendly palette, hand-printed "
        "labels, expressive curved arrows, illustrative spot icons, "
        "approachable cartoon-ish characters where appropriate"
    ),
    "Excalidraw rough": (
        "hand-drawn lines, wavy strokes, casual layout, sketchy fills"
    ),
    "IEEE technical": (
        "rigid grid, blue/grey palette, dense labels, formal"
    ),
    "corporate keynote": (
        "gradient fills, modern minimalist, large titles"
    ),
    # ── conference poster ─────────────────────────────────────────────
    "conference research poster": (
        "print-quality high-DPI scientific poster; wide landscape canvas; "
        "prominent banner with razor-sharp title typography, author list, "
        "and STYLIZED PLACEHOLDER LOGOS (abstract university crest in left "
        "corner + abstract conference badge in right corner — geometric "
        "shapes plus mock 2-4 letter acronyms, not real brand marks); "
        "3-4 vertical columns of section panels with colored header bars; "
        "rich result figures, ablation snapshots, comparison tables "
        "embedded inside panels; crisp anti-aliased text everywhere; "
        "hairline borders; bottom-right QR code; no soft/dreamy filter, "
        "this is a printed conference poster"
    ),
    # ── illustrative / stylised options ───────────────────────────────
    "anime cute illustration": (
        "soft pastels, big-eyed chibi characters, sticker shading, gentle "
        "rounded outlines, manga panel composition"
    ),
    "Studio Ghibli watercolor": (
        "warm illumination, hand-painted backdrops, gentle character "
        "expressions, watercolor textures, soft natural palette"
    ),
    "Pixar 3D cinematic": (
        "3D-styled rendering with rim lighting, expressive stylised "
        "characters, polished cinematic shading, depth-of-field"
    ),
    "children's book illustration": (
        "thick crayon-like outlines, flat saturated primary colours, "
        "whimsical hand-drawn objects, paper-texture backgrounds"
    ),
    "retro 1980s vaporwave": (
        "neon palette, grid horizon, CRT scanlines, magenta/cyan gradients, "
        "synthwave aesthetic"
    ),
    "isometric tech infographic": (
        "isometric 30°/60° geometry, flat panels with subtle gradients, "
        "vibrant tech palette, clean grid composition"
    ),
}


_BASE_RULES = """\
You are a visual-design router and translator. Given paper context,
caption, instruction, and image-type signals, you decide:
  (A) the style family (ANCHOR_NAME)
  (B) the task protocol that fits the actual input
  (C) the enrichment level — how much new visual prose to inject
and emit a single visual narrative an image-gen model can render
directly. This is a HARNESS: declared role / task are SUGGESTIONS
that you may override when context disagrees; if either is missing
you INFER from paper / instruction.

OUTPUT FORMAT (strict)
Emit ONLY:

  ANCHOR_NAME: <short name, ≤6 words>
  ANCHOR_DESCRIPTOR: <12-30 words: palette + line style + texture + composition feel>
  ---
  <visual narrative, plain prose, length per the enrichment level you chose>

No JSON, no ``` fences, no preamble before ANCHOR_NAME.

UNIVERSAL VISUAL RULES (always apply, regardless of family / protocol)
1. **Faithfulness first.** Every paper component appears with its EXACT
   name. For each: shape, fill color, label, position relative to
   neighbors. The OVERALL ARCHITECTURE in the narrative must mirror
   what the paper / input instruction describes. Faithfulness >
   cleanness > aesthetic flourish.
2. **Metaphors are role-conditional, never default.** Translate
   abstract / relational nouns (verifier, judge, memory, embedding, …)
   into concrete visual objects (spectacled owl, traffic light,
   journal-page stack, grid of colored dots) ONLY when the chosen
   style family invites illustrative metaphors. For academic paper
   figures the convention is text-labeled boxes with arrows — do NOT
   inject mascots unless the paper itself uses them.
3. Reject LaTeX / citations. Replace `$\\mathcal{S}$` with a short
   label (e.g. `S`); replace formulas with named labels (Loss L,
   Attention(Q,K,V)).
4. **Front-load STRUCTURAL CONSTRAINTS** in the first lines (unless
   enrichment = MINIMAL allows skipping):
   - Layout direction (LTR / TTB / radial / cyclic / grid)
   - 5-7 named color palette
   - Typography (sans-serif clean / hand-printed / poster banner)
   - Arrow style + meaning

FORBIDDEN
- Figure titles, figure numbers, paper titles inside the image
- Pixel dimensions, CSS values, coordinates
- Sentences inside boxes — 5-10 word labels max
- Dark / black backgrounds
- Crossing or chaotic arrows
"""


# ── Task-aware protocol clauses ──────────────────────────────────────────
# Each clause tells the translator what kind of output the downstream
# image-gen call expects for this task. Selected by router; unknown
# tasks fall back to a generic clause that defers to the input
# instruction.

_TASK_PROTOCOLS: dict[str, str] = {
    "t2i": """\
TASK PROTOCOL (t2i — text-to-image, no reference image)
Generate the figure from scratch using the paper context + input
instruction. Goal: ADD style, beauty, and clarity while keeping the
architectural components and their connections faithful to what the
paper describes. Emit a FULL visual narrative (rich, but every
component traceable to the paper). The input instruction (if any) is
authoritative — it specifies the kind of image, style, and any
architectural requirements.
""",
    "method_pipeline": """\
TASK PROTOCOL (method_pipeline — academic method-pipeline figure)
Same as t2i: generate the figure from scratch faithful to the paper's
method pipeline. Emit a clean academic-style visual narrative with the
exact components and dataflow described.
""",
    "inpaint": """\
TASK PROTOCOL (inpaint — fill a blanked region in a partial figure)
A reference image is provided whose layout, palette, and style are
ALREADY FIXED. ONE region of that image is blanked / left to fill.
Describe ONLY THE FILL — the content that should occupy the blank
region — using the SAME palette / line weight / typography / shading
as the surrounding existing parts. Do NOT redesign the rest of the
image. The narrative should be BRIEF (40-90 lines) and focused on the
fill region, not a full figure. The input instruction names the fill
content; honor it precisely.
""",
    "keyelems": """\
TASK PROTOCOL (keyelems — compose using provided element images)
The reference image contains user-provided KEY ELEMENTS (icons,
sub-figures, charts, photos) positioned on a canvas. You MUST
incorporate each provided element VERBATIM — no recolor, no restyle,
no replacement. The narrative describes the LAYOUT around the elements
(arrows, section labels, supporting text, additional connective
graphics) faithful to the paper's described context. Refer to provided
elements by their visible content (e.g. "the architecture diagram
already on the canvas at top-left"). Add freely whatever connective
graphics the figure needs.
""",
    "sketch": """\
TASK PROTOCOL (sketch — refine a rough sketch into a polished figure)
The reference image is a ROUGH SKETCH that conveys arch-level /
semantic-level intent: panel layout, components, where arrows go,
reading order. You MUST PRESERVE the sketch's arrangement
(positions, components, connections, reading order). You MUST REBUILD
every visual element in polished publication style — clean icons,
real labels, proper typography, real colors. The sketch's rough
strokes are NOT the target style. Do NOT trace, copy, or lightly
retouch the sketch — re-imagine from scratch using the sketch as
layout brief only.
""",
}

_TASK_FALLBACK_WITH_REF: str = """\
TASK PROTOCOL (generic with reference image)
A reference image is provided. The input instruction tells you whether
to treat it as a STYLE REFERENCE — mimic its look-and-feel for a
freshly composed figure — or as a DRAFT TO POLISH — preserve its
layout but upgrade its execution. Follow the instruction precisely;
when ambiguous, treat as a style reference and produce a fresh,
publication-quality figure inspired by the reference's aesthetic.
"""

_TASK_FALLBACK_NO_REF: str = """\
TASK PROTOCOL (generic text-only)
Generate a publication-quality figure from the paper context + input
instruction. Use the visual narrative to ADD style and clarity while
staying faithful to the architectural components described. The input
instruction (if any) is the authoritative spec.
"""


# ── Role-aware visual guides ─────────────────────────────────────────────
# Each guide tells the translator what visual style is appropriate for
# this role. Selected by router; unknown roles fall back to a neutral
# guide that defers to the paper / instruction.

_ROLE_VISUAL_GUIDES: dict[str, str] = {
    "academic": """\
ROLE STYLE (academic publication figure)
Clean technical figure as in a peer-reviewed paper. Boxes-and-arrows
diagrams, restrained 3-5 color palette, sans-serif typography, crisp
lines, conventional notation. AVOID metaphorical mascots, whimsical
illustration, or decorative flourishes UNLESS the paper itself uses
them. Faithfulness to the paper's architecture overrides aesthetic
flourish. Where the paper introduces a relational noun, prefer a
text-labeled box ("Verifier") over an illustrative mascot
("spectacled owl") — except when role context invites the latter.
""",
    "poster": """\
ROLE STYLE (conference research poster)
Print-quality high-DPI scientific poster, NOT a single figure. Wide
landscape canvas. Top banner with paper title (largest type), author
list, and STYLIZED PLACEHOLDER LOGOS (abstract university crest left,
abstract venue badge right — geometric shapes + mock 2-4 letter
acronyms, never real brand wordmarks). 3-4 vertical column panels
with colored header bars. Each panel contains body content INCLUDING
embedded result figures / tables / ablation panels — not just text.
Razor-crisp anti-aliased typography, hairline borders, optional
bottom-right QR code.
""",
    "infographic": """\
ROLE STYLE (academic figure embedded in a tutorial blog post)

IMPORTANT CONTEXT — the samples this benchmark labels "infographic"
are NOT magazine-style infographics. They are clean academic paper
figures (encoder-decoder diagrams, RL architectures, transformer
blocks, etc.) that happen to be embedded in tutorial blog posts
(Lilian Weng's blog, Christopher Olah's blog, Distill articles).
The captions explicitly cite their source ("Image source: Bahdanau
et al., 2015", "Vincent et al., 2010", etc.) — these are the
ORIGINAL paper figures, NOT redrawn cartoon versions.

The visual reference is therefore essentially ACADEMIC:
  - Clean block diagrams or technical graphs
  - Restrained palette (mostly neutral with at most 1-2 accent
    colors that are clearly tied to functional grouping, not
    decorative)
  - Sans-serif typography (Inter / Helvetica feel — NOT hand-
    printed, NOT chalk-script, NOT Comic Sans)
  - Thin elegant lines, generous whitespace
  - Faithful to the paper's described components and data flow

The ONLY differences from the "academic" role:
  - Slightly softer corners on boxes (rounded vs square)
  - One inline icon per major component is acceptable if a clear
    metaphor exists in the paper's text (e.g. a tiny brain glyph
    next to "Agent" only if the paper uses the agent metaphor)
  - Curved arrows where flow benefits readability

FORBIDDEN — these were observed failure modes that lose to Human:
  - Scattered background dots, sparkles, decorative bokeh
  - Corporate clip-art collages (untethered stock icons)
  - Mascot characters, cartoon scenes, decorative props
  - Fully saturated warm palette across all elements
  - Hand-printed / kid-style typography
  - "Blog-style infographic" / "Quanta magazine" / Corporate-Blog
    visual register — the actual Human reference is an academic
    paper figure, NOT a magazine illustration

**Coverage rule:** depict ALL of the paper's key components,
modules, and data-flow connections faithfully. Restraint over
flourish. The output should look like it could appear in the
original paper unchanged — NOT like a magazine illustration of it.
""",
}

_ROLE_FALLBACK_GUIDE: str = """\
ROLE STYLE (general, role unspecified)
Produce a clean, informative figure appropriate to the paper's
domain. Choose a palette and typography that match the paper's
context. Prioritize legibility and faithfulness; default to academic
publication conventions unless the input instruction asks for a
different style.
"""


_DECISION_FRAMEWORK = """\
DECISION FRAMEWORK (apply per call)

(A) STYLE FAMILY — your ANCHOR_NAME
    Priority:
      1. If `aesthetic_intent` is given in the user message, invent
         a fresh anchor name + descriptor honoring it.
      2. Else if declared `role` matches one of the styles listed
         below, use that style.
      3. Else INFER from paper context + instruction + caption:
         - paper looks like an academic methodology figure → academic
           paper figure
         - instruction or caption asks for poster / banner / multi-
           column layout → conference research poster
         - instruction or caption mentions blog / explainer / casual
           audience → blog-style explainer
         - otherwise default to "academic paper figure"

(B) TASK PROTOCOL — drives how a reference image (if any) is handled
    Priority:
      1. If declared `task` matches a known protocol, use it.
      2. Else: pick generic-with-ref if a reference image is
         attached, generic-no-ref otherwise.
      3. You MAY override the declared protocol when input structure
         disagrees (e.g. declared t2i but a reference image is
         attached → switch to a with-ref protocol).

(C) ENRICHMENT LEVEL — controls narrative depth + flourish
    - HEAVY (80-200 lines): aesthetic_intent set, role=poster, or
      instruction explicitly demands a stylistic transformation
      (e.g. "make it cute / anime / ghibli"). Output a full
      narrative with palette, typography, composition, metaphor
      mapping where appropriate.
    - MEDIUM (50-100 lines): default for role=infographic and for
      academic samples with sparse or abstract captions. Output a
      balanced narrative with structural constraints + per-
      component description; restrained palette flourish.
    - MINIMAL (20-50 lines): the input is already a complete,
      concrete spec (rich caption / detailed instruction) and the
      role is academic. Adding heavy enrichment risks hurting
      faithfulness. Output a TERSE restatement that preserves the
      original spec where possible, adding only the front-loaded
      structural constraints + palette naming. Do NOT invent visual
      metaphors or aesthetic flourishes the paper does not mention.

    Bias toward MINIMAL for clean academic with detailed captions.
    Bias toward MEDIUM for role=infographic (HEAVY risks Corporate-
    Blog-clipart drift on Distill-style blog samples). Bias toward
    HEAVY only when aesthetic_intent is explicit or role=poster.
"""


def _format_section(name: str, body: str) -> str:
    return f"[{name}]\n{body.strip()}"


def _available_protocols_block() -> str:
    pieces = [_format_section(name, body)
              for name, body in _TASK_PROTOCOLS.items()]
    pieces.append(_format_section("generic-with-ref", _TASK_FALLBACK_WITH_REF))
    pieces.append(_format_section("generic-no-ref", _TASK_FALLBACK_NO_REF))
    return "AVAILABLE TASK PROTOCOLS\n\n" + "\n\n".join(pieces)


def _available_styles_block() -> str:
    pieces = [_format_section(name, body)
              for name, body in _ROLE_VISUAL_GUIDES.items()]
    pieces.append(_format_section("general / unspecified", _ROLE_FALLBACK_GUIDE))
    # Include the broader anchor catalog as additional suggestions
    catalog = "\n".join(
        f"  - {name} — {desc}"
        for name, desc in _AESTHETIC_ANCHORS.items()
        if name not in ("academic paper figure", "conference research poster",
                         "blog-style explainer")
    )
    pieces.append(
        "[other style anchors] (you may invent anchors beyond these)\n"
        + catalog
    )
    return "AVAILABLE ROLE STYLE GUIDES\n\n" + "\n\n".join(pieces)


def _build_system_prompt(task: str = "", role: str = "",
                          has_refer: bool = False) -> str:
    """Compose the harness system prompt.

    The full menu of task protocols + role style guides + decision
    framework is included so the LLM can route, override, or
    infer based on the actual input. The function arguments are
    retained for back-compat; they no longer pre-narrow the prompt.
    """
    return (
        f"{_BASE_RULES}\n\n"
        f"{_available_protocols_block()}\n\n"
        f"{_available_styles_block()}\n\n"
        f"{_DECISION_FRAMEWORK}"
    )


_LATEX_PAT = re.compile(r"\\mathcal\{|\\mathbb\{|\\textbf\{|\$[^$]+\$|\\cite\{|\\ref\{")
_RELATIONAL_NOUNS = (
    "verifier", "critic", "judge", "refiner", "generator", "planner",
    "specification", "spec ", "embedding", "encoder", "decoder", "policy",
    "agent", "controller", "evaluator", "regressor", "classifier",
    "discriminator", "scheduler", "router", "orchestrator", "harness",
    "loop", "cycle", "iteration", "feedback", "convergence",
)


def _looks_abstract(caption: str) -> bool:
    """Return True if the caption likely needs translation (math, abstract
    nouns, or long verbose phrasing)."""
    if not caption:
        return False
    if _LATEX_PAT.search(caption):
        return True
    lc = caption.lower()
    if sum(noun in lc for noun in _RELATIONAL_NOUNS) >= 2:
        return True
    if len(caption) >= 300:
        return True
    return False


def should_translate(craft_input) -> bool:
    """Universal gate — the LLM router (inside translate()) decides
    per call whether to enrich heavily, moderately, or minimally based
    on input characteristics. Most inputs go through VMT; two narrow
    cases skip:

      1. Degenerate input (no caption text) — nothing to translate.
      2. Sketch / keyelems edit-mode samples — the reference image
         (the user's sketch or the element collage) IS the spec, and
         VMT does NOT see it (only text + caption). Asking VMT to
         emit a fresh visual narrative on top of an unseen reference
         systematically produces narratives that conflict with the
         actual refer-image layout, drives task_veto clear_fail in
         the directive critic, and regresses these slices by
         the edit-task quality. The build_edit_instruction
         path handles these with a refer-aware short prompt that
         preserves the refer image's structure directly.

    Inpaint stays in the VMT path (the fill region's surrounding
    context can be described in text from the caption + instruction;
    the fill region is described from the caption + instruction).
    """
    desc = (getattr(craft_input, "description", "") or "").strip()
    if not desc:
        return False
    figure_type = (getattr(craft_input, "figure_type", "") or "").lower()
    refer_role = (getattr(craft_input, "refer_image_role", "") or "").lower()
    if refer_role and figure_type in ("sketch", "keyelems"):
        return False
    # Sketch-passthrough samples: the inference driver extracts a text layout
    # from the rough sketch (SketchAnalyzer) and routes the case as
    # t2i / method_pipeline with the layout appended to the
    # description. VMT cannot improve on the SketchAnalyzer's
    # already-extracted layout, and emitting a separate visual
    # narrative on top regresses sketch faithfulness (-10.61pp on
    # the sketch layout. Skip when the passthrough
    # marker is present.
    if "## Extracted Sketch Layout" in desc:
        return False
    return True


class VisualMetaphorTranslator:
    """Translates technical figure captions into image-gen-friendly visual
    narratives by mapping abstract concepts to concrete visual metaphors,
    anchored to a named reference aesthetic.
    """

    def __init__(self, router: "ModelRouter"):
        self.router = router

    def translate(
        self,
        *,
        paper_text: str,
        raw_caption: str,
        instruction: str = "",
        task: str = "t2i",
        role: str = "academic",
        has_refer_image: bool = False,
        aesthetic_anchor: Optional[str] = None,
        aesthetic_intent: str = "",
    ) -> VisualTranslation:
        """Run a single LLM call to produce a visual narrative.

        The system prompt is built by a router:
          base rules + task protocol + role visual guide
        Unknown task / role names fall through to generic clauses
        (with-ref / no-ref task fallback; neutral role fallback).

        Anchor selection (priority order):
          1. ``aesthetic_anchor`` explicit name → used verbatim
          2. ``aesthetic_intent`` free-form text → LLM proposes anchor
          3. role-based auto-pick → catalog entry (unknown roles fall
             through to "academic paper figure")
        Returns an empty VisualTranslation on LLM failure.
        """
        if aesthetic_anchor:
            anchor_desc = _AESTHETIC_ANCHORS.get(aesthetic_anchor, aesthetic_anchor)
            anchor_hint = (f"FIXED anchor (use exactly this name and "
                           f"descriptor; do not invent a different one):\n"
                           f"  ANCHOR_NAME: {aesthetic_anchor}\n"
                           f"  ANCHOR_DESCRIPTOR: {anchor_desc}\n")
        elif aesthetic_intent.strip():
            anchor_hint = (f"USER AESTHETIC INTENT (invent an anchor name "
                           f"+ descriptor that honors this; ANY style is OK):\n"
                           f"  {aesthetic_intent.strip()[:400]}\n")
            aesthetic_anchor = "(LLM-proposed)"
        else:
            aesthetic_anchor = self._auto_anchor(role)
            anchor_desc = _AESTHETIC_ANCHORS.get(aesthetic_anchor, "")
            anchor_hint = (f"ROLE-DEFAULT anchor suggestion (you MAY override "
                           f"if a better-matching style fits the paper):\n"
                           f"  ANCHOR_NAME: {aesthetic_anchor}\n"
                           f"  ANCHOR_DESCRIPTOR: {anchor_desc}\n")

        # Suggest a protocol + style based on declared values; the LLM
        # may override per the DECISION FRAMEWORK in the system prompt.
        task_lc = (task or "").lower().strip()
        role_lc = (role or "").lower().strip()
        suggested_protocol = (task_lc if task_lc in _TASK_PROTOCOLS
                              else ("generic-with-ref" if has_refer_image
                                    else "generic-no-ref"))
        suggested_style = (role_lc if role_lc in _ROLE_VISUAL_GUIDES
                           else "general / unspecified")
        caption_shape = "abstract / long" if _looks_abstract(raw_caption) else "concrete / brief"

        parts = []
        if instruction:
            parts.append(
                f"## PRIMARY INSTRUCTION (authoritative — follow this verbatim)\n"
                f"{instruction[:800]}"
            )
        parts.append(f"## Paper context (excerpt)\n{paper_text[:6000]}")
        parts.append(f"## Raw figure caption\n{raw_caption[:1500]}")
        parts.append(
            f"## Input signals (you may override declared values)\n"
            f"- Declared task: {task or '(none)'}  → suggested protocol: {suggested_protocol}\n"
            f"- Declared role: {role or '(none)'}  → suggested style:    {suggested_style}\n"
            f"- Reference image attached: {'yes' if has_refer_image else 'no'}\n"
            f"- Caption shape: {caption_shape}\n"
            f"Apply the DECISION FRAMEWORK to pick the actual style /\n"
            f"protocol / enrichment level. Bias toward MINIMAL when the\n"
            f"caption is concrete and the role is academic; bias toward\n"
            f"HEAVY when aesthetic_intent is set or the role is poster /\n"
            f"infographic."
        )
        parts.append(f"## Anchor guidance\n{anchor_hint}")
        parts.append(
            "Now emit ANCHOR_NAME, ANCHOR_DESCRIPTOR, ---, then the visual "
            "narrative at the enrichment level you chose."
        )
        user = "\n\n".join(parts)

        system_prompt = _build_system_prompt(task, role, has_refer_image)

        try:
            output = self.router._chat(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user},
                ],
                model=self.router.config.planner_model,
                temperature=0.3,
                max_tokens=4000,
            )
        except Exception as e:
            logger.warning(f"VisualMetaphorTranslator LLM call failed: {e}")
            return VisualTranslation()

        output = (output or "").strip()
        if output.startswith("```"):
            lines = output.split("\n")
            output = "\n".join(lines[1:])
            if output.endswith("```"):
                output = output[:-3]
        output = output.strip()

        if len(output) < 100:
            logger.warning(f"VisualMetaphorTranslator output too short ({len(output)}); discarding")
            return VisualTranslation()

        # Parse ANCHOR_NAME / ANCHOR_DESCRIPTOR header + narrative
        parsed_name, parsed_desc, narrative = _parse_anchor_header(output)
        if parsed_name:
            aesthetic_anchor = parsed_name
        return VisualTranslation(
            visual_caption=narrative[:5000],
            aesthetic_anchor=aesthetic_anchor,
        )

    @staticmethod
    def _auto_anchor(role: str) -> str:
        role_lc = (role or "").lower().strip()
        return {
            "poster":      "conference research poster",
            "infographic": "blog-style explainer",
            "academic":    "academic paper figure",
        }.get(role_lc, "academic paper figure")


def _parse_anchor_header(text: str) -> tuple[str, str, str]:
    """Extract (anchor_name, anchor_descriptor, narrative) from LLM output.

    Expected format::

        ANCHOR_NAME: <name>
        ANCHOR_DESCRIPTOR: <descriptor>
        ---
        <narrative body...>

    Tolerates missing/whitespace variants. If the header is malformed,
    returns ("", "", original_text) so callers fall back to the
    narrative-as-whole.
    """
    if not text:
        return "", "", ""
    # Split off everything after the first '---' separator
    parts = re.split(r"\n\s*-{3,}\s*\n", text, maxsplit=1)
    if len(parts) == 2:
        header, body = parts
    else:
        # No separator — heuristically try first two lines as header
        head_lines = text.split("\n", 2)
        if len(head_lines) >= 3 and "ANCHOR_NAME" in head_lines[0]:
            header = "\n".join(head_lines[:2])
            body = head_lines[2]
        else:
            return "", "", text.strip()

    name = ""
    desc = ""
    for line in header.splitlines():
        line = line.strip().lstrip("*").lstrip("#").strip()
        if line.upper().startswith("ANCHOR_NAME"):
            name = line.split(":", 1)[1].strip() if ":" in line else ""
            name = name.strip("`*\"' ")
        elif line.upper().startswith("ANCHOR_DESCRIPTOR"):
            desc = line.split(":", 1)[1].strip() if ":" in line else ""
            desc = desc.strip("`*\"' ")
    return name, desc, body.strip()
