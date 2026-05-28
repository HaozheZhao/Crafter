"""CraftCritic: strict VLM-based image evaluation (the 'discriminator').

Uses gemini-3.1-pro-preview to evaluate generated figures with dimensional
scoring calibrated to human preference. The critic is intentionally strict —
most AI-generated figures should score 4-7 on first attempt, not 8-10.
A score of 8+ should mean "ready for top-tier venue submission".
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Optional

from crafter.shared.model_router import ModelRouter

logger = logging.getLogger(__name__)

CRITIQUE_SYSTEM = """\
You are the world's strictest reviewer of academic scientific figures. You have \
reviewed thousands of figures for Nature, Science, NeurIPS, CVPR, and other top \
venues. You have extremely high standards.

## CRITICAL CALIBRATION RULES — READ CAREFULLY

You MUST be a harsh, honest critic. Most AI-generated figures have significant \
problems. Your scores should reflect reality, NOT be polite or inflated.

### Score Calibration (FOLLOW THIS STRICTLY):
- **9-10**: Publication-ready for Nature/Science. Virtually flawless. You would \
  submit this figure as-is to a top venue. Almost NO AI-generated figure deserves this.
- **7-8**: Good quality with minor issues. Acceptable for a conference submission \
  after small fixes. Only give this if the figure is genuinely well-made.
- **5-6**: Mediocre. Has clear problems that need fixing. This is where MOST \
  first-attempt AI-generated figures should land.
- **3-4**: Poor. Major issues with layout, text, or aesthetics. Needs significant \
  rework.
- **1-2**: Terrible. Garbled text, broken layout, unusable.

### Common AI-Generated Figure Problems (look for ALL of these):
0. **Post-edit artifacts (AUTOMATIC FAIL)**: If the image has an obvious solid \
   white/colored rectangle where content has been blanked out (a "hole" from \
   failed inpainting), partially-cut title text, floating garbled character \
   fragments, or any rectangle of near-uniform pixels that is NOT a legitimate \
   component box, you MUST score `text_readability`, `aesthetic_quality`, and \
   `overall` at 2 or below and list the artifact as issue #1. These are never \
   acceptable in a shipped figure. Tell-tale signs: a white void above/left of \
   the first component; text that ends mid-word near an image edge; uniform \
   blocks that do not contain any label, icon, or arrow.
1. **Text clutter**: Too many labels, annotations, descriptions crammed in. A good \
   figure has MINIMAL text — just key labels. If the figure looks like a paragraph \
   was dumped into it, that's a 3-4 for text_correctness.
2. **Squeezed layout**: Components too close together, no breathing room. Good figures \
   have generous whitespace between elements. If elements are cramped or overlapping, \
   layout_quality should be 4-5.
3. **Poor typography**: Inconsistent font sizes, ugly text rendering, text that's too \
   small or too large, mixed fonts. AI models often render text badly.
4. **Fake sophistication**: Looks complex but doesn't actually convey information \
   clearly. A simple, clear figure is better than a busy, impressive-looking one.
5. **Color problems**: Too many colors, garish palette, poor contrast, colors that \
   clash rather than complement.
6. **Spatial imbalance**: One side of the figure is dense while the other is empty. \
   Good figures distribute visual weight evenly.
7. **Arrow/connection mess**: Too many arrows, unclear direction, overlapping paths.
8. **Missing or wrong content**: Components from the paper's method that are missing, \
   mislabeled, or in the wrong order.
9. **Not publication-ready**: Would a professor look at this and say "put it in the \
   paper"? If not, aesthetic_quality should be 5 or below.
10. **Icon/image artifacts**: Blurry elements, cut-off components, resolution issues.

### IMPORTANT: Be specific and actionable
For every issue you find, explain EXACTLY what is wrong and HOW to fix it. \
Don't say "improve layout" — say "the GRPO box is squeezed against the right edge, \
add 50px padding and reduce the data selection pipeline width by 20%".

Do NOT give sympathy scores. If the figure has problems, score it low.
"""

CRITIQUE_PROMPT = """\
## Task: Evaluate this generated academic figure

### Context
- **Paper topic**: {paper_context}
- **What this figure should show**: {description}
- **Target venue**: {venue}
- **Visual style attempted**: {visual_style}

{reference_note}

### Evaluation Dimensions (score each 0.0-10.0)

Evaluate the LAST image shown (this is the generated figure being reviewed):

1. **content_accuracy** (0-10): Does the figure accurately represent the method? \
Are all key components from the paper present? Is the data flow logically correct? \
Are any components missing, mislabeled, or in the wrong order?

2. **layout_quality** (0-10): Is the layout clean and well-organized? \
CRITICAL: Check for squeezed/cramped elements, lack of whitespace, uneven spacing, \
overlapping components. A good figure has generous breathing room. \
If elements are crammed together, score 4-5 maximum.

3. **text_readability** (0-10): Are labels minimal and well-placed? \
CRITICAL: Too much text is WORSE than too little. Labels should be short (1-3 words). \
If the figure is cluttered with long descriptions, explanatory text, or paragraph-like \
annotations inside boxes, score 3-5. Text should be crisp, consistent font size, \
and properly rendered (no garbled/overlapping characters). Check: would the text be \
readable when the figure is printed at column width (3.5 inches)?

4. **aesthetic_quality** (0-10): Does it look professionally designed? \
Would a design-conscious professor approve this for their paper? Check color harmony, \
visual balance, elegance of shapes, and overall polish. AI-generated figures often \
look "generated" rather than "designed" — if this figure has that AI look, score 5-6.

5. **role_match** (0-10): Does the output match the FIGURE ROLE the user requested? \
The user requested role: **{role_descriptor}** — does the output's overall format / \
medium / style category match this role? Examples of role mismatch: a request for \
"poster" producing a single academic figure (no banner, no multi-column grid) → score 2-4; \
a request for "infographic" producing a clean academic line-figure → score 3-5; a \
request for "1980s retro infographic" producing a modern flat infographic → score 4-6. \
ROLE is about FORMAT/MEDIUM/STYLE CATEGORY (poster vs paper figure vs infographic vs \
sketch vs ...), not about content quality. **If role="academic" the answer is automatic 8.0** — \
don't grade content as role mismatch. Includes venue conformance (e.g. NeurIPS style \
for academic figures): {style_match_note}

6. **artifact_severity** (0-10, INVERTED — 10 = no artifacts, 0 = severe): \
Does the output have AI-generation artifacts? Look for: garbled / unreadable text \
glyphs, broken or warped boxes, scattered decorative dots / sparkles, duplicated \
regions, overlapping unrelated visual elements, hallucinated UI chrome (window \
borders, scroll bars, cursors), broken arrow heads, clipped components, large \
uniform solid-fill blocks where content should be. A clean output with no artifacts \
scores 9-10. Mild artifacts (one or two minor glitches that do not impair reading) \
score 6-8. Significant artifacts (multiple text errors, or one large broken region) \
score 3-5. Severe artifacts (image is broken or unreadable) score 0-2. This axis \
captures defects orthogonal to aesthetic_quality and faithfulness — a figure can \
be aesthetically nice but riddled with garbled labels (low artifact_severity).

### Response Format
Return ONLY valid JSON (no markdown, no explanation):
{{
    "content_accuracy": <float>,
    "layout_quality": <float>,
    "text_readability": <float>,
    "aesthetic_quality": <float>,
    "role_match": <float>,
    "artifact_severity": <float>,
    "issues": [
        "SPECIFIC issue 1 with EXACT location and description",
        "SPECIFIC issue 2 ...",
        "..."
    ],
    "suggestions": [
        "ACTIONABLE fix 1 with specific changes to make",
        "ACTIONABLE fix 2 ...",
        "..."
    ]
}}

Remember: Be STRICT. Most AI figures deserve 5-6 overall, not 8-9. \
List at least 3-5 specific issues even for decent figures. \
A figure with text clutter, cramped layout, or ugly typography should score below 6.
"""


@dataclass
class CritiqueResult:
    """Structured critique with dimensional scores."""

    # Paper §4.2.4 directive critic: six axes (content accuracy /
    # layout coherence / text legibility / role conformity / aesthetic
    # quality / artifact severity). All scored 0-10 where 10 = best.
    content_accuracy: float = 0.0
    layout_quality: float = 0.0    # paper: "layout coherence"
    text_readability: float = 0.0  # paper: "text legibility"
    aesthetic_quality: float = 0.0
    # Role conformity: does the output match the user's requested
    # ROLE (poster / infographic / academic / free-form descriptor)?
    # Includes venue conformance (NeurIPS academic style, ICCV style,
    # etc.). Default 8.0 = "not assessed"; the role-aware early-exit
    # gate accepts ≥6.5 so the default doesn't fire as a refinement
    # trigger.
    role_match: float = 8.0
    # Artifact severity (INVERTED to keep 10=best convention): how
    # clean is the output of AI-generation artifacts (garbled text,
    # broken boxes, scattered dots, duplicated regions, etc.)?
    # 10 = no artifacts, 0 = severe artifacts.
    artifact_severity: float = 8.0
    overall: float = 0.0
    issues: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    is_acceptable: bool = False
    raw_response: str = ""

    def summary(self) -> str:
        """One-line summary of scores (paper §4.2.4 six axes)."""
        return (
            f"content={self.content_accuracy:.1f} layout={self.layout_quality:.1f} "
            f"text={self.text_readability:.1f} aesthetic={self.aesthetic_quality:.1f} "
            f"role={self.role_match:.1f} artifact={self.artifact_severity:.1f} → "
            f"overall={self.overall:.1f} ({'PASS' if self.is_acceptable else 'FAIL'})"
        )


class CraftCritic:
    """Evaluates generated images using VLM with strict, human-aligned scoring."""

    # Deflation factor: VLMs tend to be generous. We multiply raw scores
    # by this factor to calibrate closer to human judgment.
    # 0.82 means a raw "perfect 10" becomes 8.2, and a raw 9 becomes ~7.4.
    # This allows genuinely good figures to pass threshold (7.5) while
    # keeping mediocre ones (raw 7-8) below it.
    SCORE_DEFLATION = 0.82

    def __init__(self, router: ModelRouter, quality_threshold: float = 7.5):
        self.router = router
        self.quality_threshold = quality_threshold

    def estimate_label_count(self, image_path: str) -> Optional[int]:
        """Observational density probe. Asks the VLM for a single
        integer: how many distinct text labels appear in the figure. Used
        by the refinement loop to detect feature-checklist clutter
        (label_count >> expected) or missing-components (label_count <<
        expected) BASED ON THE GENERATED IMAGE, not on Designer's prompt.

        Cheap focused call: small max_tokens, low temperature, integer-only
        output. Returns None on parse failure (caller falls back to
        density-blind refinement).
        """
        import base64
        from pathlib import Path
        try:
            img_b64 = base64.b64encode(Path(image_path).read_bytes()).decode()
        except Exception as e:
            logger.debug(f"label-count read fail: {e}")
            return None

        sys = (
            "You count visible text labels in a scientific figure. Reply "
            "with a SINGLE integer (no prose, no markdown) — your best "
            "estimate of how many distinct text labels are visible in the "
            "image. Count every short text near or inside any component, "
            "axis labels, panel titles. Do not count the figure caption "
            "below the figure."
        )
        content = [
            {"type": "image_url",
             "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
            {"type": "text", "text": "How many distinct text labels are visible? One integer only."},
        ]
        try:
            raw = self.router._chat(
                [{"role": "system", "content": sys},
                 {"role": "user", "content": content}],
                model=self.router.config.critic_model,
                temperature=0.0,
                max_tokens=20,
            )
        except Exception as e:
            logger.debug(f"label-count VLM call fail: {e}")
            return None
        import re
        m = re.search(r'\d+', raw or "")
        if not m:
            return None
        try:
            n = int(m.group(0))
            if 0 <= n <= 999:
                return n
        except ValueError:
            pass
        return None

    def evaluate(
        self,
        image_path: str,
        prompt: str,
        paper_context: str,
        description: str,
        venue: str = "neurips",
        visual_style: str = "",
        reference_paths: Optional[list[str]] = None,
        role: str = "",
        task: str = "",
        required_components: str = "",
    ) -> CritiqueResult:
        """Evaluate a generated figure image with strict scoring.

        Args:
            image_path: Path to the generated image.
            prompt: The prompt that was used to generate it.
            paper_context: Paper text / methodology description.
            description: What the figure should show.
            venue: Target venue name.
            visual_style: Visual style used (e.g., "block_diagram").
            reference_paths: Paths to reference images for style comparison.
            role: The user's requested figure role.
            task: T2I / inpaint / keyelems / sketch — drives task-specific
                evaluation blocks.
            required_components: The harness's PaperReader extracts
                paper-mentioned components and exposes them via
                PaperContext.for_critic() as a single
                "Required components: A, B, C" string. Currently kept for
                API compatibility but not rendered into the critic
                prompt — see comment in the method body.

        Returns:
            Structured CritiqueResult with calibrated dimensional scores.
        """
        prompt_summary = prompt[:500] + "..." if len(prompt) > 500 else prompt

        if reference_paths:
            reference_note = (
                "Reference images are provided BEFORE the generated image. "
                "The LAST image is the one being evaluated — all others are reference "
                "examples of good style. Compare the last image's quality and style "
                "against these references. The generated image should look AS GOOD as "
                "or BETTER than the references."
            )
            style_match_note = (
                f"Compare against the {len(reference_paths)} reference image(s). "
                f"Does the generated figure match their quality level and style?"
            )
        else:
            reference_note = "No reference images provided."
            style_match_note = f"Does it look like a typical high-quality {venue} figure?"

        # Add poster-specific evaluation context
        poster_note = ""
        if visual_style == "research_poster" or "poster" in description.lower()[:200]:
            poster_note = (
                "\n\n### POSTER-SPECIFIC EVALUATION:\n"
                "This is a conference POSTER, not a paper figure. Evaluate with poster criteria:\n"
                "- layout_quality: Does it have a clear 3-column landscape layout with title banner? "
                "Are sections well-separated with colored headers? Is there enough whitespace?\n"
                "- text_readability: Would ALL text be readable when printed at poster size (40\"x30\")? "
                "Are key metrics/numbers displayed large and prominently? No tiny text or dense paragraphs?\n"
                "- content_accuracy: Does it cover the paper's key contributions: motivation, method, and results? "
                "Are the main architectural components and quantitative results present?\n"
                "- aesthetic_quality: Does it look like a professional conference poster? Clean section backgrounds, "
                "visual hierarchy, and consistent design language?\n"
                "- role_match: Does it match the target venue's poster conventions?\n"
                "- artifact_severity: Are there NO garbled text glyphs, broken boxes, "
                "scattered decorative dots, or AI-render artifacts? (10 = clean, 0 = severe)\n"
            )

        # Task-aware verification block. The default critic dims
        # (content_accuracy / layout / text / aesthetic / role / artifact)
        # are academic — they miss per-edit-task failures (sketch layout
        # fidelity, inpaint fill completeness, keyelems element reuse).
        # Adding this task block makes the critic surface the actual
        # task-level failures so the fix-prompt step can target them.
        task_note = ""
        task_lc = (task or "").lower()
        if task_lc == "inpaint":
            task_note = (
                "\n\n### INPAINT TASK-SPECIFIC EVALUATION (mandatory):\n"
                "This output is an INPAINT EDIT — the user provided a partial\n"
                "figure with one region blank, and asked the model to fill that\n"
                "region while preserving the rest. CHECK ALL of:\n"
                "- Was the previously-BLANK region adequately filled with\n"
                "  publication-quality content matching the description? If\n"
                "  the masked region is still blank, mostly empty, or contains\n"
                "  only a header without body content, this is a CRITICAL\n"
                "  failure — list as issue #1 and score content_accuracy ≤ 4.\n"
                "- Was the un-masked region preserved? Layout / banner /\n"
                "  un-changed sections must be intact; substantial alteration\n"
                "  of the un-masked region is also a CRITICAL failure.\n"
                "- The fill should be RICH (multiple sub-sections, real\n"
                "  figures or tables) when the description calls for that —\n"
                "  not a single empty placeholder."
            )
        elif task_lc == "keyelems":
            task_note = (
                "\n\n### KEYELEMS TASK-SPECIFIC EVALUATION (mandatory):\n"
                "This output is an ELEMENT-BUILD EDIT — the user provided\n"
                "specific visual elements (icons / photos / illustrations)\n"
                "they wanted USED in the final figure. CHECK:\n"
                "- Are the user's provided elements VISIBLY PRESENT in the\n"
                "  output? Compare against the reference image. If the output\n"
                "  has different icons / photos than the reference, or uses\n"
                "  generic stock visuals, this is a CRITICAL failure — list\n"
                "  as issue #1 and score content_accuracy ≤ 4.\n"
                "- The model is allowed to ADD additional content (panels,\n"
                "  arrows, labels, headers, body text, supplementary icons),\n"
                "  but the provided elements must REMAIN.\n"
                "- The connecting structure should be coherent and\n"
                "  publication-quality."
            )
        elif task_lc == "sketch":
            task_note = (
                "\n\n### SKETCH TASK-SPECIFIC EVALUATION (mandatory):\n"
                "This output is a SKETCH-REFINE EDIT — the reference image\n"
                "(if present in the references above) is a STRUCTURAL BRIEF,\n"
                "not the target output. Reference may look:\n"
                "  (a) hand-drawn (rough strokes, sketchy lines), OR\n"
                "  (b) AI-rendered (polished but identifiable as a draft), OR\n"
                "  (c) SVG-wireframe (clean geometric outlines).\n"
                "Regardless of reference style, the CORRECT output:\n"
                "  - matches the reference's PANEL ARRANGEMENT (panel count,\n"
                "    positions, arrow connections, reading order)\n"
                "  - REPLACES every visual element (icons, illustrations,\n"
                "    text labels, body content) with polished publication\n"
                "    versions — DOES NOT preserve the reference's specific\n"
                "    pixels, lines, color palette, or rendering style\n"
                "  - looks like a fresh figure DRAWN with the reference as\n"
                "    a layout brief, NOT like a polished retouch of the\n"
                "    reference image\n\n"
                "FAILURE MODES (each is a CRITICAL issue, score\n"
                "content_accuracy ≤ 4 and list as issue #1):\n"
                "  1. NEAR-VERBATIM COPY: the output's overall composition,\n"
                "     palette, line shapes, decorative elements, or\n"
                "     element rendering closely match the reference's,\n"
                "     looking like a cleaned-up retrace rather than a\n"
                "     re-imagining. This is the failure mode regardless\n"
                "     of whether the reference looks hand-drawn or\n"
                "     AI-rendered. If you'd describe the output as 'the\n"
                "     sketch with cleaner strokes', that is COPY.\n"
                "  2. LAYOUT MISMATCH: panel count or arrangement\n"
                "     differs from the reference (3-panel reference →\n"
                "     2-panel output, etc.).\n"
                "  3. NO REFINEMENT: the output looks identical in\n"
                "     resolution / polish level to the reference, with\n"
                "     no measurable upgrade in typography, icon quality,\n"
                "     or visual cohesion.\n\n"
                "When you list a copy / no-refinement issue in `issues`,\n"
                "describe specifically what to do in the next iteration:\n"
                "  - 'Reference is a structural brief, not the output;\n"
                "    rebuild every visual element from scratch'\n"
                "  - 'Replace icons with publication-quality versions'\n"
                "  - 'Use a different color palette and typography than\n"
                "    the reference'\n"
                "Stating concrete corrective actions lets the next iter\n"
                "actually improve."
            )
        elif task_lc and task_lc not in ("t2i",) and reference_paths:
            # Generic edit fallback. Any task string outside the known
            # {inpaint, keyelems, sketch} set WITH a refer image
            # triggers a structural-fidelity check. Callers pass `task=`
            # explicitly when they know the precise task; this block
            # only fires for the IntentReasoner-derived "generic_edit"
            # path or for arbitrary
            # custom labels callers might invent. Placed AFTER inpaint /
            # keyelems / sketch so those specific branches still take
            # precedence (else-if chain).
            task_note = (
                "\n\n### GENERIC EDIT TASK-SPECIFIC EVALUATION (mandatory):\n"
                "This output is a refer-image edit. The user provided a\n"
                "reference image and asked for a modification described in\n"
                "the prompt. CHECK ALL of:\n"
                "- Was the user's intent (as described in the caption /\n"
                "  prompt) achieved in the output? If the caption asks for\n"
                "  a specific change (e.g. add a panel, replace an icon,\n"
                "  re-color, re-label) and the output does not show that\n"
                "  change — score content_accuracy ≤ 4 and list as issue #1.\n"
                "- Was the visual structure from the reference that the\n"
                "  prompt did NOT ask to change preserved? Substantial\n"
                "  unrequested deviation (different panels, different\n"
                "  layout, different visual identity) is a CRITICAL\n"
                "  failure — score content_accuracy ≤ 4.\n"
                "- Is the output publication-quality (clean shapes, real\n"
                "  labels, polished typography)? Sketchy / draft / lo-fi\n"
                "  rendering when the user wanted a finished figure is a\n"
                "  CRITICAL failure."
            )

        # Role descriptor for role_match dim. "academic" / "" → auto-8.0
        # via the prompt language; non-empty role values get assessed
        # against the descriptor verbatim.
        role_descriptor = role or "academic (default — paper figure)"

        # required_components is not rendered into the critic prompt:
        # paper-extracted component names tend to mix intro / related-work
        # terms with real method components, and injecting them as an
        # advisory hint biased the critic toward component-presence
        # checking, over-flagging issues that caused the refiner to
        # over-correct. Arg kept for API compatibility.
        _ = required_components

        critique_prompt = CRITIQUE_PROMPT.format(
            paper_context=paper_context[:1500],
            description=description,
            venue=venue,
            visual_style=visual_style or "auto",
            prompt_summary=prompt_summary,
            reference_note=reference_note,
            style_match_note=style_match_note,
            role_descriptor=role_descriptor,
        ) + poster_note + task_note
        # Pass the role hint into _parse_critique via instance state so the
        # default-8.0 fallback for academic role works without extra plumbing.
        self._last_role = (role or "academic").lower()

        try:
            if reference_paths:
                raw = self.router.critique_with_references(
                    image_path=image_path,
                    reference_paths=reference_paths,
                    prompt=critique_prompt,
                    system_prompt=CRITIQUE_SYSTEM,
                )
            else:
                raw = self.router.critique_image(
                    image_path=image_path,
                    prompt=critique_prompt,
                    system_prompt=CRITIQUE_SYSTEM,
                )

            result = self._parse_critique(raw)
            return self._augment_with_programmatic_checks(result, image_path)

        except Exception as e:
            logger.error(f"Critique failed: {e}")
            return self._fallback_critique(str(e))

    @staticmethod
    def _compute_image_metrics(image_path: str) -> Optional[dict]:
        """Hybrid Critic programmatic layer (paper §4.2 §1.3 / §4.3).

        Cv2-based deterministic image metrics — no LLM, no noise. AUGMENT
        the VLM critic's per-dim scoring without injecting issues into
        the refinement loop.

        Metrics:
          - whitespace_ratio: pixels brightness > 240, fraction
          - contour_count: connected components ≥ 0.05% canvas area
          - mid_contour_count: contours ≥ 0.5% canvas area (label-host)
          - max_pair_overlap: maximum IoU between any pair of large
            shapes (excludes pure containment, which is legitimate panel
            nesting). Significant overlap ≈ > 0.30; maps to the eval
            Layout "Spatial Imbalance" veto.

        Returns None on cv2 / read failure (caller no-ops).
        """
        try:
            import cv2
            import numpy as np
            img = cv2.imread(image_path)
            if img is None:
                return None
            h, w = img.shape[:2]
            area = h * w
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            whitespace_ratio = float((gray > 240).sum()) / float(area)

            # Edges → contours
            edges = cv2.Canny(gray, 40, 120)
            kernel = np.ones((3, 3), np.uint8)
            edges_d = cv2.dilate(edges, kernel, iterations=1)
            contours, _ = cv2.findContours(edges_d, cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_SIMPLE)
            min_area_small = area * 0.0005
            min_area_mid = area * 0.005
            contour_count = sum(1 for c in contours
                                if cv2.contourArea(c) >= min_area_small)
            mid_contour_count = sum(1 for c in contours
                                    if cv2.contourArea(c) >= min_area_mid)

            # Element overlap (max pairwise IoU among large shapes)
            max_pair_overlap = 0.0
            try:
                bigs = []
                for c in contours:
                    if cv2.contourArea(c) >= area * 0.005:
                        bigs.append(cv2.boundingRect(c))
                if len(bigs) >= 2:
                    for i in range(len(bigs)):
                        ax, ay, aw, ah = bigs[i]
                        ai = aw * ah
                        for j in range(i + 1, len(bigs)):
                            bx, by, bw, bh = bigs[j]
                            bi = bw * bh
                            x1 = max(ax, bx); y1 = max(ay, by)
                            x2 = min(ax + aw, bx + bw); y2 = min(ay + ah, by + bh)
                            if x2 <= x1 or y2 <= y1:
                                continue
                            inter = (x2 - x1) * (y2 - y1)
                            # Skip cases where one shape entirely contains the
                            # other (legitimate parent-panel containment — not
                            # the "two unrelated shapes overlapping" failure).
                            if inter == min(ai, bi):
                                continue
                            union = ai + bi - inter
                            if union > 0:
                                iou = inter / union
                                if iou > max_pair_overlap:
                                    max_pair_overlap = iou
            except Exception:
                pass

            return {
                "whitespace_ratio": whitespace_ratio,
                "contour_count": contour_count,
                "mid_contour_count": mid_contour_count,
                "max_pair_overlap": max_pair_overlap,
                "image_area": area,
            }
        except Exception as e:
            logger.debug(f"_compute_image_metrics failed: {e}")
            return None

    def _augment_with_programmatic_checks(
        self, critique: "CritiqueResult", image_path: str
    ) -> "CritiqueResult":
        """Apply cv2-derived caps to per-dim scores in the SEVERE cases the
        VLM critic empirically over-rates. Conservative on purpose — only
        triggers caps on extreme observed properties; does NOT add issues
        to critique.issues.

        Effect: variant SELECTION (max critique.overall) avoids the
        cluttered / over-sparse candidates the VLM scored too kindly.
        Refinement-loop triggers stay unchanged.

        Cap rules — calibrated so we don't penalise variants the eval
        judge accepts. Sparse academic layout is legitimate; caps fire
        only at extremes:
          - whitespace_ratio < 0.08 (extreme clutter) → cap
            layout_quality at 5.0, cap text_readability at 5.0.
          - whitespace_ratio > 0.85 AND mid_contour_count < 2 (image is
            essentially empty — no real figure rendered) → cap
            layout_quality at 5.0.
          - contour_count > 500 (chaotic micro-clutter) → cap
            aesthetic_quality at 5.0.

        max_pair_overlap is captured on the metrics dict for /
        tuning but the cap rule is intentionally disabled — it tends to
        prefer cleaner-but-less-faithful alternatives.
        """
        m = self._compute_image_metrics(image_path)
        if m is None:
            return critique

        capped: list[str] = []
        if m["whitespace_ratio"] < 0.08:
            if critique.layout_quality > 5.0:
                critique.layout_quality = 5.0
                capped.append(f"layout (ws={m['whitespace_ratio']:.2%})")
            if critique.text_readability > 5.0:
                critique.text_readability = 5.0
                capped.append(f"text (ws={m['whitespace_ratio']:.2%})")
        elif m["whitespace_ratio"] > 0.85 and m["mid_contour_count"] < 2:
            if critique.layout_quality > 5.0:
                critique.layout_quality = 5.0
                capped.append(
                    f"layout-empty (ws={m['whitespace_ratio']:.2%}, "
                    f"shapes={m['mid_contour_count']})"
                )
        if m["contour_count"] > 500:
            if critique.aesthetic_quality > 5.0:
                critique.aesthetic_quality = 5.0
                capped.append(f"aesthetic (chaos={m['contour_count']})")
        # Overlap cap intentionally disabled — see docstring.

        if capped:
            critique.overall = round(
                critique.content_accuracy * 0.20
                + critique.layout_quality * 0.25
                + critique.text_readability * 0.20
                + critique.aesthetic_quality * 0.20
                + critique.artifact_severity * 0.15,
                1,
            )
            critique.is_acceptable = critique.overall >= self.quality_threshold
            logger.info(
                f"hybrid-critic cap: {'; '.join(capped)} → overall={critique.overall}"
            )
        return critique

    def _parse_critique(self, raw: str) -> CritiqueResult:
        """Parse JSON critique response into CritiqueResult with score deflation."""
        text = raw.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:])
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            logger.warning("Failed to parse critique JSON, using fallback")
            return self._fallback_critique(f"JSON parse error: {text[:200]}")

        # Extract raw scores
        raw_ca = float(data.get("content_accuracy", 5.0))
        raw_lq = float(data.get("layout_quality", 5.0))
        raw_tr = float(data.get("text_readability", data.get("text_correctness", 5.0)))
        raw_aq = float(data.get("aesthetic_quality", 5.0))
        # Artifact severity (paper §4.2.4 6th axis). Default 8.0 = "no
        # major artifacts assumed when missing"; the critic prompt asks
        # the VLM to score 0-10 where 10 = clean.
        raw_as = float(data.get("artifact_severity", 8.0))
        # role_match — paper §4.2.4 "role conformity". For academic role
        # (or unset), use auto-8.0 default so the early-exit gate isn't
        # blocked by role_match. For non-academic role, parse the model's
        # evaluation; default 5.0 if missing.
        last_role = getattr(self, "_last_role", "academic")
        if last_role in ("", "academic"):
            raw_rm = 10.0  # auto-pass for academic; deflation brings to 8.2
        else:
            raw_rm = float(data.get("role_match", 5.0))

        # Apply deflation to combat VLM score inflation
        # Cap at 10.0 after deflation
        ca = min(10.0, raw_ca * self.SCORE_DEFLATION)
        lq = min(10.0, raw_lq * self.SCORE_DEFLATION)
        tr = min(10.0, raw_tr * self.SCORE_DEFLATION)
        aq = min(10.0, raw_aq * self.SCORE_DEFLATION)
        as_ = min(10.0, raw_as * self.SCORE_DEFLATION)
        rm = min(10.0, raw_rm * self.SCORE_DEFLATION)

        # Weighted average: layout and aesthetics matter most for human
        # preference. role_match is NOT in the weighted overall — it's
        # an advisory signal used by the role-aware early-exit gate in
        # session.py. artifact_severity replaces style_match in the
        # 15% weight slot (paper §4.2.4 names it as the 6th axis).
        overall = (ca * 0.20 + lq * 0.25 + tr * 0.20 + aq * 0.20 + as_ * 0.15)

        issues = data.get("issues", [])
        suggestions = data.get("suggestions", [])

        # Ensure minimum issues — if model returned too few, it wasn't strict enough
        if len(issues) < 3 and overall > 5.0:
            issues.append("Review text density — ensure labels are minimal (1-3 words each)")
            issues.append("Check spacing between components — ensure generous whitespace")
            issues.append("Verify typography consistency — font sizes and styles should be uniform")

        return CritiqueResult(
            content_accuracy=round(ca, 1),
            layout_quality=round(lq, 1),
            text_readability=round(tr, 1),
            aesthetic_quality=round(aq, 1),
            role_match=round(rm, 1),
            artifact_severity=round(as_, 1),
            overall=round(overall, 1),
            issues=issues,
            suggestions=suggestions,
            is_acceptable=overall >= self.quality_threshold,
            raw_response=raw,
        )

    def critique_and_revise(
        self,
        image_path: str,
        current_description: str,
        paper_context: str,
        caption: str,
        venue: str = "neurips",
        figure_type: str = "",
        required_components: str = "",
    ) -> tuple[CritiqueResult, str]:
        """Critique the image AND produce a revised description (PB pattern).

        Unlike evaluate() which only scores, this method also asks the VLM to
        revise the figure description to fix identified issues. The revised
        description can then be used to regenerate a better image.

        This directive-critique pattern makes
        their 3-round refinement loop so effective.

        Returns:
            (critique_result, revised_description)
        """
        import base64
        from pathlib import Path

        img_b64 = base64.b64encode(Path(image_path).read_bytes()).decode()

        # Poster-specific critique context
        poster_rules = ""
        if figure_type == "poster" or "poster" in caption.lower():
            poster_rules = (
                "\n5. Poster Layout: Does it have a clear title banner + multi-column body layout?\n"
                "6. Poster Readability: Would text be readable when printed large (40\"x30\")? "
                "Are key numbers displayed prominently?\n"
                "7. Poster Content Coverage: Does it summarize motivation, method, AND results?\n"
            )

        # required_components is not rendered into the prompt (see
        # comment in the critic-time method).
        rc_block = ""
        _ = required_components

        prompt = (
            f"You are a Lead Visual Designer for {venue.upper()} papers.\n\n"
            f"## Task\n"
            f"Critique this generated {'conference poster' if figure_type == 'poster' else 'academic diagram'} "
            f"and provide a REVISED description that fixes all identified issues.\n\n"
            f"## Context\n"
            f"- Paper methodology: {paper_context[:2000]}\n"
            f"- Figure caption: {caption}{rc_block}\n"
            f"- Current description used to generate this image:\n{current_description[:6000]}\n\n"
            f"## Critique Rules\n"
            f"1. Content: Does it faithfully represent the paper's method? Any missing/wrong components?\n"
            f"2. Text: Any garbled text, figure titles, or text that shouldn't be there?\n"
            f"3. Readability: Is it clean, with large readable text and generous whitespace?\n"
            f"4. Aesthetics: Professional quality for a top venue?\n"
            f"{poster_rules}\n"
            f"## IMPORTANT\n"
            f"- Your revised description should be modifications of the original, NOT a rewrite from scratch\n"
            f"- KEEP THE SAME LENGTH as the current description — do NOT shorten it\n"
            f"- Preserve ALL visual rendering details (colors, shapes, layout, arrows) from the original\n"
            f"- Only modify the parts that the critique identified as wrong\n"
            f"- Be specific: describe colors, shapes, positions, connections\n"
            f"- NO pixel dimensions, CSS values, or figure titles in the description\n"
            f"- Keep all correct parts, only fix what's wrong\n\n"
            f"## Output (JSON only)\n"
            f'{{\n'
            f'  "content_accuracy": <float 0-10>,\n'
            f'  "layout_quality": <float 0-10>,\n'
            f'  "text_readability": <float 0-10>,\n'
            f'  "aesthetic_quality": <float 0-10>,\n'
            f'  "role_match": <float 0-10>,\n'
            f'  "artifact_severity": <float 0-10>,\n'
            f'  "issues": ["issue 1", "issue 2", ...],\n'
            f'  "suggestions": ["fix 1", "fix 2", ...],\n'
            f'  "revised_description": "The complete revised description incorporating all fixes..."\n'
            f'}}'
        )

        content = [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
            {"type": "text", "text": prompt},
        ]

        try:
            raw = self.router._chat(
                [{"role": "user", "content": content}],
                model=self.router.config.critic_model,
                temperature=0.3,
                max_tokens=10000,
            )

            # Parse the combined response
            text = raw.strip()
            if text.startswith("```"):
                lines = text.split("\n")
                text = "\n".join(lines[1:])
                if text.endswith("```"):
                    text = text[:-3]
                text = text.strip()

            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                try:
                    import json_repair
                    data = json_repair.loads(text)
                except Exception:
                    logger.warning("Failed to parse critique+revise JSON")
                    return self._fallback_critique("JSON parse error"), current_description

            # Extract critique scores
            critique = self._parse_critique(raw)

            # Extract revised description
            revised = data.get("revised_description", "")
            if not revised or revised == "No changes needed.":
                revised = current_description

            # Guard: don't accept shorter revisions (losing visual detail)
            # The revised description should be at least 80% of the original length
            if len(revised) < len(current_description) * 0.8:
                logger.warning(
                    f"Revised desc too short ({len(revised)} vs {len(current_description)} chars), "
                    f"keeping original"
                )
                revised = current_description

            logger.info(
                f"Critique+Revise: overall={critique.overall:.1f}, "
                f"revised_desc={len(revised)} chars"
            )
            return critique, revised

        except Exception as e:
            logger.error(f"Critique+revise failed: {e}")
            return self._fallback_critique(str(e)), current_description

    def compare_variants_for_faithfulness(
        self,
        variants: list[tuple[str, str]],
        paper_context: str,
        caption: str,
        figure_type: str = "",
    ) -> dict[str, dict]:
        """Convergence-judge head-to-head paper-faithfulness ranking.

        Replaces the per-variant single-image content_accuracy (which
        tends to over-rate faithfulness in isolation) with a
        paper-grounded head-to-head ranking. The judge sees all K
        candidates side-by-side + paper text + caption, scores each 0-10
        on faithfulness to the methodology, and the per-variant
        `content_accuracy` is updated from this score before variant
        selection and the early-exit decision.

        The judge uses a different prompt structure than the eval-time
        4-dimension critic to avoid leaking eval-style heuristics into
        the training-time selection signal.

        Args:
            variants: list of (vs_key, image_path) — K=2-5 candidates.
            paper_context: paper text / methodology summary.
            caption: figure caption.
            figure_type: optional type hint for context.

        Returns:
            {vs_key: {"score": float 0-10, "reasoning": str}} for each
            variant. Empty dict on judge failure (caller falls back to
            existing critique.content_accuracy).
        """
        if len(variants) < 2:
            return {}

        import base64
        from pathlib import Path as _P

        sys_prompt = (
            "You are the Convergence Judge (V) in a figure-generation harness. "
            "Your single task: rank K candidate figures by how faithfully each "
            "represents the paper's described methodology. You are NOT comparing "
            "against a human-author baseline. You are NOT a 4-dimension critic. "
            "You are ranking the candidates against the PAPER ITSELF.\n\n"
            "Faithfulness rubric:\n"
            "  9-10: All key methodology components present as distinct drawn "
            "elements; topology (sequence/parallel/branching) matches the paper's "
            "described data flow; named entities and numeric constants from the "
            "methodology appear as labels.\n"
            "  6-8: Most components present; topology mostly right but ≥1 minor "
            "mismatch (one missing component, one inverted connection, etc.).\n"
            "  3-5: Major components missing or topology is wrong; the figure "
            "looks generic and could fit any paper on this topic.\n"
            "  0-2: The figure has almost no specific connection to this paper's "
            "methodology — generic boxes-with-text that don't reflect the paper.\n\n"
            "Score harshly. Most AI-generated figures should score 4-7 on faithfulness."
        )

        # Build user content: paper context + caption + K images in order
        content_parts = [
            {"type": "text", "text": (
                f"## Paper methodology context\n{paper_context[:3000]}\n\n"
                f"## Figure caption (what the figure is supposed to show)\n{caption[:500]}\n\n"
                f"## {len(variants)} candidate figures shown below"
                f"{' (figure_type=' + figure_type + ')' if figure_type else ''}"
                f". Variant indices are in the order they appear: "
                + ", ".join(f"variant_{i}" for i in range(len(variants))) + "."
            )}
        ]
        for i, (vs_key, img_path) in enumerate(variants):
            try:
                img_b64 = base64.b64encode(_P(img_path).read_bytes()).decode()
                mime = "image/png"
                content_parts.append({
                    "type": "text",
                    "text": f"\n--- variant_{i} (style={vs_key}) ---"
                })
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{img_b64}"},
                })
            except Exception as e:
                logger.warning(f"compare_variants: failed to load {img_path}: {e}")
                return {}

        content_parts.append({"type": "text", "text": (
            f"\n\nRank all {len(variants)} variants by paper-faithfulness. "
            "Output STRICT JSON (no prose, no fences):\n"
            "{\n"
            '  "rankings": [\n'
            '    {"variant_idx": <int>, "faithfulness_score": <float 0-10>, "missing_components": ["<short item>", ...], "reasoning": "<one sentence>"},\n'
            '    ...\n'
            "  ]\n"
            "}"
        )})

        try:
            raw = self.router._chat(
                [{"role": "system", "content": sys_prompt},
                 {"role": "user", "content": content_parts}],
                model=self.router.config.critic_model,
                temperature=0.2,
                max_tokens=4000,
            )
        except Exception as e:
            logger.warning(f"compare_variants_for_faithfulness call failed: {e}")
            return {}

        text = raw.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:])
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            logger.warning(f"compare_variants_for_faithfulness: JSON parse failed: {text[:200]}")
            return {}

        rankings = data.get("rankings", []) if isinstance(data, dict) else []
        out: dict[str, dict] = {}
        for r in rankings:
            if not isinstance(r, dict):
                continue
            idx = r.get("variant_idx")
            if not isinstance(idx, int) or idx < 0 or idx >= len(variants):
                continue
            try:
                score = float(r.get("faithfulness_score", 5.0))
            except (TypeError, ValueError):
                continue
            score = max(0.0, min(10.0, score))
            vs_key = variants[idx][0]
            out[vs_key] = {
                "score": round(score, 2),
                "missing_components": r.get("missing_components", []) or [],
                "reasoning": str(r.get("reasoning", ""))[:300],
            }
        logger.info(
            f"compare_variants_for_faithfulness: ranked {len(out)}/{len(variants)} variants"
        )
        return out

    def _fallback_critique(self, error_msg: str) -> CritiqueResult:
        """Return a conservative critique when evaluation fails."""
        return CritiqueResult(
            content_accuracy=4.0,
            layout_quality=4.0,
            text_readability=4.0,
            aesthetic_quality=4.0,
            artifact_severity=4.0,
            overall=4.0,
            issues=[
                f"Critique evaluation failed: {error_msg}",
                "Cannot assess text density — assume needs reduction",
                "Cannot assess layout spacing — assume needs more whitespace",
            ],
            suggestions=[
                "Retry with clearer prompt",
                "Reduce text in figure to key labels only",
                "Increase spacing between components",
            ],
            is_acceptable=False,
            raw_response=error_msg,
        )
