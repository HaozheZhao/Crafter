"""PromptRefiner: uses the most powerful LLM to improve generation prompts.

Uses claude-4.6-opus to analyze critique feedback and iteration history,
then produces an improved prompt that addresses identified issues.
Includes anti-leak sanitization to prevent formatting artifacts in output.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Optional

from crafter.shared.model_router import ModelRouter

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Anti-leak patterns: formatting instructions that image models render as text
# ──────────────────────────────────────────────────────────────────────────────

_LEAK_PATTERNS = [
    # Pixel dimensions: "180 x 50 px", "200x100px", "160 x 45 px"
    (r'\b\d+\s*x\s*\d+\s*px\b', ''),
    # CSS-style values: "8px", "12pt", "2.5px", "24px", "1.5px"
    (r'\b\d+\.?\d*\s*px\b', ''),
    (r'\b\d+\.?\d*\s*pt\b', ''),
    # Explicit coordinate instructions: "at position (100, 200)"
    (r'at position\s*\(\s*\d+\s*,\s*\d+\s*\)', ''),
    # Border/margin/padding specs: "border: 2px", "margin: 10px"
    (r'\b(?:border|margin|padding|radius|width|height)\s*:\s*\d+\.?\d*\s*(?:px|pt|em|rem)\b', ''),
    # RGB/hex color codes that might render: "#4A90D9", "rgb(74, 144, 217)"
    # (keep these but wrap in description — they're ok in prompts)
]


def _sanitize_prompt(prompt: str) -> str:
    """Remove formatting instructions that image models render as literal text.

    AI image generators (especially Gemini) sometimes render CSS/pixel values,
    coordinate specifications, and other formatting instructions as visible
    text in the output image. This function strips those patterns.
    """
    sanitized = prompt
    for pattern, replacement in _LEAK_PATTERNS:
        sanitized = re.sub(pattern, replacement, sanitized, flags=re.IGNORECASE)

    # Clean up double spaces left by removals
    sanitized = re.sub(r'  +', ' ', sanitized)
    # Clean up empty parentheses
    sanitized = re.sub(r'\(\s*\)', '', sanitized)

    return sanitized


# ──────────────────────────────────────────────────────────────────────────────
# Prompt templates
# ──────────────────────────────────────────────────────────────────────────────

# Targeted guidance for the readability-only side pass. Intentionally
# NOT injected into the main refinement loop — an earlier experiment
# this repo showed that steering the refiner toward readability while it is
# still fixing content can regress the overall win rate (empirically,
# showed this regressed quality. This block runs once, after the refinement loop has settled the
# content, when text_readability < threshold AND content_accuracy is already
# acceptable. See CraftSession._readability_polish().
READABILITY_FIX_GUIDANCE = """\
READABILITY is the weakest dimension. Focus your fixes on TYPOGRAPHY ONLY — do \
NOT remove components, do NOT shorten component names, do NOT restructure. The \
figure must keep the same faithful content. Only change:
- Increase font size of ALL text so every label is legible at column width (3.5"). \
  Labels inside boxes, arrow labels, axis labels, inset captions — all should grow.
- Add more whitespace between boxes and between rows of boxes. Loosen a cramped \
  layout; do NOT pack edge-to-edge.
- Fix any overlapping text or crossing arrows. Arrows should flow cleanly in one \
  primary direction (left-to-right OR top-to-bottom, not both).
- Ensure high contrast: dark text on a light background, no low-contrast pastel \
  text on pastel fill.
Keep the same components, same labels, same layout structure — just render the \
text bigger and the spacing more generous."""


POSTER_TEXT_FIX_GUIDANCE = """\
This is a CONFERENCE POSTER. Text fidelity at poster scale is the weakest link. \
Focus your fixes on:
- Every word in the poster should be a real, common English word. Avoid niche \
  technical terms in tiny text; prefer phrases the image model renders reliably.
- Section headers ("Motivation", "Method", "Results") should be short and bold.
- Replace paragraph-style body text with 3-5 bullet points, each 4-8 words.
- Key numbers/metrics as huge colored badges (e.g., "82% ↑"), not embedded in text.
- Keep the method diagram self-contained with minimal internal labels. Put \
  explanatory text OUTSIDE the diagram.
- For author/affiliation lines, keep to one line each; long strings garble."""


REFINE_SYSTEM_PROMPT_GEMINI = """\
Rewrite the prompt below to fix the critic's top 2-3 issues. Keep the same length or shorter.

OUTPUT FORMAT
- Begin DIRECTLY with the refined prompt on the next line. No preamble.
- Same prose style as input. NO JSON, NO ``` fences.

FIXES TO APPLY
- Address the critic's identified issues; ignore minor nitpicks.
- For each fix: describe the new visual state, not the change action.
  (Write "Encoder shown as a wide blue rounded box on the left" — not "Make encoder bigger".)
- Keep ALL existing paper components (faithfulness over brevity).

PRESERVE
- White background, sans-serif typography, generous whitespace.
- NO figure titles / numbers / paper titles in image.
- NO pixel sizes / coordinates / full equations.
- Short labels (5-10 words). No sentences inside boxes.
- Clean arrow routing.

OUTPUT START: write the refined prompt on the next line.
"""


REFINE_SYSTEM_PROMPT = """\
You are a world-class prompt engineer specializing in generating academic scientific \
figures using AI image generation models (like Gemini). Your job is to take a figure \
generation prompt that produced an imperfect result, analyze what went wrong based on \
a VLM critic's evaluation, and produce an IMPROVED prompt.

## CRITICAL ANTI-LEAK RULES (MUST FOLLOW):

**NEVER include any of the following in your prompt — the image model will \
render them as visible text in the figure:**
- Pixel dimensions (e.g., "180 x 50 px", "200x100", "24px")
- CSS/styling values (e.g., "border: 2px", "font-size: 12pt", "padding: 8px")
- Coordinate positions (e.g., "at position (100, 200)", "x=50, y=100")
- Implementation-level sizing (e.g., "box is 3cm wide", "arrow length 5cm")

**Instead, use relative/descriptive sizing:**
- "large box" instead of "200x100px box"
- "small label" instead of "12pt text"
- "generous spacing between components" instead of "margin: 20px"
- "wide landscape figure" instead of "1400x900 pixels"

## Rules for prompt improvement:

1. **Keep what works**: If the current prompt produced good structure, preserve it. \
Only change what the critic identified as problematic.

2. **Be specific but visual, not technical**: Describe what to SEE, not how to \
IMPLEMENT it. Say "a rounded blue box labeled 'Encoder'" not "a 180x50px rounded \
rectangle with border-radius: 8px and fill: #4A90D9 containing 12pt Arial text".

3. **Use CRITICAL annotations sparingly**: Mark only the 2-3 most important fixes \
with "CRITICAL:" prefix. Don't mark everything as critical.

4. **Keep labels concise but faithful**: Use exact paper terminology. No sentences \
inside boxes, but DO keep full component names. Faithfulness > conciseness. \
Replace equations with short names (e.g. "Loss L" not "L = Σ...").

5. **Focus on spatial relationships**: Describe layout in terms of relative position \
("to the right of", "below", "centered between") not coordinates.

6. **Describe data flow explicitly**: For pipeline figures, trace the exact path: \
"Input → Encoder (blue) → Features → Decoder (orange) → Output"

7. **Don't repeat failed approaches**: Check iteration history. If adding detail \
made things worse, try SIMPLIFYING instead.

8. **Keep prompt length STABLE or SHORTER**: If the current prompt is already long, \
make the refined version the SAME length or SHORTER. Longer prompts confuse the \
model and cause more artifacts. Aim for 80-200 lines maximum.

9. **Balance readability AND faithfulness**: Include ALL key components but make \
the diagram clean. Use generous whitespace, large readable text, and clear visual \
hierarchy. NEVER include figure titles, figure numbers, or paper titles in the image. \
A readable figure with all key components is the goal.

10. **Readability is CRITICAL**: The figure must be readable when printed at column \
width (3.5 inches). This means: \
- Labels must be SHORT (1-3 words maximum per label) \
- NO long descriptions or sentences inside boxes \
- Use LARGE font sizes — if in doubt, make text bigger \
- Generous spacing between all elements \
- Clean, uncluttered layout with clear visual hierarchy \
- Arrows should be clean with clear direction, not tangled
"""

REFINE_USER_PROMPT = """\
## Current Prompt (produced an imperfect result):
```
{current_prompt}
```

## Critic's Evaluation:
- Content accuracy: {content_accuracy}/10
- Layout quality: {layout_quality}/10
- Text readability: {text_readability}/10
- Aesthetic quality: {aesthetic_quality}/10
- Role match: {role_match}/10
- Artifact severity (10=clean): {artifact_severity}/10
- Overall: {overall}/10

### Issues identified:
{issues}

### Suggestions:
{suggestions}

## Figure Description:
{description}

## Style Guidelines (keep these):
{style_guidelines}

{components_section}

{history_section}

{user_feedback_section}

## Task:
Write an IMPROVED generation prompt. Focus on fixing the TOP 2-3 issues. \
Keep the prompt the SAME length or SHORTER than the current one. \
DO NOT add pixel dimensions, CSS values, or coordinate positions — \
the image model will render them as visible text. \
NEVER add figure titles, figure numbers, or paper titles to the image. \
All text must be LARGE and readable — no tiny text.

Balance faithfulness with visual quality — include key components but keep the layout clean and professional.

Return ONLY the improved prompt text.
"""


class PromptRefiner:
    """Refines generation prompts based on critique feedback."""

    def __init__(self, router: ModelRouter):
        self.router = router

    def create_initial_prompt(
        self,
        paper_context: str,
        description: str,
        figure_type: str,
        style_guidelines: str,
        venue: str = "neurips",
        visual_style: str = "",
    ) -> str:
        """Create the initial generation prompt from paper context."""
        visual_style_instruction = ""
        if visual_style:
            from crafter.generation.craft.venue_styles import VISUAL_STYLES
            vs = VISUAL_STYLES.get(visual_style)
            if vs:
                visual_style_instruction = (
                    f"\n\n## Visual Style: {vs['name']}\n"
                    f"IMPORTANT: Render this figure as a {vs['name']}. "
                    f"{vs['description']}.\n"
                    f"The figure should NOT be a generic block diagram unless "
                    f"that is the specified visual style."
                )

        # Inject poster-specific instructions when figure_type is "poster"
        poster_inst = POSTER_INSTRUCTIONS if figure_type == "poster" else ""

        # Auto-select Gemini-friendly system prompt when planner is Gemini.
        # The Opus-tuned prompt asks for elaborate behavior that Gemini-pro
        # follows poorly; the Gemini variant is more directive + ~half length.
        _planner = (self.router.config.planner_model or "").lower()
        sys_prompt = (
            INITIAL_PROMPT_SYSTEM_GEMINI
            if "gemini" in _planner
            else INITIAL_PROMPT_SYSTEM
        )

        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": INITIAL_PROMPT_USER.format(
                paper_context=paper_context[:10000],
                description=description,
                figure_type=figure_type,
                style_guidelines=style_guidelines,
                venue=venue,
                visual_style_instruction=visual_style_instruction,
                poster_instructions=poster_inst,
            )},
        ]

        prompt = self.router.plan(messages, temperature=0.7, max_tokens=16000)
        prompt = _sanitize_prompt(prompt)

        # Guard: if prompt is empty or very short, retry with fallback model
        if len(prompt) < 100:
            logger.warning(f"Prompt too short ({len(prompt)} chars), retrying with fallback...")
            try:
                prompt = self.router._chat(
                    messages, model="gemini-3.1-pro-preview",
                    temperature=0.7, max_tokens=16000,
                )
                prompt = _sanitize_prompt(prompt)
            except Exception:
                pass

        # Final fallback: generate a minimal prompt from description
        if len(prompt) < 100:
            logger.warning("Prompt generation failed, using description as fallback")
            prompt = (
                f"Generate a clean academic diagram for a NeurIPS paper.\n"
                f"White background, flat design, pastel colors, short labels.\n\n"
                f"{description[:3000]}"
            )

        logger.info(f"Initial prompt generated ({len(prompt)} chars)")
        return prompt

    def create_variant_prompts(
        self,
        paper_context: str,
        description: str,
        figure_type: str,
        venue: str,
        visual_styles: list[str],
        style_guidelines_map: dict[str, str],
    ) -> dict[str, str]:
        """Create initial prompts for multiple visual style variants."""
        prompts = {}
        for vs in visual_styles:
            guidelines = style_guidelines_map.get(vs, "")
            prompt = self.create_initial_prompt(
                paper_context=paper_context,
                description=description,
                figure_type=figure_type,
                style_guidelines=guidelines,
                venue=venue,
                visual_style=vs,
            )
            prompts[vs] = prompt
        return prompts

    def refine(
        self,
        current_prompt: str,
        critique: Any,  # CritiqueResult
        paper_context: str,
        description: str,
        style_guidelines: str,
        iteration_history: Optional[list] = None,
        user_feedback: Optional[str] = None,
        must_preserve_components: Optional[list[str]] = None,
    ) -> str:
        """Refine a generation prompt based on critique feedback."""
        # Build history section
        history_section = ""
        if iteration_history and len(iteration_history) > 1:
            history_parts = ["## Iteration History (avoid repeating failed approaches):"]
            for it in iteration_history[:-1]:
                crit = it.get("critique")
                if crit:
                    score_str = f"overall={crit.overall:.1f}"
                    issue_str = "; ".join(crit.issues[:2]) if crit.issues else "none"
                    history_parts.append(
                        f"- Iteration {it.get('iteration', '?')}: "
                        f"{score_str}, issues: {issue_str}"
                    )
            history_section = "\n".join(history_parts)

        user_feedback_section = ""
        if user_feedback:
            user_feedback_section = f"## User Feedback:\n{user_feedback}"

        # Truncate current prompt if too long to avoid context overflow
        current_prompt_display = current_prompt
        if len(current_prompt) > 6000:
            current_prompt_display = current_prompt[:6000] + "\n... [truncated]"

        # Build component awareness section (soft guide, not hard constraint)
        components_section = ""
        if must_preserve_components:
            components_section = (
                "\n## Key Components (try to include with exact names):\n"
                + ", ".join(must_preserve_components[:12])
                + "\n"
            )

        user_prompt = REFINE_USER_PROMPT.format(
            current_prompt=current_prompt_display,
            content_accuracy=critique.content_accuracy,
            layout_quality=critique.layout_quality,
            text_readability=critique.text_readability,
            aesthetic_quality=critique.aesthetic_quality,
            role_match=critique.role_match,
            artifact_severity=critique.artifact_severity,
            overall=critique.overall,
            issues="\n".join(f"- {i}" for i in critique.issues),
            suggestions="\n".join(f"- {s}" for s in critique.suggestions),
            description=description,
            style_guidelines=style_guidelines[:1000],
            history_section=history_section,
            user_feedback_section=user_feedback_section,
            components_section=components_section,
        )

        # Auto-select Gemini-friendly system prompt when refiner is Gemini.
        _refiner = (self.router.config.refiner_model or "").lower()
        refine_sys = (
            REFINE_SYSTEM_PROMPT_GEMINI
            if "gemini" in _refiner
            else REFINE_SYSTEM_PROMPT
        )

        messages = [
            {"role": "system", "content": refine_sys},
            {"role": "user", "content": user_prompt},
        ]

        refined = self.router.refine_prompt(messages, temperature=0.7, max_tokens=16000)

        # Strip markdown code blocks
        refined = refined.strip()
        if refined.startswith("```"):
            lines = refined.split("\n")
            refined = "\n".join(lines[1:])
            if refined.endswith("```"):
                refined = refined[:-3]
            refined = refined.strip()

        # Apply anti-leak sanitization
        refined = _sanitize_prompt(refined)

        # Warn if prompt grew significantly (sign of prompt bloat)
        if len(refined) > len(current_prompt) * 1.5:
            logger.warning(
                f"Refined prompt grew significantly: {len(current_prompt)} → {len(refined)} chars. "
                f"May cause more artifacts."
            )

        logger.info(f"Prompt refined ({len(refined)} chars)")
        return refined


INITIAL_PROMPT_SYSTEM_GEMINI = """\
Write an 80-200 line image-generation prompt for a publication-quality academic figure.

OUTPUT FORMAT
- Begin DIRECTLY with the prompt text on the next line. No "Here is...", no preamble.
- Plain prose with optional bullet/numbered lists. NO JSON, NO ``` fences.
- 80-200 lines total.

REQUIRED CONTENT (must include all 7 items)
1. Overall layout: panel count, orientation, primary flow direction (left-to-right / top-down / radial / multi-column / cyclic).
2. Background: WHITE or very light. Generous whitespace throughout.
3. Color palette: 5-7 specific pastel/professional colors by NAME (e.g. "soft coral", "mint green", "sky blue", "warm cream", "lavender"). NO vague words like "professional pastels".
4. Typography: clean sans-serif, LARGE readable labels with high contrast on background. No tiny text.
5. Component-by-component: every paper component listed by its EXACT name from the paper. For each component: shape, fill color, label text, relative position. Include ALL key components — faithfulness over cleanness.
6. Connections: arrow direction, line style (solid / dashed / curved), label naming what data flows.
7. Visual aids where relevant: example thumbnails for inputs/outputs, color-coded blocks for feature maps, icon representations of real-world objects.

FORBIDDEN
- Figure titles / figure numbers / paper titles inside the image
- Pixel dimensions / CSS values / coordinate positions (use "large box", "narrow strip", "centered", "to the right of")
- Full math equations (use "Loss L", "Attention(Q, K, V)", not "L = -log p(y|x)")
- Long sentences inside component boxes (max 5-10 word labels)
- Dark / black backgrounds
- Crossing or chaotic arrows
"""


INITIAL_PROMPT_SYSTEM = """\
You are a world-class prompt engineer for AI image generation. Your task is to \
create a detailed, structured prompt that will generate a publication-quality \
academic figure.

Describe the EXACT visual style in concrete terms: colors, shapes, layout, spacing. \
The image model needs specific visual instructions, not abstract style names.

## STYLE RULES:
- Professional pastel colors on white background, generous whitespace
- Rounded rectangles for modules, with purposeful color coding
- Include visual elements where relevant: thumbnail images for inputs/outputs, \
  colored grids for feature maps, small representative examples
- Clean arrows with clear data flow direction
- Consistent sans-serif font throughout
- NO figure titles, NO figure numbers, NO paper titles in the image

## CONTENT RULES:

1. **NEVER include pixel dimensions, CSS values, or coordinate positions** in \
your prompt. The image model will render them as visible text. \
Say "large rounded box" not "180x50px box". Say "generous spacing" not "20px margin".

2. **Keep text labels concise**: Use exact paper terminology for labels. \
No sentences or paragraphs inside boxes — only component names and short phrases. \
Do NOT oversimplify labels — faithfulness to the paper is the top priority.

3. **NEVER include figure titles, figure numbers, or paper titles** in the prompt. \
The diagram should start directly with boxes and arrows. No "Figure 1:" text.

4. **No full equations** in the diagram. Replace formulas like "L = Σ..." with \
short names like "Loss L". Equations rendered in images are unreadable.

5. **All text must be LARGE and READABLE**. Never include text so small it can't be read. \
Use generous whitespace between components. Avoid overlapping text or cramped layouts.

6. **Readability rules** (violation = readability veto):
   - WHITE or very light background only (no black/dark backgrounds)
   - No visual noise or overlapping elements
   - Clear routing — arrows should not cross chaotically
   - High contrast text on background
   - No illegible tiny fonts

7. **Describe visually, not technically**: Say what to SEE, not how to implement it.

8. Write a prompt that is:
   - Self-contained (includes all visual details)
   - Specific about colors, shapes, and relative layout
   - Describes data flow explicitly for pipeline/architecture figures
   - Uses descriptive sizing ("wide landscape figure", "small icon", "large title")
   - Follows the provided style guidelines
   - Adapts to the requested visual style

9. **Include ALL components from the paper** with their EXACT names. \
Faithfulness is the top priority — missing key components is worse than a slightly \
cluttered layout.

10. Keep the prompt between 80-200 lines. Longer prompts cause more artifacts.

Return ONLY the prompt text, no explanation.
"""

INITIAL_PROMPT_USER = """\
## Paper Context:
{paper_context}

## Figure Description:
{description}

## Figure Type: {figure_type}
## Target Venue: {venue}

## Style Guidelines:
{style_guidelines}
{visual_style_instruction}

Create a detailed generation prompt for this figure. Be specific about:
- Background color (WHITE for academic figures)
- Each component: shape, color, label using EXACT paper terminology, relative position
- Connections/arrows between components with direction
- Typography style (sans-serif, clean, LARGE readable text)
- Visual elements where relevant (thumbnail images, colored grids, icons)

CRITICAL:
- Do NOT include pixel dimensions, CSS values, font sizes in pt/px, \
or coordinate positions. Use descriptive terms instead.
- Do NOT include any figure title, figure number, or paper title in the diagram.
- Do NOT include full equations — use short names instead.
- Include ALL key components from the paper with their EXACT names.
- Faithfulness to the paper is the TOP priority.

{poster_instructions}

The prompt should be 80-200 lines.
"""

POSTER_INSTRUCTIONS = """\
## POSTER-SPECIFIC INSTRUCTIONS (this is a conference poster, NOT a paper figure):

This prompt is for generating a CONFERENCE POSTER. The poster is a single large \
landscape image that summarizes an entire paper. It is NOT a single method diagram.

### POSTER STRUCTURE (follow this layout):
1. TOP BANNER (full width): Paper title in very large bold text. Below: author names \
   and affiliations in smaller but still readable text. Conference badge in top-right corner.

2. MAIN BODY (3 equal columns):
   - Column 1 (Motivation & Key Idea): Problem statement with icons/bullets, \
     key challenges, and the proposed solution — use visual elements not paragraphs.
   - Column 2 (Method): Architecture/pipeline diagram as the centerpiece. \
     Training procedure summary. Keep it visual — the diagram should dominate.
   - Column 3 (Results): Key quantitative results displayed as large numbers in \
     colored boxes. A comparison chart or table. Qualitative examples if applicable.

3. BOTTOM STRIP (full width): Key takeaway sentence in a highlighted box.

### POSTER DESIGN RULES:
- Follow the orientation specified in the description (portrait=taller than wide, landscape=wider than tall)
- For portrait posters: use 2-column layout with stacked rows of content
- For landscape posters: use 3-column layout side by side
- Each section has a colored header bar and a light-tinted background
- Generous whitespace — poster is viewed from several feet away
- Key numbers/metrics: display them EXTRA LARGE in colored badges
- Bullet points and icons preferred over paragraphs
- Architecture diagram should be clean and self-contained
- Author affiliations should use university names, not abbreviations
- NO dense paragraphs of text — if you need to explain, use bullet points
- NO reference lists, NO figure numbers
"""
