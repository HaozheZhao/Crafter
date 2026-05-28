"""paper understanding, grounding, role planning,
reference search, and visual-style selection.

paper_reader: runs PaperReader to extract component names,
then domain-visual hint enrichment.

figure_spec / VisualGrounder: builds the EvolvingFigureSpec
when use_figure_spec=True; runs VisualGrounder to extract
paper-specific concrete elements.

RolePlanner: for non-academic roles, generate a structural
layout blueprint. Skipped for academic / "" role to preserve R1
BananaBench numbers.

research: load user-provided references + Serper search.

visual style selection: map (visual_style|role|venue) →
visual_styles list.
"""
from __future__ import annotations

import concurrent.futures
import logging
import os
from typing import TYPE_CHECKING, Optional

from rich.console import Console

from crafter.generation.craft.venue_styles import (
    VISUAL_STYLES, get_recommended_visual_styles,
)

# paper_reader timeout (s). The PaperReader runs up to 3 LLM
# passes; on samples with very large paper_full_text (~200KB), OPENROUTER
# occasionally hangs the chat-completions call indefinitely. Wrapping
# `read()` in a timeout lets us fall through to caption-only grounding
# instead of blocking the whole pipeline. Default 180s is comfortably
# above the typical 30-90s healthy-path time.
_PAPER_READER_TIMEOUT_S = int(os.environ.get("PAPER_READER_TIMEOUT_S", "180"))

if TYPE_CHECKING:
    from crafter.generation.craft.session import CraftSession, CraftInput

logger = logging.getLogger(__name__)
console = Console()


def run_paper_understanding(
    session: "CraftSession",
    craft_input: "CraftInput",
) -> tuple[list[str], str, Optional[object]]:
    """Returns (component_names, enriched_description,
    figure_spec). Mutates session: sets session._component_names,
    session._claims, session._visual_grounding, session._figure_spec
    (when use_figure_spec=True)."""
    component_names: list[str] = []
    paper_ctx = None
    try:
        # Wrap paper_reader.read in a hard timeout — daemonized worker so
        # a hung OPENROUTER call doesn't block forever. Existing `except
        # Exception` block catches TimeoutError + falls through to
        # caption-only.
        _ex = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        try:
            _fut = _ex.submit(
                session.paper_reader.read,
                caption=craft_input.description,
                category=craft_input.figure_type,
                raw_text=craft_input.paper_text,
                max_passes=3,
            )
            paper_ctx = _fut.result(timeout=_PAPER_READER_TIMEOUT_S)
        finally:
            _ex.shutdown(wait=False)
        all_components = [
            c["name"] for c in paper_ctx.components
            if isinstance(c, dict) and "name" in c
        ]
        if len(all_components) > 10:
            # Keep first 10 (PaperReader returns them in importance order)
            component_names = all_components[:10]
            console.print(
                f"  Extracted {len(all_components)} components, "
                f"prioritizing top {len(component_names)}: "
                f"{', '.join(component_names[:8])}"
            )
        else:
            component_names = all_components
            console.print(
                f"  Extracted {len(component_names)} components: "
                f"{', '.join(component_names[:8])}"
            )
        session.history.add(
            "paper_reader", "extract",
            f"{len(component_names)}/{len(all_components)} components",
            score=8.0 if component_names else 4.0,
        )
    except Exception as e:
        logger.warning(
            f"PaperReader failed: {e}, continuing without component extraction"
        )
        console.print(
            "  [yellow]Paper reading failed, continuing without components[/yellow]"
        )
        session.history.add(
            "paper_reader", "extract", f"Failed: {e}", score=3.0,
        )

    session._component_names = component_names
    # Expose the structured PaperContext to downstream phases so the
    # Designer / VisualGrounder read from the harness's own
    # intent-reasoner output (method_summary, components, connections,
    # equations, concrete_examples) rather than raw paper_text. This is
    # the spec the rest of the harness operates over.
    session._paper_ctx = paper_ctx

    # Build enriched description with domain-specific visual suggestions.
    enriched_description = craft_input.description
    if component_names:
        enriched_description += f"\n\nKey components: {', '.join(component_names[:10])}"
    visual_hints = session._suggest_domain_visuals(
        craft_input.paper_text, craft_input.description, craft_input.figure_type,
    )
    if visual_hints:
        enriched_description += f"\n\n{visual_hints}"

    # build EvolvingFigureSpec for coordinated agent edits.
    figure_spec = None
    if getattr(session.config, "use_figure_spec", False):
        from crafter.generation.craft.figure_spec import (
            EvolvingFigureSpec, PanelLayout, StyleSpec,
        )
        figure_spec = EvolvingFigureSpec()
        figure_spec.set_style(StyleSpec(
            palette="muted pastel",
            typography="sans-serif (Arial/Helvetica)",
            background="pure white",
        ))
        figure_spec.set_layout(PanelLayout(
            panel_count=1, arrangement="single",
        ))
        # Propagate refer_image_role from CraftInput to the spec so
        # render_image_gen_prompt emits the matching hint.
        figure_spec.refer_image_role = getattr(craft_input, "refer_image_role", "") or ""
        session._figure_spec = figure_spec

    # VisualGrounder
    session._visual_grounding = None
    # Feed VG the PaperReader's method_summary when available. VG's hard
    # truncate on raw paper_text means long arxiv-style sources can have
    # VG seeing only intro / related-works (not methodology); the
    # focused method_summary avoids that. Falls back to raw paper_text
    # when the reader returned empty / failed.
    _vg_paper_text = craft_input.paper_text
    if paper_ctx is not None and paper_ctx.method_summary:
        _vg_paper_text = paper_ctx.method_summary
    if session.visual_grounder is not None:
        try:
            if figure_spec is not None:
                spec_elements = session.visual_grounder.extract_to_spec(
                    paper_text=_vg_paper_text,
                    caption=craft_input.description,
                    figure_type=craft_input.figure_type or "",
                )
                figure_spec.set_elements(spec_elements)
                if spec_elements:
                    console.print(
                        f"  VisualGrounder→spec: {len(spec_elements)} elements → "
                        f"{', '.join(e.name for e in spec_elements[:4])}"
                        + (" ..." if len(spec_elements) > 4 else "")
                    )
                    session.history.add(
                        "visual_grounder", "extract→spec",
                        f"{len(spec_elements)} elements", score=8.0,
                    )
            else:
                grounding = session.visual_grounder.extract(
                    paper_text=_vg_paper_text,
                    caption=craft_input.description,
                    figure_type=craft_input.figure_type or "",
                )
                session._visual_grounding = grounding
                block = grounding.to_prompt_block()
                if block:
                    enriched_description += f"\n\n{block}"
                    console.print(
                        f"  VisualGrounder: {len(grounding.elements)} concrete "
                        f"elements → "
                        f"{', '.join(e.name for e in grounding.elements[:4])}"
                        + (" ..." if len(grounding.elements) > 4 else "")
                    )
                    session.history.add(
                        "visual_grounder", "extract",
                        f"{len(grounding.elements)} concrete elements",
                        score=8.0 if grounding.elements else 5.0,
                    )
        except Exception as e:
            logger.warning(
                f"VisualGrounder failed: {e}; continuing without grounding"
            )
            session.history.add(
                "visual_grounder", "extract", f"Failed: {e}", score=3.0,
            )

    # RolePlanner blueprint for non-academic roles.
    role_lc = (craft_input.role or "").lower().strip()
    if role_lc not in ("", "academic"):
        try:
            blueprint = session.role_planner.plan(
                paper_text=craft_input.paper_text,
                caption=craft_input.description,
                role=craft_input.role,
            )
            if blueprint:
                enriched_description += f"\n\n{blueprint}"
                console.print(
                    f"  RolePlanner: blueprint generated for role='{craft_input.role}' "
                    f"({len(blueprint)} chars)"
                )
                session.history.add(
                    "role_planner", "blueprint",
                    f"role={craft_input.role} len={len(blueprint)}",
                    score=8.0,
                )
        except Exception as e:
            logger.warning(
                f"RolePlanner failed for role={craft_input.role!r}: {e}"
            )
            session.history.add(
                "role_planner", "blueprint", f"Failed: {e}", score=3.0,
            )

    return component_names, enriched_description, figure_spec


_STYLE_PICKER_PROMPT = """\
You pick the visual-style approach(es) for an academic figure.

You have a fixed VOCABULARY of style keys. Pick the most relevant subset
for this paper's figure intent — typically 1-3 keys.

Available style vocabulary (key → name):
  block_diagram → Block Diagram / Pipeline
  conceptual_illustration → Conceptual Illustration
  infographic → Infographic / Visual Summary
  flowchart → Flowchart / Decision Tree
  multi_panel → Multi-Panel Figure
  comparison_grid → Comparison Grid / Table Figure
  annotated_diagram → Annotated Technical Diagram
  timeline → Timeline / Process Steps
  data_visualization → Data Visualization / Chart
  equation_figure → Equation / Mathematical Figure
  research_poster → Research Conference Poster

Selection guidance (decide based on the figure's nature, not by formula):
- Single clear method → 1 variant of the best-fitting style
- Multi-stage method or multi-aspect comparison → 2-3 variants of
  complementary styles (different angles on same content)
- Highly ambiguous / unclear figure intent → 3 variants exploring
  different framings
- For edit tasks (refer image attached as inpaint base / element layout
  / sketch), the FINAL output is constrained by the refer; 1 variant
  is usually sufficient because variants on the same refer base don't
  diversify outcome much.

Output STRICT JSON (no prose, no markdown):
{
  "selected": ["<style_key>", ...],   // 1-3 keys from the vocabulary
  "reason": "<≤30 words>"
}
"""


def pick_variant_styles(
    session: "CraftSession",
    craft_input: "CraftInput",
    num_variants_cap: int,
) -> list[str]:
    """Model-decided variant style selection. Asks a quick LLM call to pick
    1-3 style keys from VISUAL_STYLES vocabulary based on figure intent.
    Returns picked keys clipped to num_variants_cap. Falls back to
    venue-recommended on LLM failure.

    Skipped (caller falls through) when role is poster/infographic — those
    are role-locked to specific style keys (research_poster / infographic).
    """
    user = (
        f"## Figure caption\n{craft_input.description[:300]}\n\n"
        f"## Paper content excerpt\n{(craft_input.paper_text or '')[:1200]}\n\n"
        f"## Role\n{craft_input.role or 'academic'}\n"
        f"## Task\n{craft_input.figure_type or 't2i'}\n"
        f"## Edit-mode\n{'yes — refer image attached' if getattr(craft_input, 'refer_image_role', '') else 'no'}\n\n"
        f"Pick the variant styles. Return JSON only."
    )
    try:
        data = session.router.chat_json(
            messages=[
                {"role": "system", "content": _STYLE_PICKER_PROMPT},
                {"role": "user", "content": user},
            ],
            temperature=0.2,
        )
        picked = data.get("selected", []) or []
        # Filter to known vocabulary keys
        picked = [k for k in picked if k in VISUAL_STYLES][:num_variants_cap]
        if not picked:
            raise ValueError("style picker returned no valid keys")
        reason = str(data.get("reason", ""))[:80]
        console.print(
            f"  Style picker: {', '.join(picked)} — {reason}"
        )
        return picked
    except Exception as e:
        logger.debug(f"Style picker failed ({e}); using venue-recommended fallback")
        return list(get_recommended_visual_styles(craft_input.venue))[:num_variants_cap]


def select_visual_styles(
    craft_input: "CraftInput",
    *,
    num_variants: int,
    session: "CraftSession" = None,
) -> list[str]:
    """Pick visual_styles list. Order:
       1. caller-set visual_style override → just that one
       2. role poster/infographic → role-locked keys
       3. otherwise → model-decided pick (when session+router available),
          else venue-recommended fallback
    """
    if craft_input.visual_style:
        console.print(f"  Using specified style: {craft_input.visual_style}")
        return [craft_input.visual_style]
    if craft_input.role == "poster" or craft_input.figure_type == "poster":
        # R2: poster role → research_poster. The role-conditional preamble
        # in figure_spec.render_image_gen_prompt provides the structural brief;
        # this style key picks the matching skill / build_style_prompt branch.
        styles = ["research_poster"]
        if num_variants > 1:
            styles.append("infographic")
        console.print(f"  Poster mode: using {', '.join(styles)}")
        return styles
    if craft_input.role == "infographic":
        # R2: infographic → infographic style key. Single variant — no
        # obvious second-best.
        styles = ["infographic"]
        console.print(f"  Infographic mode: using {', '.join(styles)}")
        return styles
    # Stage F design directive: hardcoded VISUAL_STYLES is the vocabulary;
    # final pick + count is model-decided per-sample. The picker can return
    # 1-3 keys clipped to num_variants config cap.
    if session is not None:
        picked = pick_variant_styles(session, craft_input, num_variants)
        if picked:
            console.print(f"  Exploring {len(picked)} styles: {', '.join(picked)}")
            return picked
    recommended = get_recommended_visual_styles(craft_input.venue)
    styles = recommended[:num_variants]
    console.print(f"  Fallback ({len(styles)} venue-recommended styles): {', '.join(styles)}")
    return styles
