"""parallel style-variant generation + critic evaluation.

For each visual style, build the initial prompt (edit-mode short brief
or T2I full narrative), call the image generator (Flash → Pro fallback),
and run the critic. Variants are generated in parallel via a
ThreadPoolExecutor (capped at 3 workers).

Returns a list of (style, image_path, prompt, critique) tuples plus the
style→path map. Empty list when ALL variants failed.
"""
from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from rich.console import Console

from crafter.generation.craft.venue_styles import VISUAL_STYLES, build_style_prompt

if TYPE_CHECKING:
    from crafter.generation.craft.session import CraftSession, CraftInput

logger = logging.getLogger(__name__)
console = Console()


def run_variant_generation(
    session: "CraftSession",
    craft_input: "CraftInput",
    *,
    visual_styles: list[str],
    enriched_description: str,
    figure_spec,  # Optional[EvolvingFigureSpec]
    reference_images: list[bytes],
    reference_paths: list[str],
    run_dir: Path,
) -> tuple[list[tuple], dict[str, str]]:
    """Returns (variant_results, variant_images). variant_results is a
    list of (vs_key, image_path, prompt, critique). Empty list when all
    variant generations failed."""
    variant_results: list[tuple] = []
    variant_images: dict[str, str] = {}

    # Pre-compute desc_for_initial once (same for every variant)
    intent_preamble = (
        getattr(session, "_intent_preamble", "")
        or getattr(session.config, "_intent_preamble", "")
    )
    if figure_spec is not None and figure_spec.required_elements:
        from crafter.generation.craft.figure_spec import render_image_gen_prompt
        spec_block = render_image_gen_prompt(
            figure_spec, craft_input.paper_text, craft_input.description,
            intent_preamble=intent_preamble,
            role=craft_input.role,
        )
        desc_for_initial = f"{spec_block}\n\n{enriched_description}"
    elif intent_preamble:
        desc_for_initial = f"{intent_preamble}\n\n{enriched_description}"
    else:
        # No spec, no intent preamble — but role-conditional preamble may
        # still be needed for poster / infographic.
        from crafter.generation.craft.figure_spec import _role_preamble
        rp = _role_preamble(craft_input.role)
        desc_for_initial = (
            f"{rp}\n\n{enriched_description}" if rp else enriched_description
        )

    def _generate_one_variant(vs_key: str) -> Optional[tuple]:
        """Generate + critique one variant. Pure function (no shared state
        mutation), safe to run in parallel across variants."""
        vs_name = VISUAL_STYLES.get(vs_key, {}).get("name", vs_key)
        # Refer-aware prompt construction triggers on `refer_image_role`
        # being set; otherwise we build a from-scratch T2I narrative via
        # PromptRefiner.create_initial_prompt.
        use_edit_prompt = bool(getattr(craft_input, "refer_image_role", ""))
        if use_edit_prompt:
            # Short, focused edit instruction. Both the /v1/images/edits
            # endpoint and the chat-completions multimodal path benefit
            # from the refer-aware prompt over a from-scratch narrative.
            #
            # enriched_description carries the VMT translator's visual
            # narrative when VMT fired. For edit tasks (inpaint /
            # keyelems / sketch) the VMT translator emits a task-aware
            # narrative (e.g. "describe ONLY the fill, in the style of
            # the surrounding image") — feed it to build_edit_instruction
            # so the translator output reaches the image-gen prompt.
            from crafter.generation.craft.figure_spec import build_edit_instruction
            spec_for_prompt = None
            prompt = build_edit_instruction(
                task=craft_input.figure_type,
                style=craft_input.role,
                description=enriched_description,
                spec=spec_for_prompt,
            )
            style_prefix = ""
        else:
            style_prefix = build_style_prompt(
                craft_input.venue, craft_input.figure_type, vs_key,
            )
            # Feed the planner PaperReader's structured for_planner()
            # output instead of raw paper_text. On long arxiv-style
            # sources, raw text gets truncated before reaching the
            # methodology section; for_planner() surfaces method_summary
            # + components + connections as a compact block.
            _paper_ctx = getattr(session, "_paper_ctx", None)
            paper_context = (
                _paper_ctx.for_planner(15000) if _paper_ctx is not None
                else craft_input.paper_text
            )
            prompt = session.refiner.create_initial_prompt(
                paper_context=paper_context,
                description=desc_for_initial,
                figure_type=craft_input.figure_type,
                style_guidelines=style_prefix,
                venue=craft_input.venue,
                visual_style=vs_key,
            )
        (run_dir / f"prompt_variant_{vs_key}.txt").write_text(
            prompt, encoding="utf-8",
        )

        image_bytes = session.generator.generate(
            prompt=prompt,
            reference_images=reference_images,
            style_prefix=style_prefix,
            debug_dir=str(run_dir),
        )
        if image_bytes is None:
            image_bytes = session.generator.generate(
                prompt=prompt,
                reference_images=reference_images,
                style_prefix=style_prefix,
                model=session.config.generator_model,
                debug_dir=str(run_dir),
            )
        if image_bytes is None:
            logger.warning(f"Variant {vs_name} generation failed")
            return None

        image_path = str(run_dir / f"variant_{vs_key}.png")
        Path(image_path).write_bytes(image_bytes)
        # Feed the critic the required-component checklist so
        # faithfulness scoring is paper-specific (per-component presence
        # check) instead of a generic "does it look like the method?".
        _paper_ctx_eval = getattr(session, "_paper_ctx", None)
        _required_components_eval = (
            _paper_ctx_eval.for_critic() if _paper_ctx_eval is not None else ""
        )
        critique = session.critic.evaluate(
            image_path=image_path,
            prompt=prompt,
            paper_context=craft_input.paper_text,
            description=craft_input.description,
            venue=craft_input.venue,
            visual_style=vs_key,
            reference_paths=reference_paths,
            role=craft_input.role,
            task=craft_input.figure_type,
            required_components=_required_components_eval,
        )
        return (vs_key, image_path, prompt, critique)

    max_parallel = min(len(visual_styles), 3)
    console.print(
        f"  [dim]Spawning {max_parallel} parallel variant workers...[/dim]"
    )
    with ThreadPoolExecutor(max_workers=max_parallel) as ex:
        futures = {ex.submit(_generate_one_variant, vs): vs for vs in visual_styles}
        for fut in as_completed(futures):
            vs_key = futures[fut]
            vs_name = VISUAL_STYLES.get(vs_key, {}).get("name", vs_key)
            try:
                result = fut.result()
            except Exception as e:
                logger.exception(f"Variant {vs_name} failed: {e}")
                result = None
            if result is None:
                console.print(f"  [red]Variant {vs_name} skipped[/red]")
                continue
            _, image_path, _, critique = result
            variant_images[vs_key] = image_path
            variant_results.append(result)
            # History writes here (single-threaded) so no race.
            session.history.add(
                "drawer", f"variant:{vs_name}", score=critique.overall,
            )
            session.history.add(
                "critic", f"evaluate:{vs_name}", score=critique.overall,
                detail="; ".join(critique.issues[:2]) if critique.issues else "",
            )
            console.print(
                f"  [green]✓ {vs_name}[/green] (score={critique.overall:.1f})"
            )
            if critique.issues:
                for issue in critique.issues[:3]:
                    console.print(f"    [dim]issue: {issue[:90]}[/dim]")

    return variant_results, variant_images
