"""Readability-only polish pass.

Runs at most once after the refinement loop has settled content, when the current
best figure scores high on content but low on readability. The fix
guidance is typography-only — it forbids component removal or layout
restructuring — so content cannot regress. The polish output is
accepted iff:
  - text_readability strictly improved, AND
  - content_accuracy did not fall below `readability_polish_content_min`, AND
  - overall did not drop by more than 0.5.

Note: injecting readability guidance
into every refinement regresses overall — isolating the lever to one
dedicated pass is the documented fix.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from rich.console import Console

if TYPE_CHECKING:
    from crafter.generation.craft.session import CraftSession, CraftInput

logger = logging.getLogger(__name__)
console = Console()


def run_readability_polish(
    session: "CraftSession",
    *,
    best_image_path: str,
    best_score: float,
    current_prompt: str,
    style_prefix: str,
    reference_images: list,
    reference_paths: list,
    best_style: str,
    enriched_description: str,
    craft_input: "CraftInput",
    run_dir: Path,
    iterations: list,
) -> tuple[str, float]:
    """Returns (image_path, overall_score) — either the polish output if
    all gates pass, or the original best on rejection."""
    from crafter.generation.craft.session import CraftIteration  # late import to avoid cycle

    best_iter = next(
        (it for it in iterations if it.image_path == best_image_path and it.critique),
        None,
    )
    if best_iter is None:
        return best_image_path, best_score
    best_critique = best_iter.critique

    cfg = session.config
    READABILITY_THRESHOLD = getattr(cfg, "readability_polish_threshold", 7.5)
    CONTENT_PREREQ = getattr(cfg, "readability_polish_content_min", 5.5)
    if not getattr(cfg, "enable_readability_polish", True):
        return best_image_path, best_score
    if best_critique.text_readability >= READABILITY_THRESHOLD:
        return best_image_path, best_score
    if best_critique.content_accuracy < CONTENT_PREREQ:
        # Content is still broken; fixing readability now would just cement
        # a content error. Skip and let the Quality Guard run.
        return best_image_path, best_score

    from crafter.generation.craft.prompt_refiner import (
        READABILITY_FIX_GUIDANCE, POSTER_TEXT_FIX_GUIDANCE,
    )
    guidance = (
        POSTER_TEXT_FIX_GUIDANCE
        if craft_input.figure_type == "poster"
        else READABILITY_FIX_GUIDANCE
    )

    console.print(
        f"\n[bold cyan]Readability Polish[/bold cyan] "
        f"(text={best_critique.text_readability:.1f}, content={best_critique.content_accuracy:.1f})"
    )

    try:
        polish_prompt = session.refiner.refine(
            current_prompt=current_prompt,
            critique=best_critique,
            paper_context=craft_input.paper_text,
            description=enriched_description,
            style_guidelines=style_prefix,
            iteration_history=[],
            user_feedback=guidance,
            must_preserve_components=session._component_names,
        )
    except Exception as e:
        logger.warning(f"Readability polish refine failed: {e}")
        return best_image_path, best_score

    (run_dir / "prompt_polish.txt").write_text(polish_prompt, encoding="utf-8")

    try:
        image_bytes = session.generator.generate(
            prompt=polish_prompt,
            reference_images=reference_images,
            style_prefix=style_prefix,
            debug_dir=str(run_dir),
        )
    except Exception as e:
        logger.warning(f"generate failed: {e}")
        return best_image_path, best_score
    if not image_bytes:
        console.print("  [yellow]Polish generation returned nothing; keeping Phase-5 best[/yellow]")
        return best_image_path, best_score

    polish_path = str(run_dir / "polish.png")
    Path(polish_path).write_bytes(image_bytes)

    try:
        _paper_ctx_polish = getattr(session, "_paper_ctx", None)
        _req_components_polish = (
            _paper_ctx_polish.for_critic() if _paper_ctx_polish is not None else ""
        )
        polish_crit = session.critic.evaluate(
            image_path=polish_path,
            prompt=polish_prompt,
            paper_context=craft_input.paper_text,
            description=craft_input.description,
            venue=craft_input.venue,
            visual_style=best_style,
            reference_paths=reference_paths,
            role=craft_input.role,
            task=craft_input.figure_type,
            required_components=_req_components_polish,
        )
    except Exception as e:
        logger.warning(f"critique failed: {e}")
        return best_image_path, best_score

    readability_improved = polish_crit.text_readability > best_critique.text_readability
    content_ok = polish_crit.content_accuracy >= CONTENT_PREREQ
    overall_ok = polish_crit.overall >= best_score - 0.5

    iterations.append(CraftIteration(
        iteration=iterations[-1].iteration + 1 if iterations else 1,
        prompt=polish_prompt,
        image_path=polish_path,
        visual_style=best_style,
        critique=polish_crit,
        duration_seconds=0.0,
    ))

    if readability_improved and content_ok and overall_ok:
        console.print(
            f"  [green]Accepted polish: text {best_critique.text_readability:.1f}→"
            f"{polish_crit.text_readability:.1f}, "
            f"overall {best_score:.1f}→{polish_crit.overall:.1f}[/green]"
        )
        return polish_path, polish_crit.overall

    reasons = []
    if not readability_improved:
        reasons.append(f"text did not improve ({polish_crit.text_readability:.1f})")
    if not content_ok:
        reasons.append(f"content dropped to {polish_crit.content_accuracy:.1f}")
    if not overall_ok:
        reasons.append(f"overall dropped to {polish_crit.overall:.1f}")
    console.print(f"  [yellow]Rejected polish: {'; '.join(reasons)}[/yellow]")
    return best_image_path, best_score
