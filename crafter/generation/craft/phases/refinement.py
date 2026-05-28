"""Iterative refinement loop.

Up to `max_iter_cap` rounds of (critique → fix-guidance → refine prompt
→ regenerate → critic → AgentJudge stop-or-iterate). Includes:

- Early-exit when the variant is already content-accurate, role-matched
  and artifact-free (skipped for edit-mode and for non-academic T2I).
- C_PHASE5_ENABLE=0 disables the refinement loop entirely.
- SE→spec→prompt wiring under C_SE_SPEC_PASS=1 (Mechanism A).
- SE_ABLATE_P2B=1 disables the critic.issues append path.
- Edit-mode keeps the short /edits-track prompt structure
  (build_edit_instruction wrapper around description, not the Opus
  refiner path that would inflate it back to academic narrative).
- Stochastic-retry: when the critic score drops, retry once with the
  same prompt; accept the retry only if it improves.
- Per-iteration TextCritic OCR pass — flags text rendering issues that
  the next round's image-gen call repairs by regenerating from the
  spec-rendered prompt (no pixel patching).

Returns: best_image_path, best_score, current_prompt, iterations,
style_prefix, best_style, stop_reason.
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING

from rich.console import Console

from crafter.generation.craft.venue_styles import VISUAL_STYLES, build_style_prompt

if TYPE_CHECKING:
    from crafter.generation.craft.session import (
        CraftSession, CraftInput, CraftIteration, StopReason,
    )

logger = logging.getLogger(__name__)
console = Console()


def run_iter_refinement(
    session: "CraftSession",
    craft_input: "CraftInput",
    *,
    best_path: str,
    best_critique,
    current_prompt: str,
    best_style: str,
    best_vs_name: str,
    enriched_description: str,
    reference_images: list,
    reference_paths: list,
    figure_spec,
    run_dir: Path,
):
    """See module docstring."""
    from crafter.generation.craft.session import (
        CraftIteration, StopReason, _has_solid_artifact,
        compact_iteration_history,
    )

    style_prefix = build_style_prompt(
        craft_input.venue, craft_input.figure_type, best_style,
    )
    iterations: list[CraftIteration] = []
    best_image_path = best_path
    best_score = best_critique.overall
    stop_reason = StopReason.MAX_TURNS

    iterations.append(CraftIteration(
        iteration=0,
        prompt=current_prompt,
        image_path=best_path,
        visual_style=best_style,
        critique=best_critique,
        duration_seconds=0,
    ))

    max_iter_cap = min(
        max(1, craft_input.max_iterations),
        getattr(session.config, "agent_judge_max_iter", 3),
    )
    console.print(
        f"\n[bold cyan]Iterative Refinement ({best_vs_name}) "
        f"— up to {max_iter_cap} rounds, judge decides[/bold cyan]"
    )

    # Early-exit gates: model-decided via critic scores.
    role = (craft_input.role or "academic").lower()
    role_ok = role in ("", "academic") or best_critique.role_match >= 6.5
    early_exit_ok = (
        best_critique.content_accuracy >= 6.5
        and role_ok
        and not _has_solid_artifact(best_path)
    )
    if early_exit_ok:
        # Universal early-exit applies to all roles — model decides via
        # critic scores.
        console.print(
            f"  [green]Initial variant already ships (role='{role}': "
            f"content={best_critique.content_accuracy:.1f} ≥ 6.5, "
            f"role_match={best_critique.role_match:.1f} ≥ 6.5, no artifact). "
            f"Skipping refinement.[/green]"
        )
        max_iter_cap = 0
    elif role in ("poster", "infographic"):
        console.print(
            f"  [yellow]Non-academic T2I (role='{role}') below early-exit: "
            f"forcing ≥1 refinement iter for skill adaptation.[/yellow]"
        )
        max_iter_cap = max(max_iter_cap, 1)
    elif (best_critique.content_accuracy >= 6.5
            and not role_ok):
        console.print(
            f"  [yellow]Content OK but role_match too low "
            f"({best_critique.role_match:.1f} < 6.5 on role='{role}') — "
            f"running refinement to fix format/role mismatch.[/yellow]"
        )
    elif best_critique.issues:
        console.print("\n  [yellow]Issues to address:[/yellow]")
        for issue in best_critique.issues[:5]:
            console.print(f"    - {issue}")

    # Initialize per-session evolved skill state (test-time self-evolution).
    from crafter.generation.craft.skill_evolver import SkillState
    skill_state = SkillState()

    for i in range(max_iter_cap):
        iter_start = time.time()
        console.print(f"\n[bold]── Refinement {i + 1}/{max_iter_cap} ──[/bold]")

        last_critique = iterations[-1].critique
        last_image = iterations[-1].image_path

        # Test-time skill evolution.
        evolved_block = ""
        if session.skill_evolver is not None and last_critique:
            try:
                crit_text = "\n".join([
                    f"Overall: {last_critique.overall:.1f}/10",
                    f"Content accuracy: {getattr(last_critique, 'content_accuracy', 0):.1f}/10",
                    f"Text readability: {getattr(last_critique, 'text_readability', 0):.1f}/10",
                    "Issues: " + "; ".join(last_critique.issues[:5]),
                    "Suggestions: " + "; ".join(
                        getattr(last_critique, "suggestions", [])[:3]
                    ),
                ])
                if figure_spec is not None:
                    # structured edits to spec, no free-text addendum.
                    prev_history_len = len(figure_spec.history)
                    figure_spec = session.skill_evolver.evolve_spec(
                        spec=figure_spec,
                        critique_text=crit_text,
                        iteration=i,
                    )
                    new_edits = figure_spec.history[prev_history_len:]
                    if new_edits:
                        console.print(
                            f"  [cyan]SE→spec iter {i}: {len(new_edits)} structured "
                            f"edits → {', '.join(e.action for e in new_edits)}[/cyan]"
                        )
                        session.history.add(
                            "skill_evolver", f"evolve_spec:r{i}",
                            f"edits={len(new_edits)}: "
                            f"{', '.join(e.action for e in new_edits)}",
                            score=8.0,
                        )
                    evolved_block = ""
                else:
                    skill_state = session.skill_evolver.evolve(
                        state=skill_state,
                        caption=craft_input.description,
                        figure_type=craft_input.figure_type or "",
                        critique_text=crit_text,
                        iteration=i,
                    )
                    evolved_block = session.skill_evolver.to_prompt_block(skill_state)
                    if evolved_block:
                        console.print(
                            f"  [cyan]SkillEvolver iter {i}: addendum "
                            f"({len(skill_state.addendum)} chars)[/cyan]"
                        )
                        session.history.add(
                            "skill_evolver", f"evolve:r{i}",
                            f"addendum_len={len(skill_state.addendum)}",
                            score=8.0,
                        )
            except Exception as e:
                logger.warning(f"SkillEvolver iter {i} failed: {e}")

        # Directive critic (V) emits a multi-dim diagnostic; the
        # reviser (R, session.skill_evolver) converts it into typed
        # edits on the evolving spec S. The next round's prompt is
        # rendered from the updated S, not from accumulated free-text
        # amendments (§4.2.3).
        console.print("  Critiquing image & identifying fixes...")
        # Required-component checklist for the directive critic —
        # produces per-component missing diagnostics.
        _paper_ctx_cr = getattr(session, "_paper_ctx", None)
        _required_components_cr = (
            _paper_ctx_cr.for_critic() if _paper_ctx_cr is not None else ""
        )
        new_critique, revised_desc = session.critic.critique_and_revise(
            image_path=last_image,
            current_description=current_prompt,
            paper_context=craft_input.paper_text,
            caption=craft_input.description,
            venue=craft_input.venue,
            figure_type=craft_input.figure_type,
            required_components=_required_components_cr,
        )

        # Augment the diagnostic with OCR-based text-issue findings.
        # A dedicated VLM pass reads every visible text region and
        # flags rendering problems (garbled glyphs, off-vocabulary
        # decorative strings, duplicated titles). The findings are
        # merged into critic.issues + critic.suggestions so the
        # downstream skill_evolver converts them into typed edits on
        # S — no pixel patching, the next round's image-gen call
        # repairs the text by regenerating from the updated spec.
        text_critic = getattr(session, "text_critic", None)
        if text_critic is not None:
            try:
                text_issues = text_critic.analyze(
                    image_path=last_image,
                    paper_text=craft_input.paper_text,
                    caption=craft_input.description,
                )
                if text_issues:
                    from crafter.generation.craft.text_critic import TextCritic
                    issue_lines, sugg_lines = TextCritic.to_critique_lines(text_issues)
                    new_critique.issues = list(new_critique.issues) + issue_lines
                    new_critique.suggestions = (
                        list(new_critique.suggestions) + sugg_lines
                    )
                    session.history.add(
                        "text_critic", f"r{i+1}",
                        f"flagged={len(text_issues)} text issue(s)",
                        score=7.0,
                    )
            except Exception as e:
                logger.warning(f"TextCritic iter {i+1} failed: {e}")

        session.history.add(
            "critic", f"critique_revise:r{i+1}", score=new_critique.overall,
            detail=("; ".join(new_critique.issues[:2])
                    if new_critique.issues else ""),
        )

        # Build targeted fix guidance from critic's issues.
        fix_guidance_parts = []
        if new_critique.issues:
            fix_guidance_parts.append(
                "The critic identified these specific issues in the generated image:\n"
                + "\n".join(f"- {issue}" for issue in new_critique.issues[:5])
            )
        if new_critique.suggestions:
            fix_guidance_parts.append(
                "\nSuggested fixes:\n"
                + "\n".join(f"- {s}" for s in new_critique.suggestions[:3])
            )
        fix_guidance = "\n".join(fix_guidance_parts) if fix_guidance_parts else None

        # Render the next round's prompt from the (now updated) spec
        # S — typed edits applied above by skill_evolver flow through
        # render_image_gen_prompt(). For edit-mode, the short refer-
        # aware prompt structure is preserved; for t2i, the Opus-
        # backed refiner rewrites the prompt around the updated spec
        # block + critic fix guidance.
        if fix_guidance:
            console.print("  Refining prompt with targeted fixes...")
            history = [
                {"iteration": it.iteration, "critique": it.critique}
                for it in iterations if it.critique
            ]
            if len(history) > 3:
                compact_iteration_history(history, keep_last=3)
            # Refer-aware short prompt path triggers on refer_image_role.
            use_edit_prompt = bool(getattr(craft_input, "refer_image_role", ""))
            if use_edit_prompt:
                # Refer-aware refinement keeps the short prompt structure
                # (build_edit_instruction wrapper around description) and
                # appends the critic's specific fixes. Bypasses
                # session.refiner.refine to prevent the Opus path from
                # inflating the prompt back to the academic narrative,
                # which would dilute the refer signal.
                from crafter.generation.craft.figure_spec import build_edit_instruction
                spec_for_p5 = None
                edit_head = build_edit_instruction(
                    task=craft_input.figure_type,
                    style=craft_input.role,
                    description=craft_input.description,
                    spec=spec_for_p5,
                )
                fix_text = "\n".join(
                    f"  - {iss}" for iss in (new_critique.issues or [])[:5]
                )
                current_prompt = (
                    f"{edit_head}\n\n"
                    f"PRIOR ATTEMPT SHIPPED THESE ISSUES — fix them in this re-edit:\n"
                    f"{fix_text}"
                )
            else:
                # Prepend evolved per-paper guidance (test-time skill evolution).
                # render the spec as a single coherent block.
                if figure_spec is not None:
                    from crafter.generation.craft.figure_spec import render_image_gen_prompt
                    intent_preamble = (
                        getattr(session, "_intent_preamble", "")
                        or getattr(session.config, "_intent_preamble", "")
                    )
                    spec_block = render_image_gen_prompt(
                        figure_spec, craft_input.paper_text, craft_input.description,
                        intent_preamble=intent_preamble,
                        role=craft_input.role,
                    )
                    desc_with_evolved = f"{spec_block}\n\n{enriched_description}"
                else:
                    desc_with_evolved = (
                        f"{evolved_block}\n\n{enriched_description}"
                        if evolved_block else enriched_description
                    )
                # Refiner reads PaperReader's structured output so
                # iterative repairs stay grounded in the paper's actual
                # components rather than raw arxiv chrome / TOC.
                _paper_ctx_ref = getattr(session, "_paper_ctx", None)
                _paper_for_refine = (
                    _paper_ctx_ref.for_planner(15000)
                    if _paper_ctx_ref is not None
                    else craft_input.paper_text
                )
                current_prompt = session.refiner.refine(
                    current_prompt=current_prompt,
                    critique=new_critique,
                    paper_context=_paper_for_refine,
                    description=desc_with_evolved,
                    style_guidelines=style_prefix,
                    iteration_history=history,
                    user_feedback=fix_guidance,
                    must_preserve_components=session._component_names,
                )
            console.print(f"  Prompt refined ({len(current_prompt)} chars)")
        else:
            console.print(
                "  [yellow]No issues identified, keeping current prompt[/yellow]"
            )

        (run_dir / f"prompt_refine_{i + 1}.txt").write_text(
            current_prompt, encoding="utf-8",
        )

        # Generate.
        console.print("  Generating image...")
        image_bytes = session.generator.generate(
            prompt=current_prompt,
            reference_images=reference_images,
            style_prefix=style_prefix,
            debug_dir=str(run_dir),
        )
        if image_bytes is None:
            console.print("  [red]Generation failed, retrying...[/red]")
            image_bytes = session.generator.generate(
                prompt=current_prompt,
                reference_images=reference_images,
                style_prefix=style_prefix,
                model=session.config.generator_model,
                debug_dir=str(run_dir),
            )
        if image_bytes is None:
            console.print("  [red]Generation failed, skipping[/red]")
            iterations.append(CraftIteration(
                iteration=i + 1,
                prompt=current_prompt,
                image_path="",
                visual_style=best_style,
                critique=None,
                duration_seconds=time.time() - iter_start,
            ))
            continue

        image_path = str(run_dir / f"refine_{i + 1}.png")
        Path(image_path).write_bytes(image_bytes)
        console.print(f"  Saved: {image_path}")

        # Critique.
        console.print("  Evaluating...")
        critique = session.critic.evaluate(
            image_path=image_path,
            prompt=current_prompt,
            paper_context=craft_input.paper_text,
            description=craft_input.description,
            venue=craft_input.venue,
            visual_style=best_style,
            reference_paths=reference_paths,
            role=craft_input.role,
            task=craft_input.figure_type,
            required_components=_required_components_cr,
        )

        duration = time.time() - iter_start
        iteration = CraftIteration(
            iteration=i + 1,
            prompt=current_prompt,
            image_path=image_path,
            visual_style=best_style,
            critique=critique,
            duration_seconds=duration,
        )
        iterations.append(iteration)
        session._print_scores(critique, duration)

        if critique.issues:
            console.print("  [yellow]Issues found:[/yellow]")
            for issue in critique.issues[:5]:
                console.print(f"    - {issue}")

        # Track best.
        prev_score = (
            iterations[-2].critique.overall
            if len(iterations) >= 2 and iterations[-2].critique else 0
        )
        if critique.overall > best_score:
            best_score = critique.overall
            best_image_path = image_path
            console.print(
                f"  [green]Improved: {prev_score:.1f} → "
                f"{critique.overall:.1f}[/green]"
            )
        elif critique.overall < prev_score - 0.3:
            # Stochastic-retry: regen once with same prompt.
            console.print(
                f"  [yellow]Score dropped: {prev_score:.1f} → "
                f"{critique.overall:.1f}. Retrying with same prompt...[/yellow]"
            )
            retry_bytes = session.generator.generate(
                prompt=current_prompt,
                reference_images=reference_images,
                style_prefix=style_prefix,
                debug_dir=str(run_dir),
            )
            if retry_bytes:
                retry_path = str(run_dir / f"refine_{i + 1}_retry.png")
                Path(retry_path).write_bytes(retry_bytes)
                retry_critique = session.critic.evaluate(
                    image_path=retry_path,
                    prompt=current_prompt,
                    paper_context=craft_input.paper_text,
                    description=craft_input.description,
                    venue=craft_input.venue,
                    visual_style=best_style,
                    reference_paths=reference_paths,
                    role=craft_input.role,
                    task=craft_input.figure_type,
                    required_components=_required_components_cr,
                )
                if retry_critique.overall > critique.overall:
                    console.print(
                        f"  [green]Retry improved: {critique.overall:.1f} → "
                        f"{retry_critique.overall:.1f}[/green]"
                    )
                    critique = retry_critique
                    image_path = retry_path
                    iterations[-1] = CraftIteration(
                        iteration=i + 1,
                        prompt=current_prompt,
                        image_path=retry_path,
                        visual_style=best_style,
                        critique=retry_critique,
                        duration_seconds=time.time() - iter_start,
                    )
                    if retry_critique.overall > best_score:
                        best_score = retry_critique.overall
                        best_image_path = retry_path
                else:
                    console.print(
                        f"  [red]Retry also regressed (score "
                        f"{retry_critique.overall:.1f}). "
                        f"Reverting to best prompt.[/red]"
                    )

            if critique.overall <= prev_score:
                # Still regressed after retry — revert prompt.
                best_iter = max(
                    (it for it in iterations if it.critique),
                    key=lambda it: it.critique.overall,
                )
                current_prompt = best_iter.prompt

        # AgentJudge stop-or-iterate.
        pixel_artifact = _has_solid_artifact(image_path)
        decision = session.agent_judge.decide(
            image_path=image_path,
            critique=critique,
            iteration_idx=i + 1,
            has_pixel_artifact=pixel_artifact,
            paper_context=craft_input.paper_text,
            caption=craft_input.description,
        )
        console.print(
            f"  [cyan]AgentJudge: {'SHIP' if decision.stop else 'ITERATE'} "
            f"— {decision.reason}[/cyan]"
        )
        session.history.add(
            "agent_judge", f"decide:r{i + 1}",
            f"{'stop' if decision.stop else 'continue'}: {decision.reason}",
            score=decision.judge_quality,
        )
        if decision.stop:
            stop_reason = StopReason.COMPLETED
            break

    return (
        best_image_path,
        best_score,
        current_prompt,
        iterations,
        style_prefix,
        best_style,
        stop_reason,
    )
