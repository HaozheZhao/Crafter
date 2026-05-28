"""CraftSession: the main agentic loop for iterative figure generation.

Flow:
1. RESEARCH: Search for reference figures, load user-provided references
2. PLAN: Use powerful LLM to understand paper and plan figure structure
3. VARIANTS: Generate initial images in multiple visual styles
4. SELECT: Pick the best variant based on critic scores
5. REFINE: Iteratively improve the best variant (generate → critique → refine)
6. LOOP: Steps until quality threshold or max iterations
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from crafter.generation.core.config import CraftConfig
from crafter.generation.craft.critic import CraftCritic, CritiqueResult
from crafter.generation.craft.image_generator import CraftImageGenerator
from crafter.shared.model_router import ModelRouter
from crafter.generation.craft.prompt_refiner import PromptRefiner
from crafter.generation.craft.reference_search import ReferenceImage, ReferenceSearcher
from crafter.generation.craft.agent_registry import (
    AgentSpec, HistoryLog, StopReason, TurnResult,
    compact_iteration_history, get_agent_chain,
)
from crafter.generation.craft.skill_manager import SkillManager, SkillSet, SkillTestCase
from crafter.generation.craft.venue_styles import (
    VISUAL_STYLES,
    build_style_prompt,
    get_recommended_visual_styles,
)

logger = logging.getLogger(__name__)
console = Console()


def _has_solid_artifact(image_path: str) -> bool:
    """Detect the specific "botched title inpaint" failure mode: a near-white
    rectangular block whose bottom edge CUTS THROUGH visible content.

    Legitimate white background in academic figures does not have a sharp
    horizontal content boundary at its bottom — the content shapes the
    boundary organically. A botched inpaint DOES: the mask was rectangular,
    so the content (cut text, component edges) aligns along the mask's
    bottom row.

    We look for, within the top half of the image:
      - a connected near-uniform white/light region ≥ 20% of the image width
        and ≥ 8% of image height
      - whose bottom row has a sharp content boundary (high Canny edge
        density just below the region) AND
      - which is NOT touching the image's bottom edge (pure margin).

    False positive rate was the first concern, so this
    narrower rule should flag only the user-reported failure pattern.
    """
    try:
        import cv2
        import numpy as np
        img = cv2.imread(image_path)
        if img is None:
            return False
        h, w = img.shape[:2]
        top = img[: h // 2]
        gray = cv2.cvtColor(top, cv2.COLOR_BGR2GRAY)
        # "White-ish" uniform area: bright AND low local variance
        white = (gray > 240).astype(np.uint8) * 255
        blurred = cv2.blur(gray, (5, 5))
        flat = (cv2.absdiff(gray, blurred) < 3).astype(np.uint8) * 255
        candidate = cv2.bitwise_and(white, flat)
        candidate = cv2.erode(candidate, np.ones((5, 5), np.uint8), iterations=2)
        num, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, connectivity=8)
        if num <= 1:
            return False
        # Canny of the full image (to probe the region below the candidate)
        edges_full = cv2.Canny(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 40, 120)
        for i in range(1, num):
            area = stats[i, cv2.CC_STAT_AREA]
            bw = stats[i, cv2.CC_STAT_WIDTH]
            bh = stats[i, cv2.CC_STAT_HEIGHT]
            x0 = stats[i, cv2.CC_STAT_LEFT]
            y0 = stats[i, cv2.CC_STAT_TOP]
            # Require the candidate to cover a meaningful slab
            if bw < 0.20 * w or bh < 0.08 * h or area < 0.02 * (h * w):
                continue
            # Candidates hugging both left-and-right edges are paper margin
            if x0 <= 2 and (x0 + bw) >= w - 2:
                continue
            # Candidate must NOT extend to the bottom of the image half we
            # examined (otherwise it's just a top margin with nothing to cut)
            if (y0 + bh) >= (h // 2) - 2:
                continue
            # Sample the 4px strip immediately below the candidate for edge
            # density — a botched inpaint mask has hard content boundary here.
            band_y0 = y0 + bh
            band_y1 = min(h, band_y0 + 6)
            if band_y1 <= band_y0 + 1:
                continue
            band = edges_full[band_y0:band_y1, x0:x0 + bw]
            if band.size == 0:
                continue
            density = float((band > 0).sum()) / band.size
            if density > 0.06:
                return True
        return False
    except Exception as e:
        logger.debug(f"Solid-artifact check failed: {e}")
        return False


@dataclass
class CraftInput:
    """Input to a craft session."""

    # Content
    paper_text: str = ""
    description: str = ""  # What the figure should show

    # Figure specification
    figure_type: str = "method_pipeline"
    venue: str = "neurips"
    visual_style: str = ""  # Specific visual style, or "" for auto-select

    # Reference images
    reference_paths: list[str] = field(default_factory=list)

    # Settings
    max_iterations: int = 5
    num_variants: int = 3  # Number of style variants to try
    output_path: str = ""
    skill_round: int = 0  # Which round of iterated skills to load (0 = base)
    # Communicative role of the target figure, one of
    # {"academic", "poster", "infographic"}. Drives visual-style selection
    # and the per-role preamble in the image prompt.
    #
    # Leave blank ("" or "auto") to let the session classify it from the
    # caption + instruction at start; this is the recommended default.
    role: str = ""

    # Unified-API refer field. When non-empty, signals that the
    # reference image at reference_paths[0] should be treated according
    # to the role hint. Mirrors EvolvingFigureSpec.refer_image_role and
    # is propagated to the spec when use_figure_spec is True. Empty
    # string suppresses the spec render's refer hint (pure-T2I path).
    refer_image_role: str = ""

    # Free-form aesthetic intent (e.g. "anime cute illustration",
    # "Studio Ghibli watercolor"). When non-empty, VisualMetaphorTranslator
    # is always triggered and the LLM proposes its own anchor honouring
    # this intent. Leave "" to use the role-based default anchor.
    aesthetic_intent: str = ""


@dataclass
class CraftIteration:
    """Record of one iteration in the craft loop."""

    iteration: int = 0
    prompt: str = ""
    image_path: str = ""
    visual_style: str = ""
    critique: Optional[CritiqueResult] = None
    duration_seconds: float = 0.0


@dataclass
class CraftResult:
    """Output from a craft session."""

    final_image_path: str = ""
    best_image_path: str = ""
    final_prompt: str = ""
    iterations: list[CraftIteration] = field(default_factory=list)
    variant_images: dict[str, str] = field(default_factory=dict)  # style → path
    reference_images_used: list[str] = field(default_factory=list)
    selected_style: str = ""
    total_duration_seconds: float = 0.0
    run_id: str = ""
    stop_reason: str = ""       # Why the loop stopped
    history_log: str = ""       # Markdown audit trail


class CraftSession:
    """Orchestrates the full agentic craft loop with multi-variant support."""

    def __init__(self, config: CraftConfig):
        self.config = config
        config.ensure_dirs()

        self.router = ModelRouter(config)
        self.generator = CraftImageGenerator(self.router)
        self.critic = CraftCritic(self.router, quality_threshold=config.quality_threshold)
        self.refiner = PromptRefiner(self.router)
        self.searcher = ReferenceSearcher(
            serper_api_key=config.serper_api_key,
            cache_dir=config.reference_cache_dir,
        )
        self.skill_manager = SkillManager(output_dir=config.output_dir)
        self.skills: Optional[SkillSet] = None
        self.history = HistoryLog()

        # PaperReader for component extraction (faithfulness)
        from crafter.generation.craft.paper_reader import PaperReader
        self.paper_reader = PaperReader(self.router)

        # TextCritic: OCR-based text-issue detector. One VLM call per
        # iteration reads every visible text region in the rendered
        # figure and flags rendering problems (garbled glyphs, off-
        # vocabulary decorative strings, duplicated titles). The
        # findings feed into the directive critic's diagnostic so the
        # downstream skill_evolver converts them into typed edits on
        # S (§4.2.3). Text is repaired by the next round's image-gen
        # call rendering an updated spec — no pixel patching.
        from crafter.generation.craft.text_critic import TextCritic
        self.text_critic = TextCritic(self.router, model=config.critic_model)

        # AgentJudge: decides stop-vs-iterate at each refinement step so
        # complex figures get up to `max_iter` passes and simple ones
        # ship immediately.
        from crafter.generation.craft.agent_judge import AgentJudge
        self.agent_judge = AgentJudge(
            self.router,
            max_iter=getattr(config, "agent_judge_max_iter", 3),
        )

        # SVG generator for code-based diagram generation (perfect text)
        from crafter.generation.craft.svg_generator import SVGGenerator
        self.svg_generator = SVGGenerator(self.router)

        # VisualGrounder: paper-specific concrete-element extractor.
        # Closes the concretization gap that drives faith=0% under
        # stricter VLM judges.
        from crafter.generation.craft.visual_grounder import VisualGrounder
        self.visual_grounder = (
            VisualGrounder(self.router)
            if getattr(config, "use_visual_grounder", True) else None
        )

        # TestTimeSkillEvolver: per-session per-iteration ephemeral skill
        # addendum that evolves based on each iteration's critique.
        # Distinct from SkillSelector (which picks static rules); this
        # writes new guidance that adapts as iterations proceed. The
        # addendum is paper-specific, ephemeral, and never saved.
        from crafter.generation.craft.skill_evolver import TestTimeSkillEvolver
        self.skill_evolver = (
            TestTimeSkillEvolver(self.router)
            if getattr(config, "use_skill_evolver", True) else None
        )

        # RolePlanner pre-generates a per-role structural blueprint
        # (paper title / authors / banner / row-by-row sections for
        # poster; numbered-step flow plan for infographic; layout
        # components for free-form roles). Appended to
        # enriched_description before the drawing phase so the drawer
        # has explicit structural guidance to follow.
        from crafter.generation.craft.role_planner import RolePlanner
        self.role_planner = RolePlanner(self.router)

        # VisualMetaphorTranslator — gated on non-academic roles or
        # abstract captions; short academic captions bypass it.
        from crafter.generation.craft.visual_metaphor_translator import (
            VisualMetaphorTranslator, should_translate,
        )
        self.visual_metaphor_translator = (
            VisualMetaphorTranslator(self.router)
            if getattr(config, "use_visual_metaphor_translator", True) else None
        )
        self._should_translate = should_translate

    def craft(self, craft_input: CraftInput) -> CraftResult:
        """Execute the full craft session with multi-variant exploration.

        Args:
            craft_input: Input specification for the figure.

        Returns:
            CraftResult with the best image and iteration history.
        """
        run_id = uuid.uuid4().hex[:8]
        run_dir = Path(self.config.output_dir) / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        total_start = time.time()

        # Auto-infer the communicative role if not supplied.
        if not (craft_input.role or "").strip() or craft_input.role.lower() == "auto":
            craft_input.role = self._infer_role(craft_input)

        # Load skills (prefer iterated + figure-type-specific overrides)
        self.skills = self.skill_manager.load_skills(
            round_idx=craft_input.skill_round,
            figure_type=craft_input.figure_type,
        )
        console.print(f"  Loaded {sum(1 for n in ['paper_reader','planner','stylist','drawer','critic','retriever'] if self.skills.get(n))} agent skills")

        # Adaptive K plan exploration (paper §4.2.2: "K is set
        # adaptively based on input constraints"):
        #   visual_style override     → K=1 (single pinned style)
        #   edit-mode w/ refer image  → K=2 (refer image fixes most of
        #                                    the visual framing, so
        #                                    fewer branches needed)
        #   aesthetic_intent OR poster → K=4 (high stylistic variance
        #                                    benefits from more
        #                                    parallel framings)
        #   else (default t2i)        → K=craft_input.num_variants (=3)
        if craft_input.visual_style:
            num_variants = 1
        elif getattr(craft_input, "refer_image_role", ""):
            num_variants = 2
        elif (getattr(craft_input, "aesthetic_intent", "") or "").strip():
            num_variants = 4
        elif (craft_input.role or "").lower() == "poster":
            num_variants = 4
        else:
            num_variants = craft_input.num_variants

        console.print(Panel(
            f"[bold]PaperCraft — Agentic Figure Generation[/bold]\n"
            f"Run: {run_id} | Venue: {craft_input.venue} | "
            f"Type: {craft_input.figure_type} | Variants: {num_variants} | "
            f"Max iterations: {craft_input.max_iterations}",
            style="blue",
        ))

        # ── Paper understanding + grounding + role plan ──
        console.print("\n[bold cyan]Paper Understanding[/bold cyan]")
        self.history = HistoryLog()
        stop_reason = StopReason.MAX_TURNS

        from crafter.generation.craft.phases.reading import (
            run_paper_understanding, select_visual_styles,
        )
        component_names, enriched_description, figure_spec = run_paper_understanding(
            self, craft_input,
        )

        # ── VisualMetaphorTranslator (gated) ──
        # Translates abstract / math / role-noun captions to concrete visual
        # narratives. Bypassed for academic T2I with short plain captions.
        if (self.visual_metaphor_translator is not None
                and self._should_translate(craft_input)):
            console.print("\n[bold cyan]Visual Metaphor Translation[/bold cyan]")
            try:
                vt = self.visual_metaphor_translator.translate(
                    paper_text=craft_input.paper_text,
                    raw_caption=craft_input.description,
                    instruction=getattr(craft_input, "instruction", "") or "",
                    task=craft_input.figure_type,
                    role=craft_input.role,
                    has_refer_image=bool(
                        getattr(craft_input, "refer_image_role", "")
                        or getattr(craft_input, "reference_paths", None)
                    ),
                    aesthetic_intent=getattr(
                        craft_input, "aesthetic_intent", "") or "",
                )
                if vt.visual_caption:
                    console.print(
                        f"  Translated → {len(vt.visual_caption)} chars, "
                        f"anchor={vt.aesthetic_anchor}"
                    )
                    self.history.add(
                        "visual_metaphor_translator", "translate",
                        f"anchor={vt.aesthetic_anchor}, len={len(vt.visual_caption)}",
                        score=8.0,
                    )
                    # Augment enriched_description with the visual narrative
                    enriched_description = (
                        f"{vt.visual_caption}\n\n"
                        f"## Original caption context\n"
                        f"{enriched_description}"
                    )
            except Exception as e:
                logger.warning(f"VisualMetaphorTranslator failed: {e}")
                self.history.add(
                    "visual_metaphor_translator", "error",
                    str(e)[:100], score=3.0,
                )

        # ── Reference search ──
        console.print("\n[bold cyan]Reference Search[/bold cyan]")
        reference_images, reference_paths = self._research(craft_input)
        console.print(f"  Found {len(reference_images)} reference images")
        self.history.add("retriever", "search", f"Found {len(reference_images)} references")

        # ── Determine visual styles ──
        console.print("\n[bold cyan]Visual Style Selection[/bold cyan]")
        visual_styles = select_visual_styles(
            craft_input, num_variants=num_variants, session=self,
        )

        # ── Generate variants (parallel) ──
        console.print("\n[bold cyan]Style Variant Generation (parallel)[/bold cyan]")
        from crafter.generation.craft.phases.generation import run_variant_generation
        variant_results, variant_images = run_variant_generation(
            self, craft_input,
            visual_styles=visual_styles,
            enriched_description=enriched_description,
            figure_spec=figure_spec,
            reference_images=reference_images,
            reference_paths=reference_paths,
            run_dir=run_dir,
        )

        # SVG variant generation is available but disabled for eval speed.
        # The SVG generator can be used separately via self.svg_generator.generate()
        # for editable output (SVG/drawio) after the best raster variant is selected.

        if not variant_results:
            console.print("\n[red]All variant generations failed[/red]")
            return CraftResult(run_id=run_id, total_duration_seconds=time.time() - total_start)

        # ── Select best variant ──
        console.print("\n[bold cyan]Variant Selection[/bold cyan]")
        from crafter.generation.craft.phases.selection import select_best_variant
        best_style, best_path, current_prompt, best_critique, best_vs_name = select_best_variant(
            variant_results,
        )

        # ── Iterative refinement on the selected variant ──
        from crafter.generation.craft.phases.refinement import run_iter_refinement
        (
            best_image_path, best_score,
            current_prompt, iterations,
            style_prefix, best_style, stop_reason,
        ) = run_iter_refinement(
            self, craft_input,
            best_path=best_path,
            best_critique=best_critique,
            current_prompt=current_prompt,
            best_style=best_style,
            best_vs_name=best_vs_name,
            enriched_description=enriched_description,
            reference_images=reference_images,
            reference_paths=reference_paths,
            figure_spec=figure_spec,
            run_dir=run_dir,
        )

        # ── Readability polish pass ──
        # Runs at most ONCE, after the refinement loop has settled content, when the
        # best image's text_readability is weak but its content_accuracy is
        # already acceptable. Single dedicated pass; guidance explicitly
        # forbids removing components or shortening labels.
        best_image_path, best_score = self._readability_polish(
            best_image_path=best_image_path,
            best_score=best_score,
            current_prompt=current_prompt,
            style_prefix=style_prefix,
            reference_images=reference_images,
            reference_paths=reference_paths,
            best_style=best_style,
            enriched_description=enriched_description,
            craft_input=craft_input,
            run_dir=run_dir,
            iterations=iterations,
        )

        # ── Final Quality Guard ──
        # Run a final check on the best image to catch disasters:
        # 1. Missing key components
        # 2. Text readability below minimum threshold
        # 3. Large solid-fill artifact regions (pixel check, catches
        #    cases where the critic's VLM fixated on the rest of the
        #    figure and missed a blank/uniform block)
        if iterations and best_image_path:
            console.print(f"\n[bold cyan]Quality Guard Check[/bold cyan]")

            # Use the last critique, or re-evaluate if it's an edited image
            guard_critique = None
            last_it = iterations[-1]
            if last_it.critique:
                guard_critique = last_it.critique
            elif best_critique:
                guard_critique = best_critique

            if guard_critique:
                issues = []
                if guard_critique.text_readability < 3.0:
                    issues.append(f"text_readability={guard_critique.text_readability:.1f}")
                if guard_critique.content_accuracy < 3.0:
                    issues.append(f"content_accuracy={guard_critique.content_accuracy:.1f}")
                if guard_critique.overall < 4.0:
                    issues.append(f"overall={guard_critique.overall:.1f}")

                # Pixel-level artifact check: detect large solid white /
                # uniform rectangles in the rendered figure. The critic's
                # VLM occasionally misses these because it fixates on the
                # rest of the figure; the pixel check is a cheap backstop.
                artifact_detected = _has_solid_artifact(best_image_path)
                if artifact_detected:
                    issues.append("solid-fill block / inpaint artifact (pixel check)")

                if issues:
                    console.print(f"  [red]Quality guard triggered: {', '.join(issues)}[/red]")
                    # Prefer a non-edited iteration when artifact was detected:
                    # the edited "_edited.png" is probably the broken one, and
                    # the pre-edit image (saved as `variant_*.png` or
                    # `refine_*.png`) is visually cleaner.
                    non_edited = [
                        it for it in iterations
                        if it.critique and it.image_path
                        and not it.image_path.endswith("_edited.png")
                    ]
                    if artifact_detected and non_edited:
                        best_iter = max(non_edited, key=lambda it: it.critique.overall)
                    else:
                        best_iter = max(
                            (it for it in iterations if it.critique and it.image_path),
                            key=lambda it: it.critique.overall,
                            default=None,
                        )
                    if best_iter and best_iter.image_path != best_image_path:
                        console.print(f"  [yellow]Reverting to best iteration "
                                      f"(score={best_iter.critique.overall:.1f}"
                                      f"{' — artifact-safe' if artifact_detected else ''})[/yellow]")
                        best_image_path = best_iter.image_path
                        self.history.add("quality_guard", "revert",
                                         f"Reverted: {', '.join(issues)}")
                else:
                    console.print(f"  [green]Passed (overall={guard_critique.overall:.1f})[/green]")

        # ── Final output ──
        total_duration = time.time() - total_start
        final_image_path = best_image_path

        # Copy best image to output path if specified
        if craft_input.output_path and best_image_path:
            import shutil
            Path(craft_input.output_path).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(best_image_path, craft_input.output_path)
            final_image_path = craft_input.output_path

        # Save skills for this round
        self.skill_manager.save_skills(self.skills, round_idx=0, output_dir=str(run_dir))

        # Save test cases from this run for skill iteration
        for it in iterations:
            if it.critique and it.image_path:
                tc = SkillTestCase(
                    description=craft_input.description,
                    paper_context=craft_input.paper_text[:500],
                    generated_image_path=it.image_path,
                    critique_scores={
                        "content_accuracy": it.critique.content_accuracy,
                        "layout_quality": it.critique.layout_quality,
                        "text_readability": it.critique.text_readability,
                        "aesthetic_quality": it.critique.aesthetic_quality,
                        "role_match": it.critique.role_match,
                        "artifact_severity": it.critique.artifact_severity,
                        "overall": it.critique.overall,
                        "issues": it.critique.issues,
                    },
                    round_idx=0,
                    skill_round=0,
                )
                self.skill_manager.add_test_case(tc)

        result = CraftResult(
            final_image_path=final_image_path,
            best_image_path=best_image_path,
            final_prompt=current_prompt,
            iterations=iterations,
            variant_images=variant_images,
            reference_images_used=reference_paths,
            selected_style=best_style,
            total_duration_seconds=total_duration,
            run_id=run_id,
            stop_reason=stop_reason.value,
            history_log=self.history.as_markdown(),
        )

        # Save history log
        (run_dir / "history.md").write_text(self.history.as_markdown(), encoding="utf-8")
        (run_dir / "history.json").write_text(self.history.to_json(), encoding="utf-8")

        # Run skill iteration — core component, always runs. Triggered only
        # when the average critic score is below the quality bar,
        # so we don't waste LLM cost re-iterating already-strong skills.
        if self.skill_manager.test_cases:
            try:
                avg_score = self.skill_manager.evaluate_skill_round(0)
                if avg_score < 7.0:
                    console.print(f"\n  [cyan]Running skill iteration (avg={avg_score:.1f}, mode=additive)...[/cyan]")
                    updated_skills = self.skill_manager.run_skill_iteration(
                        current_skills=self.skills,
                        test_cases=self.skill_manager.test_cases,
                        router=self.router,
                        round_idx=1,
                        mode="additive",
                    )
                    if updated_skills:
                        # Debug copy under the run dir 
                        self.skill_manager.save_skills(
                            updated_skills, round_idx=1, output_dir=str(run_dir),
                        )
                        self.history.add("skill_manager", "iterate",
                                         f"Updated skills for round 1 (avg={avg_score:.1f}, mode=additive)")
            except Exception as e:
                logger.warning(f"Skill iteration failed: {e}")

        # Log bottleneck agents
        bottlenecks = self.history.get_bottleneck_agents()
        if bottlenecks:
            console.print(f"  [yellow]Bottleneck agents: {', '.join(bottlenecks)}[/yellow]")

        self._print_summary(result)
        return result

    _ROLE_KEYWORDS = {
        "poster": ("poster", "conference poster", "banner", "tri-fold"),
        "infographic": ("infographic", "blog", "explainer", "tutorial", "lil'log",
                         "distill", "magazine"),
    }

    def _infer_role(self, craft_input: "CraftInput") -> str:
        """Classify the figure's communicative role.

        Heuristic-first (cheap keyword pass on the instruction), then a
        single quick-model fallback when the keywords are silent.
        """
        text = " ".join([craft_input.description or "", craft_input.paper_text[:1500]]).lower()
        for role, kws in self._ROLE_KEYWORDS.items():
            if any(kw in text for kw in kws):
                return role
        try:
            verdict = self.router.quick_task([
                {"role": "system",
                 "content": "Classify the requested figure into exactly one role: "
                            "'academic' (paper figure), 'poster' (conference poster), "
                            "or 'infographic' (blog / explainer)."},
                {"role": "user",
                 "content": f"Caption / instruction:\n{craft_input.description[:600]}\n\n"
                            "Reply with exactly one word: academic, poster, or infographic."},
            ], temperature=0.0, max_tokens=8).strip().lower()
            for r in ("academic", "poster", "infographic"):
                if r in verdict:
                    return r
        except Exception:
            pass
        return "academic"

    def _suggest_domain_visuals(
        self, paper_text: str, caption: str, figure_type: str
    ) -> str:
        """Suggest CONCRETE visual elements based on the paper's domain.

        Instead of saying "include visual elements", this tells the prompt
        engineer exactly WHAT to draw based on detected keywords.
        """
        text = (paper_text[:3000] + " " + caption).lower()
        suggestions = []

        # Vision / Image domains
        if any(w in text for w in ["image", "pixel", "visual", "photo", "camera",
                                     "video", "frame", "scene", "render"]):
            suggestions.append(
                "VISUAL: Include small thumbnail images as inputs/outputs "
                "(e.g., a photo for 'input image', a rendered view for 'output'). "
                "Show feature maps as small colored grids."
            )

        # Graph / Network domains
        if any(w in text for w in ["graph", "node", "edge", "vertex",
                                     "adjacency", "GNN", "GCN"]):
            suggestions.append(
                "VISUAL: Show graphs as circles connected by lines. "
                "Use colored nodes to distinguish types."
            )

        # NLP / Text domains
        if any(w in text for w in ["token", "embedding", "language model",
                                     "LLM", "prompt", "text", "sentence"]):
            suggestions.append(
                "VISUAL: Show token sequences as colored bars/rectangles in a row. "
                "Use document or chat bubble icons for text inputs."
            )

        # Attention / Transformer
        if any(w in text for w in ["attention", "transformer", "self-attention",
                                     "cross-attention", "query", "key", "value"]):
            suggestions.append(
                "VISUAL: Show attention maps as small heatmap grids (warm colors). "
                "Use Q/K/V labels for query/key/value streams."
            )

        # 3D / Gaussian / NeRF
        if any(w in text for w in ["3d", "gaussian", "nerf", "point cloud",
                                     "volume", "mesh", "voxel"]):
            suggestions.append(
                "VISUAL: Show 3D data as colored blobs/splats or point clouds. "
                "Use camera icons for viewpoints."
            )

        # Biology / Medical
        if any(w in text for w in ["protein", "molecule", "drug", "gene",
                                     "cell", "brain", "medical", "clinical"]):
            suggestions.append(
                "VISUAL: Include molecular/protein structure thumbnails. "
                "Use scientific notation and proper chemical/biological symbols."
            )

        # RL / Agent
        if any(w in text for w in ["agent", "reward", "policy", "environment",
                                     "action", "state", "reinforcement"]):
            suggestions.append(
                "VISUAL: Show agent-environment loop with distinct icons. "
                "Use feedback arrows for reward signals."
            )

        # Poster-specific visual hints
        if figure_type == "poster":
            suggestions.append(
                "POSTER LAYOUT: This is a conference poster, not a paper figure. "
                "Use landscape orientation with a title banner at top, 3-column "
                "body layout, and conclusion strip at bottom. "
                "Display key metrics as large colored badges. "
                "Use bullet points and icons instead of paragraphs."
            )

        if not suggestions:
            suggestions.append(
                "VISUAL: Use icons and small illustrations instead of text "
                "to represent concepts. Replace descriptions with visual elements."
            )

        return "\n".join(suggestions)

    def _research(
        self, craft_input: CraftInput
    ) -> tuple[list[bytes], list[str]]:
        """Find reference images."""
        all_refs: list[ReferenceImage] = []

        # Load user-provided references
        if craft_input.reference_paths:
            local_refs = self.searcher.load_local_references(craft_input.reference_paths)
            all_refs.extend(local_refs)
            console.print(f"  Loaded {len(local_refs)} user-provided references")

        # Search for more references via Serper
        if self.config.serper_api_key and len(all_refs) < self.config.max_reference_images:
            remaining = self.config.max_reference_images - len(all_refs)
            topic = craft_input.description[:100]
            search_refs = self.searcher.search(
                topic=topic,
                venue=craft_input.venue,
                figure_type=craft_input.figure_type,
                max_results=remaining,
            )
            all_refs.extend(search_refs)
            if search_refs:
                console.print(f"  Found {len(search_refs)} references via Serper search")

        # Load image bytes
        ref_bytes = []
        ref_paths = []
        for ref in all_refs[:self.config.max_reference_images]:
            try:
                data = Path(ref.path).read_bytes()
                ref_bytes.append(data)
                ref_paths.append(ref.path)
            except Exception as e:
                logger.warning(f"Failed to load reference {ref.path}: {e}")

        return ref_bytes, ref_paths

    def _readability_polish(
        self,
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
        """Readability polish. Implementation lives in
        `phases/polish.py`; see that module's docstring for full notes."""
        from crafter.generation.craft.phases.polish import run_readability_polish
        return run_readability_polish(
            self,
            best_image_path=best_image_path,
            best_score=best_score,
            current_prompt=current_prompt,
            style_prefix=style_prefix,
            reference_images=reference_images,
            reference_paths=reference_paths,
            best_style=best_style,
            enriched_description=enriched_description,
            craft_input=craft_input,
            run_dir=run_dir,
            iterations=iterations,
        )

    def _print_scores(self, critique: CritiqueResult, duration: float, label: str = "") -> None:
        """Print a Rich table with critique scores."""
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Dimension", style="cyan")
        table.add_column("Score", style="bold")

        dims = [
            ("Content", critique.content_accuracy),
            ("Layout", critique.layout_quality),
            ("Text", critique.text_readability),
            ("Aesthetic", critique.aesthetic_quality),
            ("Role", critique.role_match),
            ("Artifact", critique.artifact_severity),
        ]
        for name, score in dims:
            color = "green" if score >= 7.5 else "yellow" if score >= 5 else "red"
            table.add_row(name, f"[{color}]{score:.1f}/10[/{color}]")

        table.add_row("", "")
        overall_color = "green" if critique.is_acceptable else "yellow"
        table.add_row(
            "[bold]Overall[/bold]",
            f"[bold {overall_color}]{critique.overall:.1f}/10[/bold {overall_color}]",
        )
        if duration > 0:
            table.add_row("Time", f"{duration:.1f}s")

        console.print(table)

    def _print_summary(self, result: CraftResult) -> None:
        """Print final session summary."""
        n_iter = len(result.iterations)
        successful = [it for it in result.iterations if it.image_path]
        best_score = max(
            (it.critique.overall for it in result.iterations if it.critique),
            default=0,
        )
        selected_name = VISUAL_STYLES.get(result.selected_style, {}).get("name", result.selected_style)

        variant_info = ""
        if result.variant_images:
            variant_info = f"Variants generated: {len(result.variant_images)} ({', '.join(result.variant_images.keys())})\n"

        console.print(Panel(
            f"[bold green]Craft Complete[/bold green]\n\n"
            f"Run ID: {result.run_id}\n"
            f"{variant_info}"
            f"Selected style: {selected_name}\n"
            f"Refinement iterations: {n_iter} ({len(successful)} successful)\n"
            f"Best score: {best_score:.1f}/10\n"
            f"Best image: {result.best_image_path}\n"
            f"Final image: {result.final_image_path}\n"
            f"References used: {len(result.reference_images_used)}\n"
            f"Total time: {result.total_duration_seconds:.1f}s",
            style="green",
        ))