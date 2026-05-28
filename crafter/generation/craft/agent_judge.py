"""AgentJudge: decides whether the pipeline should keep iterating.

The pipeline iterates up to `max_iter` times on complex cases and stops
early on simple ones. The judge's decision is a single call at each
iteration that inspects the image + critique + any structured signals
(claim-pass rate, artifact detection) and returns

    IterationDecision(stop: bool, reason: str, confidence: float)

Stop policy, in priority order:

1. Hard cap: ``iteration >= max_iter`` → stop regardless.
2. Content+readability bar: if critic's ``overall >= overall_threshold``
   AND ``content_accuracy >= content_threshold`` AND ``text_readability >=
   reada_threshold`` → stop; we've produced a shippable figure.
3. Pixel artifact check: if ``_has_solid_artifact(image)`` is True AND
   the critique score does not separately look great → KEEP iterating
   because the artifact will likely be reviewed down during refinement.
4. Judge VLM: ask the VLM one yes/no question — "is this figure ready to
   publish or does it need another iteration?" Low temperature. The VLM's
   yes/no is taken at face value, but filtered through the pixel + score
   rules above so VLM flattery can't override hard signals.

Callers pass the full :class:`CritiqueResult` plus the most recent
:class:`VerificationResult` if available — the judge does NOT recompute
either; it only consumes them.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class IterationDecision:
    stop: bool
    reason: str
    confidence: float = 0.0   # VLM's self-reported confidence, or 1.0 for hard rules
    judge_quality: float = 0.0  # VLM-provided ship-quality score in [0,10]


_JUDGE_SYSTEM = """\
You are the final shipping gate for an academic figure generation pipeline.
You will see ONE image along with the critic's scores and the figure's
intended content. Decide whether this image is READY TO SHIP in a
conference paper, or whether the pipeline should SPEND ONE MORE
REFINEMENT iteration.

Be strict about shipping:
- Any solid white/colored rectangle where content should be → NOT ready.
- Any cut-off label text or floating character fragments → NOT ready.
- Any component that was requested but is missing → NOT ready.
- A figure that is visibly boring but correct IS ready (don't iterate on
  pure aesthetic preference — each iteration costs money and can break
  content).

Respond with a single JSON object:
    {"ship": "yes"|"no", "quality": <0-10>, "reason": "<≤20 words>"}
No prose, no markdown."""


class AgentJudge:
    """Decides iterate-vs-stop at each Phase-5 step."""

    def __init__(
        self,
        router,
        model: Optional[str] = None,
        max_iter: int = 3,
        overall_threshold: float = 7.5,
        content_threshold: float = 6.5,
        readability_threshold: float = 6.0,
    ):
        self.router = router
        self.model = model or router.config.critic_model
        self.max_iter = max_iter
        self.overall_threshold = overall_threshold
        self.content_threshold = content_threshold
        self.readability_threshold = readability_threshold

    def decide(
        self,
        *,
        image_path: str,
        critique,                     # CritiqueResult
        iteration_idx: int,
        has_pixel_artifact: bool = False,
        paper_context: str = "",
        caption: str = "",
    ) -> IterationDecision:
        # 1. Hard cap
        if iteration_idx >= self.max_iter:
            return IterationDecision(True, f"hit max_iter={self.max_iter}", 1.0, critique.overall)

        # 2. Pixel artifact overrides score
        if has_pixel_artifact:
            return IterationDecision(
                False, "pixel artifact detected; keep iterating", 1.0,
                critique.overall,
            )

        # 3. Content-anchored early stop — ship when content is fine
        # even if aesthetics could be more polished. Each refinement
        # round may trade content fidelity for polish; iterating only when
        # content is genuinely weak avoids dragging faith down.
        if critique.content_accuracy >= self.content_threshold:
            return IterationDecision(
                True,
                f"content bar met (content={critique.content_accuracy:.1f} >= "
                f"{self.content_threshold})",
                1.0, critique.overall,
            )
        if critique.overall >= self.overall_threshold:
            return IterationDecision(
                True,
                f"overall ship bar met ({critique.overall:.1f} >= {self.overall_threshold})",
                1.0, critique.overall,
            )

        # 4. VLM judge — asked only when content AND overall both fail the
        # bar. The judge's default (per its system prompt) is to iterate
        # on aesthetic imperfection; we override: a 'no' from the judge
        # means iterate, but only once we've already exhausted the fast
        # content check above.
        return self._vlm_decision(image_path, critique, caption)

    def _vlm_decision(self, image_path, critique, caption) -> IterationDecision:
        user = (
            f"## Figure intent\n{(caption or '').strip()[:400]}\n\n"
            f"## Critic scores\n"
            f"content_accuracy={critique.content_accuracy:.1f}\n"
            f"layout_quality={critique.layout_quality:.1f}\n"
            f"text_readability={critique.text_readability:.1f}\n"
            f"aesthetic_quality={critique.aesthetic_quality:.1f}\n"
            f"overall={critique.overall:.1f}\n\n"
            f"Inspect the image and decide. Respond with the JSON only."
        )
        try:
            raw = self.router.critique_image(
                image_path=image_path,
                prompt=user,
                model=self.model,
                system_prompt=_JUDGE_SYSTEM,
            )
        except Exception as e:
            logger.debug(f"AgentJudge VLM failed: {e}; keeping iteration")
            return IterationDecision(False, f"vlm unavailable ({type(e).__name__})", 0.3, critique.overall)

        import json, re
        text = (raw or "").strip()
        if text.startswith("```"):
            text = "\n".join(text.split("\n")[1:]).rstrip("`").strip()
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", text, re.S)
            data = json.loads(m.group(0)) if m else {}
        ship = str(data.get("ship", "no")).lower().startswith("y")
        quality = float(data.get("quality", critique.overall))
        reason = str(data.get("reason", ""))[:120]
        return IterationDecision(ship, reason or ("vlm: ship" if ship else "vlm: iterate"),
                                 0.7, quality)
