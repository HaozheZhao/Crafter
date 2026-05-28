"""Multi-VLM ensemble judge for rendered-vs-original comparison."""
from __future__ import annotations

import logging
from typing import Optional

from crafter.editor.raster_to_svg.model_router import ModelRouter, encode_image, parse_json_response
from crafter.editor.raster_to_svg.schema import JudgeScore, JudgeVerdict

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

_JUDGE_PROMPT = """\
You are an expert visual judge comparing a rendered scientific figure against \
the original raster image. Both images are shown: the first is the ORIGINAL, \
the second is the RENDERED reproduction.

Evaluate how faithfully the rendered version reproduces the original on a 0-10 \
scale (10 = pixel-perfect match) for EACH of these aspects:

1. **position** - Bounding boxes, alignment, spacing, layout structure.
2. **color** - Fill colors, border colors, background.
3. **text** - Content accuracy, font size, weight, color, alignment.
4. **icon** - Presence, shape fidelity, detail level.
5. **arrow** - Direction, curvature, line style, arrowhead style.
6. **style** - Shadows, rounded corners, gradients, opacity, overall polish.

Also provide an **overall** score (weighted holistic impression, not a simple \
average).

List any concrete issues you notice, categorised into:
- `layout_issues` - wrong positions, overlaps, spacing errors
- `text_issues` - wrong text, font mismatch, truncation
- `icon_issues` - missing/wrong icons, poor segmentation
- `grounding_issues` - missing elements, extra elements, wrong element types

Return ONLY a JSON object (no markdown fence):

{
  "overall": <float 0-10>,
  "position": <float 0-10>,
  "color": <float 0-10>,
  "text": <float 0-10>,
  "icon": <float 0-10>,
  "arrow": <float 0-10>,
  "style": <float 0-10>,
  "layout_issues": ["..."],
  "text_issues": ["..."],
  "icon_issues": ["..."],
  "grounding_issues": ["..."],
  "summary": "One-sentence overall assessment."
}
"""

# Aspect keys expected in the JSON response
_ASPECT_KEYS = ("position", "color", "text", "icon", "arrow", "style")
_ISSUE_CATEGORIES = ("layout_issues", "text_issues", "icon_issues", "grounding_issues")


class JudgeAgent:
    """Ensemble judge that sends the same comparison to multiple VLMs and
    aggregates their scores into a single :class:`JudgeVerdict`."""

    def __init__(
        self,
        router: ModelRouter,
        models: list[str],
        threshold: float = 6.5,
    ) -> None:
        self.router = router
        self.models = models
        self.threshold = threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def judge(
        self,
        original_path: str,
        rendered_path: str,
    ) -> JudgeVerdict:
        """Compare *original_path* against *rendered_path* using all models.

        Returns a :class:`JudgeVerdict` with per-model scores, merged issues,
        and a ``passed`` flag based on the consensus rule.
        """
        logger.info(
            "Judge: comparing %s vs %s with %d models",
            original_path,
            rendered_path,
            len(self.models),
        )

        # Build the multi-image message (original first, rendered second)
        messages = self._build_messages(original_path, rendered_path)

        # Fan out to all models in parallel
        raw_responses: dict[str, str] = self.router.chat_parallel(
            messages=messages,
            models=self.models,
            temperature=0.15,
            max_tokens=4000,
        )

        # Parse each response into a JudgeScore
        scores: list[JudgeScore] = []
        for model, text in raw_responses.items():
            score = self._parse_score(model, text)
            scores.append(score)
            logger.info(
                "  %s  overall=%.1f  pos=%.1f col=%.1f txt=%.1f ico=%.1f arr=%.1f sty=%.1f",
                model,
                score.overall,
                score.position,
                score.color,
                score.text,
                score.icon,
                score.arrow,
                score.style,
            )

        # Retry any model that gave an outlier score (< 3.0 overall).
        # VLM judges can be volatile; one retry filters noise.
        for i, score in enumerate(scores):
            if score.overall < 3.0:
                logger.info("  Retrying %s (outlier score %.1f)", score.model, score.overall)
                try:
                    retry_resp = self.router.chat(
                        messages, model=score.model, temperature=0.15, max_tokens=4000,
                    )
                    retry_score = self._parse_score(score.model, retry_resp)
                    logger.info(
                        "  %s retry: overall=%.1f (was %.1f)",
                        score.model, retry_score.overall, score.overall,
                    )
                    if retry_score.overall > score.overall:
                        scores[i] = retry_score
                except Exception as e:
                    logger.warning("  Retry failed for %s: %s", score.model, e)

        # Aggregate into a verdict
        verdict = self._aggregate(scores)
        logger.info(
            "Judge verdict: passed=%s  avg_score=%.2f  issues=%d",
            verdict.passed,
            verdict.avg_score,
            len(verdict.all_issues),
        )
        return verdict

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_messages(
        self,
        original_path: str,
        rendered_path: str,
    ) -> list[dict]:
        """Construct the chat message list with two inline images."""
        orig_b64 = encode_image(original_path, max_dim=1200)
        rend_b64 = encode_image(rendered_path, max_dim=1200)

        content: list[dict] = [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{orig_b64}"},
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{rend_b64}"},
            },
            {"type": "text", "text": _JUDGE_PROMPT},
        ]
        return [{"role": "user", "content": content}]

    @staticmethod
    def _parse_score(model: str, raw_text: str) -> JudgeScore:
        """Best-effort parse of a single model's JSON response."""
        score = JudgeScore(model=model)
        if not raw_text.strip():
            score.issues = [f"{model}: empty response"]
            return score

        try:
            data = parse_json_response(raw_text)
            if not isinstance(data, dict):
                raise ValueError("Expected a JSON object, got a list")
        except Exception as exc:
            logger.warning("Failed to parse %s response: %s", model, exc)
            score.issues = [f"{model}: JSON parse error"]
            return score

        # Numeric fields
        score.overall = _clamp(data.get("overall", 0.0))
        score.position = _clamp(data.get("position", 0.0))
        score.color = _clamp(data.get("color", 0.0))
        score.text = _clamp(data.get("text", 0.0))
        score.icon = _clamp(data.get("icon", 0.0))
        score.arrow = _clamp(data.get("arrow", 0.0))
        score.style = _clamp(data.get("style", 0.0))
        score.summary = str(data.get("summary", ""))

        # Collect categorised issues into flat list
        issues: list[str] = []
        for cat in _ISSUE_CATEGORIES:
            items = data.get(cat, [])
            if isinstance(items, list):
                for item in items:
                    issues.append(f"[{cat}] {item}")
            elif isinstance(items, str) and items:
                issues.append(f"[{cat}] {items}")
        score.issues = issues

        return score

    def _aggregate(self, scores: list[JudgeScore]) -> JudgeVerdict:
        """Merge per-model scores into a single :class:`JudgeVerdict`."""
        verdict = JudgeVerdict()
        verdict.scores = scores

        if not scores:
            verdict.passed = False
            return verdict

        # Average overall score across models
        valid_overalls = [s.overall for s in scores if s.overall > 0]
        verdict.avg_score = (
            sum(valid_overalls) / len(valid_overalls) if valid_overalls else 0.0
        )

        # Consensus rule --------------------------------------------------
        # Passed if:
        #   (a) at least 60% of models score overall >= threshold
        #       (with 5 models, need >=3; with 3 models, need >=2)
        #   (b) no more than 1 model has ANY aspect score below 4.0
        n = len(scores)
        min_pass = max(2, int(n * 0.6))  # 60% majority
        above_threshold = sum(1 for s in scores if s.overall >= self.threshold)

        critical_failures = 0
        for s in scores:
            for aspect in _ASPECT_KEYS:
                if getattr(s, aspect, 10.0) < 4.0:
                    critical_failures += 1
                    break

        verdict.passed = (above_threshold >= min_pass) and (critical_failures <= 1)

        # Merge and categorise issues from all models ----------------------
        all_issues: list[str] = []
        layout_issues: list[str] = []
        text_issues: list[str] = []
        icon_issues: list[str] = []
        grounding_issues: list[str] = []

        for s in scores:
            for issue in s.issues:
                all_issues.append(issue)
                lower = issue.lower()
                if "[layout_issues]" in lower:
                    layout_issues.append(issue)
                elif "[text_issues]" in lower:
                    text_issues.append(issue)
                elif "[icon_issues]" in lower:
                    icon_issues.append(issue)
                elif "[grounding_issues]" in lower:
                    grounding_issues.append(issue)
                else:
                    # Un-categorised issues go to grounding as catch-all
                    grounding_issues.append(issue)

        # De-duplicate while preserving order
        verdict.all_issues = _dedup(all_issues)
        verdict.layout_issues = _dedup(layout_issues)
        verdict.text_issues = _dedup(text_issues)
        verdict.icon_issues = _dedup(icon_issues)
        verdict.grounding_issues = _dedup(grounding_issues)

        return verdict


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _clamp(value, lo: float = 0.0, hi: float = 10.0) -> float:
    """Clamp a numeric value (tolerant of non-numeric input)."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(lo, min(hi, v))


def _dedup(items: list[str]) -> list[str]:
    """Remove duplicate strings while preserving insertion order."""
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out
