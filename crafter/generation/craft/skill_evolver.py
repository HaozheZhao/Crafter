"""TestTimeSkillEvolver — per-session, per-iteration skill self-evolution.

Skills self-evolve at inference time (test-time) rather than being
statically learned offline. The skill template rewrites based on what
the critic observed in the prior iteration.

Key distinction from related approaches:
  - Static skills: a single fixed `drawer.md` for all papers
  - Offline-trained: rules learned from a training set, statically
    injected at test
  - SkillSelector: picks a subset of pre-learned rules per-sample, but
    rules themselves are static
  - **TestTimeSkillEvolver (this)**: produces an ephemeral per-session,
    per-iteration "skill addendum" — paper-specific guidance that
    evolves based on the prior iteration's critique. The addendum
    accumulates corrective deltas, never adds noise.

Flow per session:
  iter 1: addendum = "" (empty); generate
          critic observes → "labels too dense"
          → evolver writes addendum_v1 = "for THIS paper, labels are
            crowding; reduce to 8 max per panel, 14pt min"
  iter 2: prompt += addendum_v1; generate
          critic observes → "now layout is sparse but text IS readable"
          → evolver writes addendum_v2 = "labels max 8 per panel 14pt min;
            ALSO maintain visual density of ~5 components per panel"
  iter 3: ...

The addendum is INSTANCE-SPECIFIC — never reused across papers, never
saved to global skill files. It exists only for the duration of one
craft session.

This is different from prompt_refiner's existing iter loop because:
  - prompt_refiner rewrites the IMAGE-GEN PROMPT each iteration (the
    output that goes to the gen model). The "skill" (drawer prompt
    template) stays the same.
  - SkillEvolver evolves the PROMPT-CONSTRUCTION GUIDANCE itself — it
    tells the prompt_refiner agent "for THIS paper, optimize for X".
    Operates one level higher than prompt_refiner.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from crafter.shared.model_router import ModelRouter

logger = logging.getLogger(__name__)


@dataclass
class SkillState:
    """Per-session ephemeral skill addendum that evolves across iterations."""
    addendum: str = ""                              # current addendum text
    history: list[dict] = field(default_factory=list)  # iter → observation+delta

    def is_empty(self) -> bool:
        return not self.addendum.strip()


_EVOLVE_SYSTEM = """You are a SkillEvolver agent. You watch an academic
figure-generation pipeline iterate on ONE paper. Your job: write or refine
a short PAPER-SPECIFIC SKILL ADDENDUM that gets prepended to the prompt
for the next iteration.

Inputs each round:
  - The paper's caption + figure type
  - The current SKILL ADDENDUM (might be empty on iter 1)
  - The most recent critique of the generated image (what failed, what
    worked)
  - The iteration number

Output a NEW addendum that:
  1. Carries forward the parts of the previous addendum that the critique
     suggests are still helpful (don't lose progress).
  2. Adds 1-3 SPECIFIC paper-specific guidances based on what the critique
     observed. Each guidance is concrete and actionable for the
     next iteration's prompt construction.
  3. Removes parts of the previous addendum that became counterproductive
     (e.g., over-correction).
  4. Stays SHORT — 4-8 short bullets max. The addendum competes for
     attention budget with the rest of the prompt.

The addendum is EPHEMERAL — it applies only to this paper's session, will
not be saved or reused.

Critical: this is for THIS specific paper, not a general rule. Bad:
"always use 14pt sans-serif". Good: "for this paper's hippocampus analogy
panel, use a clearly anatomical brain icon, NOT a generic abstract head
silhouette — the prior iteration produced an unrecognizable shape".

If the critique says everything is fine OR no specific actionable issue:
return the addendum unchanged (or empty if there was none).

Output STRICT JSON:
{
  "new_addendum": "<the new full addendum text — short bullets — or empty>",
  "reasoning": "<1-2 sentences — what changed and why>"
}
"""


class TestTimeSkillEvolver:
    """LLM agent that evolves a per-session ephemeral skill addendum."""

    def __init__(
        self, router: "ModelRouter", model: Optional[str] = None,
    ) -> None:
        self.router = router
        # Use the strong model — this is a meta-reasoning task (observe
        # critique, decide what guidance helps next iter)
        self.model = model or router.config.critic_model

    def evolve(
        self,
        state: SkillState,
        caption: str,
        figure_type: str,
        critique_text: str,
        iteration: int,
    ) -> SkillState:
        """Mutate state in place: rewrite addendum based on critique.
        Returns the same state object."""
        user = (
            f"Paper figure caption: {caption[:300]}\n"
            f"Figure type: {figure_type or 'method/architecture diagram'}\n"
            f"Iteration: {iteration}\n\n"
            f"Current SKILL ADDENDUM (from prior iterations):\n"
            f"```\n{state.addendum or '(empty — first iteration)'}\n```\n\n"
            f"Most recent critique:\n```\n{critique_text[:2000]}\n```\n\n"
            f"Update the addendum for iteration {iteration + 1}."
        )
        try:
            data = self.router.chat_json(
                [{"role": "system", "content": _EVOLVE_SYSTEM},
                 {"role": "user", "content": user}],
                model=self.model, temperature=0.2,
            )
        except Exception as e:
            logger.warning(f"SkillEvolver failed: {e}; addendum unchanged")
            return state

        new_text = str(data.get("new_addendum", "")).strip()
        reason = str(data.get("reasoning", ""))[:200]
        prev = state.addendum
        state.addendum = new_text
        state.history.append({
            "iter": iteration,
            "prev_addendum_len": len(prev),
            "new_addendum_len": len(new_text),
            "reasoning": reason,
        })
        if new_text != prev:
            logger.info(
                f"SkillEvolver iter {iteration}: addendum {len(prev)}→"
                f"{len(new_text)} chars; {reason[:120]}"
            )
        return state

    def to_prompt_block(self, state: SkillState) -> str:
        """Format the current addendum as a prompt prefix.
        Returns empty string if addendum is empty (no dilution)."""
        if state.is_empty():
            return ""
        return (
            "PAPER-SPECIFIC GUIDANCE (evolved from prior iterations on this "
            "specific paper — apply in addition to general drawing skills):\n"
            f"{state.addendum.strip()}"
        )

    # ─── Structured edits to EvolvingFigureSpec ───
    def evolve_spec(
        self,
        spec: "EvolvingFigureSpec",
        critique_text: str,
        iteration: int,
    ) -> "EvolvingFigureSpec":
        """Instead of writing a free-text addendum, output STRUCTURED EDITS
        to the spec (resize_element / demote_element / add_layout_note /
        add_style_avoid). This eliminates the prompt-conflict source we
        (addresses the inconsistency between addendum and downstream prompts)."""
        from crafter.generation.craft.figure_spec import EvolvingFigureSpec

        # Build a compact view of the spec for the LLM
        elements_view = "\n".join(
            f"  - {e.id}: {e.name} ({'must' if e.must_have else 'optional'}"
            f"{', size=' + e.size_hint if e.size_hint else ''})"
            for e in spec.required_elements
        ) or "  (none)"
        avoids_view = ", ".join(spec.style.avoid) or "(none)"

        user = (
            f"Critique observations from prior iteration:\n```\n{critique_text[:2000]}\n```\n\n"
            f"Current spec state:\n"
            f"- elements:\n{elements_view}\n"
            f"- style.avoid: {avoids_view}\n"
            f"- layout.notes: {len(spec.layout.notes)} existing\n\n"
            f"Output STRICTLY-LIMITED edits as JSON. Available actions:\n"
            f"  - resize_element: {{\"elem_id\": \"elem_X\", \"size_hint\": \"small\"|\"medium\"|\"large\"}}\n"
            f"  - add_element_note: {{\"elem_id\": \"elem_X\", \"note\": \"<short, max 60 chars>\"}}\n"
            f"  - demote_element: {{\"elem_id\": \"elem_X\"}}  (only if critique CLEARLY says element hurts; never delete)\n"
            f"  - add_layout_note: {{\"note\": \"<short, max 80 chars>\"}}\n"
            f"  - add_style_avoid: {{\"item\": \"<short, e.g. 'drop shadows on text'>\"}}\n\n"
            f"Pick 0-3 edits TOTAL, ONLY if critique clearly justifies them. Empty list = no change.\n\n"
            f"Output STRICT JSON:\n"
            f"{{\"edits\": [{{\"action\": \"...\", \"params\": {{...}}, \"reason\": \"<short>\"}}]}}"
        )

        try:
            data = self.router.chat_json(
                [{"role": "system", "content":
                    "You are a SkillEvolver. Edit the figure spec based on critique. "
                    "Use STRUCTURED EDITS only — no free text. When in doubt, edit nothing. "
                    "Goal: respond to specific critique observations with targeted spec changes."},
                 {"role": "user", "content": user}],
                model=self.model, temperature=0.2,
            )
        except Exception as e:
            logger.warning(f"SkillEvolver.evolve_spec failed: {e}; spec unchanged")
            return spec

        edits = data.get("edits", []) if isinstance(data, dict) else []
        if not isinstance(edits, list):
            return spec

        applied_count = 0
        for edit in edits[:3]:  # cap at 3
            if not isinstance(edit, dict):
                continue
            action = edit.get("action", "")
            params = edit.get("params", {})
            reason = str(edit.get("reason", ""))[:120]
            if not isinstance(params, dict):
                continue

            try:
                if action == "resize_element":
                    if spec.resize_element(
                        params.get("elem_id", ""), params.get("size_hint", "medium"),
                        iter=iteration, reason=reason,
                    ):
                        applied_count += 1
                elif action == "add_element_note":
                    if spec.add_element_note(
                        params.get("elem_id", ""), str(params.get("note", ""))[:80],
                        iter=iteration, reason=reason,
                    ):
                        applied_count += 1
                elif action == "demote_element":
                    if spec.demote_element(
                        params.get("elem_id", ""),
                        iter=iteration, reason=reason,
                    ):
                        applied_count += 1
                elif action == "add_layout_note":
                    spec.add_layout_note(
                        str(params.get("note", ""))[:120],
                        iter=iteration, reason=reason,
                    )
                    applied_count += 1
                elif action == "add_style_avoid":
                    spec.add_style_avoid(
                        str(params.get("item", ""))[:80],
                        iter=iteration, reason=reason,
                    )
                    applied_count += 1
            except Exception as e:
                logger.warning(f"SE.evolve_spec edit '{action}' failed: {e}")

        if applied_count > 0:
            logger.info(
                f"SkillEvolver.evolve_spec iter {iteration}: applied {applied_count} "
                f"structured edits"
            )
        return spec
