"""Self-evolving prompt module for the refine loop (single-case).

Idea: refine has multi-iter trajectory of (prompt, outcome). After
each accepted iter (from iter 2 onward) we run a small text-only
gpt-5.5 call that reflects on the trajectory and emits a 1-2 sentence
"lessons-for-next-iter" addendum. The addendum is prepended to the
next REFINE_PROMPT so the LLM concentrates on what worked / avoids
what failed in this case.

Cost: 1 text-only call (~5s, ~400 max_tokens) per iter ≥ 2.

Output of `evolve(trajectory) -> str`: a 1-2 sentence imperative
addendum, e.g.

  "Iter 1's bbox-locked moves regressed text positions; on the next
   iter, prefer SMALL local moves (≤ 30 px) and verify each by visual
   match against Image #1 before continuing."
"""
from __future__ import annotations
import os

import json
import logging
import re
import sys
from pathlib import Path

REPO = Path(os.environ.get("CRAFTER_HOME", str(Path(__file__).resolve().parents[2])))

from crafter.editor.approach.iter_svg_fix import call_model  # noqa: E402

logger = logging.getLogger("harness.prompt_evolver")


REFLECTION_PROMPT = """You are coaching an SVG-refinement loop running
on ONE academic figure. Below is the trajectory of refinement
iterations so far. Reflect on what worked and what didn't, then emit a
1-2-sentence imperative LESSONS addendum that the NEXT iteration's
prompt will read.

Goals of refinement:
  • Increase 'overall' judge score.
  • Reduce 'checker_issues' total.
  • Without dropping <text> / <image> / <rect> beyond safety guards.

TRAJECTORY (most-recent iter LAST):
{TRAJECTORY}

Your output must be exactly:
  • 1-2 imperative sentences (max ~40 words total).
  • Concrete and actionable for the next iter.
  • Either focused on something to KEEP DOING or to STOP / AVOID.
  • If the trajectory is monotone-improving and ALL fixes are being
    applied cleanly, output exactly "(continue with current strategy)".

Examples of GOOD lessons:
  • "Iter 1's overlap moves were good; on iter 2 do not aggressively
     resize images — instead, only nudge them by ≤ 30 px to break
     overlap while preserving size."
  • "Score dropped after large structural restructure; for next iter
     keep the panel layout intact and focus only on fixing missing
     text labels."
  • "Continue applying missing_text fixes — score is climbing and no
     elements are being lost."

Examples of BAD (do NOT output):
  • Vague: "Try harder" / "Keep improving"
  • Verbose explanations of the trajectory itself
  • New 5-step plans

Return ONLY the addendum sentence(s), no JSON, no quotes, no preamble.
"""


def _summarise_trajectory(trajectory: list[dict],
                          max_iters_in_window: int = 4) -> str:
    """Render the trajectory list (compact, recent-most-first)."""
    if not trajectory:
        return "(no prior iterations)"
    window = trajectory[-max_iters_in_window:]
    lines = []
    for t in window:
        i = t.get("iter")
        sc_b = t.get("score_before", "?")
        sc_a = t.get("score_after", "?")
        try:
            delta = (float(sc_a) - float(sc_b))
            d_str = f"{delta:+.2f}"
        except Exception:
            d_str = "?"
        chk_changes = t.get("checker_changes") or {}
        chk_str = " ".join(f"{k}={v:+d}" for k, v in chk_changes.items()
                           if isinstance(v, int) and v != 0) or "no_chg"
        applied = t.get("judge_fixes_addressed") or []
        applied_str = (f"{sum(applied)}/{len(applied)} judge-fixes addressed"
                       if applied else "")
        notable = t.get("notable") or ""
        status = t.get("status", "accepted")
        lines.append(
            f"  iter {i} [{status}]: score {sc_b} → {sc_a} ({d_str}); "
            f"checkers: {chk_str}; {applied_str}; {notable}"
        )
    return "\n".join(lines)


def evolve(trajectory: list[dict],
           model: str = "openai/gpt-5.5") -> str:
    """Run a single text-only reflection call. Returns lesson string
    (empty string on failure or "(continue ...)" when nothing to add)."""
    if len(trajectory) < 1:
        return ""
    prompt = REFLECTION_PROMPT.replace("{TRAJECTORY}",
                                       _summarise_trajectory(trajectory))
    msgs = [{"role": "user", "content": prompt}]
    try:
        resp = call_model(msgs, model=model, max_tokens=400)
    except Exception as e:
        logger.warning("prompt_evolver api failed: %s", e)
        return ""
    out = resp.strip()
    # Strip trivial wrapping
    out = re.sub(r"^['\"`]+|['\"`]+$", "", out).strip()
    # Cap length
    if len(out) > 400:
        out = out[:400] + "..."
    return out


def format_for_refine(lesson: str) -> str:
    """Wrap the lesson in a clear header for REFINE_PROMPT injection."""
    if not lesson or "continue with current strategy" in lesson.lower():
        return ""
    return ("LESSONS FROM PRIOR ITERATIONS THIS CASE (read first):\n"
            f"  • {lesson}\n\n")
