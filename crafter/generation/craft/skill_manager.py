"""SkillManager: loads, saves, and iterates agent skills.

Skills are markdown files defining each agent's behavior (system prompts,
guidelines, rules). The manager supports two iteration modes:

- ``rewrite`` — the LLM regenerates each skill wholesale from feedback. This
  is an alternative path that empirically regresses the win rate
  by adding content-reducing rules. Kept only for
  research toggles.
- ``additive`` — the LLM extracts 1-line validated rules from the cases that
  IMPROVED across refinement, those rules are filtered against a banned-phrase
  list, required to be supported by ≥2 cases, and then APPENDED under an
  ``## Iterated rules (round N)`` header. Original skill content is never
  touched. This mode is the default and the one the paper will report.

Lookup order in :meth:`load_skills` (first match wins):
  1. Figure-type iterated:  skills/iterated/{name}_{figure_type}_round{N}.md
  2. Generic iterated:      skills/iterated/{name}_round{N}.md
  3. Figure-type base:      {skills_dir}/{name}_{figure_type}.md
  4. Generic base:          {skills_dir}/{name}.md
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

logger = logging.getLogger(__name__)

SKILL_NAMES = [
    "paper_reader",
    "planner",
    "stylist",
    "drawer",
    "critic",
    "retriever",
]

DEFAULT_SKILLS_DIR = Path(__file__).parent / "skills"
DEFAULT_ITERATED_DIR = DEFAULT_SKILLS_DIR / "iterated"

# Phrases that have historically degraded faithfulness when added to
# skills. Content-reducing rules are rejected (they delete paper content);
# readability-improving rules — trim labels, pick larger fonts, add
# whitespace — are kept. A broader banned list previously blocked
# legitimate typography rules and regressed readability.
_BANNED_PHRASES = (
    "reduce components", "fewer components", "fewer boxes",
    "remove components", "remove a component", "drop a component",
    "strip out components", "omit components", "skip components",
    "minimise content", "minimize content",
    "condense the content", "shorten the content",
    "less information", "fewer labels overall",
    "skip labeling", "remove labels",
)

# Phrases that are ENCOURAGED — the rule extractor is told these are
# acceptable visual-clarity moves (shorter labels, larger fonts, more
# whitespace). They are NOT content-reducing.
_ALLOWED_VISUAL_TRIMS = (
    "prefer short labels", "use 2-3 word labels", "use 14pt or larger",
    "generous whitespace", "trim verbose annotations",
    "keep labels concise", "avoid paragraph-length text",
)


@dataclass
class SkillSet:
    """A versioned set of agent skills."""

    paper_reader: str = ""
    planner: str = ""
    stylist: str = ""
    drawer: str = ""
    critic: str = ""
    retriever: str = ""
    round_idx: int = 0
    # Names of skills that were loaded from an iterated or figure-type-specific
    # override (i.e. the caller opted into a learned skill). The initial-prompt
    # builder injects only these as preamble, leaving the base system prompt
    # alone for the rest.
    iterated_names: set[str] = field(default_factory=set)

    def get(self, name: str) -> str:
        return getattr(self, name, "")


@dataclass
class SkillTestCase:
    """A test case for evaluating skill quality."""

    description: str = ""
    paper_context: str = ""
    generated_image_path: str = ""
    critique_scores: dict = field(default_factory=dict)
    round_idx: int = 0
    skill_round: int = 0
    # Optional: before/after deltas from a refinement pass, used by additive
    # iteration to learn what worked.
    prompt_before: str = ""
    prompt_after: str = ""
    scores_before: dict = field(default_factory=dict)
    scores_after: dict = field(default_factory=dict)


class SkillManager:
    """Manages agent skill files with versioning and iteration."""

    def __init__(
        self,
        skills_dir: str = "",
        output_dir: str = "./craft_output",
        iterated_dir: str = "",
    ):
        self.skills_dir = Path(skills_dir) if skills_dir else DEFAULT_SKILLS_DIR
        self.output_dir = Path(output_dir)
        self.iterated_dir = Path(iterated_dir) if iterated_dir else DEFAULT_ITERATED_DIR
        self.test_cases: list[SkillTestCase] = []
        self.skill_history: list[dict] = []  # round → avg_score mapping

    # ── Load / save ───────────────────────────────────────────────

    def load_skills(
        self,
        round_idx: int = 0,
        figure_type: str = "",
    ) -> SkillSet:
        """Load skills, preferring iterated and figure-type-specific overrides.

        See the module docstring for the full lookup order. ``iterated_names``
        records which skills were satisfied by an override (vs. the generic base),
        so downstream code can decide whether to inject the skill into prompts.
        """
        skill_set = SkillSet(round_idx=round_idx)

        for name in SKILL_NAMES:
            content, from_override = self._resolve_skill_file(name, round_idx, figure_type)
            if content:
                setattr(skill_set, name, content)
                if from_override:
                    skill_set.iterated_names.add(name)

        return skill_set

    def _resolve_skill_file(
        self,
        name: str,
        round_idx: int,
        figure_type: str,
    ) -> tuple[str, bool]:
        """Return (content, is_from_override) for the first matching lookup step."""
        # 1. Figure-type iterated
        if round_idx > 0 and figure_type:
            p = self.iterated_dir / f"{name}_{figure_type}_round{round_idx}.md"
            if p.exists():
                logger.info(f"Loaded iterated skill: {p}")
                return p.read_text(encoding="utf-8"), True
        # 2. Generic iterated
        if round_idx > 0:
            p = self.iterated_dir / f"{name}_round{round_idx}.md"
            if p.exists():
                logger.info(f"Loaded iterated skill: {p}")
                return p.read_text(encoding="utf-8"), True
            # Alternative path under output_dir
            p_alt = self.output_dir / f"skill_{name}_round{round_idx}.md"
            if p_alt.exists():
                logger.info(f"Loaded skill: {p_alt}")
                return p_alt.read_text(encoding="utf-8"), True
        # 3. Figure-type base
        if figure_type:
            p = self.skills_dir / f"{name}_{figure_type}.md"
            if p.exists():
                return p.read_text(encoding="utf-8"), True
        # 4. Generic base
        p = self.skills_dir / f"{name}.md"
        if p.exists():
            return p.read_text(encoding="utf-8"), False
        logger.warning(f"Skill not found: {name} (round={round_idx}, figure_type={figure_type!r})")
        return "", False

    def save_skills(
        self,
        skill_set: SkillSet,
        round_idx: int,
        output_dir: Optional[str] = None,
        figure_type: str = "",
        as_iterated: bool = False,
    ) -> None:
        """Persist the skill set.

        Args:
            as_iterated: when True, write to :attr:`iterated_dir` using the
                canonical ``{name}[_{figure_type}]_round{N}.md`` naming so that
                a future run with ``--skill-round N`` picks them up.
                Otherwise, write a debug/audit copy under ``output_dir``
                (alternative filename pattern).
        """
        if as_iterated:
            save_dir = self.iterated_dir
            save_dir.mkdir(parents=True, exist_ok=True)
            for name in SKILL_NAMES:
                content = skill_set.get(name)
                if not content:
                    continue
                suffix = f"_{figure_type}" if figure_type else ""
                path = save_dir / f"{name}{suffix}_round{round_idx}.md"
                path.write_text(content, encoding="utf-8")
                logger.info(f"Saved iterated skill: {path}")
            return

        save_dir = Path(output_dir) if output_dir else self.output_dir
        save_dir.mkdir(parents=True, exist_ok=True)
        for name in SKILL_NAMES:
            content = skill_set.get(name)
            if content:
                path = save_dir / f"skill_{name}_round{round_idx}.md"
                path.write_text(content, encoding="utf-8")
                logger.info(f"Saved skill: {path}")

    # ── Test cases ────────────────────────────────────────────────

    def add_test_case(self, test_case: SkillTestCase) -> None:
        self.test_cases.append(test_case)

    def get_test_cases_for_round(self, round_idx: int) -> list[SkillTestCase]:
        return [tc for tc in self.test_cases if tc.round_idx == round_idx]

    def evaluate_skill_round(self, round_idx: int) -> float:
        cases = self.get_test_cases_for_round(round_idx)
        if not cases:
            return 0.0
        scores = [
            tc.critique_scores.get("overall", 0.0) for tc in cases
            if tc.critique_scores
        ]
        return sum(scores) / len(scores) if scores else 0.0

    def get_best_round(self) -> int:
        if not self.skill_history:
            return 0
        best = max(self.skill_history, key=lambda x: x.get("avg_score", 0))
        return best.get("round", 0)

    # ── Rewrite-mode iteration (alternative path, off by default) ──

    def iterate_skill(
        self,
        skill_name: str,
        current_content: str,
        test_results: list[SkillTestCase],
        router,
    ) -> str:
        """Wholesale rewrite of a skill based on test feedback.

        Alternative path; regresses faithfulness in our runs
        . Prefer ``propose_additive_rules``.
        """
        feedback_parts = []
        for tc in test_results[-5:]:
            scores = tc.critique_scores
            if not scores:
                continue
            feedback_parts.append(
                f"- Description: {tc.description[:100]}\n"
                f"  Scores: overall={scores.get('overall', 0):.1f}, "
                f"content={scores.get('content_accuracy', 0):.1f}, "
                f"layout={scores.get('layout_quality', 0):.1f}, "
                f"text={scores.get('text_readability', 0):.1f}, "
                f"aesthetic={scores.get('aesthetic_quality', 0):.1f}\n"
                f"  Issues: {'; '.join(scores.get('issues', [])[:3])}"
            )

        if not feedback_parts:
            return current_content

        feedback = "\n".join(feedback_parts)

        prompt = f"""You are improving an agent skill for academic figure generation.

## Current Skill ({skill_name}):
```markdown
{current_content}
```

## Recent Test Results (figures generated using this skill):
{feedback}

## Task:
Analyze the test results. Identify patterns in what went wrong. Then rewrite
the skill to address these issues. Keep what works, fix what doesn't.

Common problems to address:
- If text_readability is low: Add stronger rules about minimal text
- If layout_quality is low: Add rules about spacing and alignment
- If aesthetic_quality is low: Improve color and typography guidelines
- If content_accuracy is low: Improve extraction/planning rules

Return the COMPLETE improved skill as markdown. Keep the same structure
(heading, sections, rules) but update the content.
"""

        messages = [
            {"role": "system", "content": "You improve agent skills based on empirical results."},
            {"role": "user", "content": prompt},
        ]
        improved = router.refine_prompt(messages, temperature=0.7, max_tokens=4096)
        return _strip_markdown_fence(improved)

    # ── Additive-mode iteration (default; non-regressing by design) ──

    def propose_additive_rules(
        self,
        skill_name: str,
        current_content: str,
        test_results: list[SkillTestCase],
        router,
        round_idx: int,
        min_support: int = 3,
        max_rules: int = 6,
    ) -> str:
        """Extract validated 1-line rules from winning test cases and append.

        Steps:
          1. Pick test cases where this skill's responsible dimensions
             improved across refinement (``scores_after > scores_before``
             on the relevant dim). A case with no before/after uses the
             final overall score >= 7.0 as a fallback "winning" signal.
          2. Ask the LLM to extract at most 2 short rules per case — atomic
             (each addresses one concrete behaviour) and PROMOTIONAL (add
             X, prefer Y), never reductive (simplify Z, remove W).
          3. Filter out rules containing any banned phrase.
          4. Cluster near-duplicates by normalized-text hash prefix and keep
             clusters with ``>= min_support`` supporting cases.
          5. Append the survivors under a new ``## Iterated rules (round N)``
             heading. Original skill content is preserved verbatim.
        """
        # Map skills to the dims they most directly influence; used to pick
        # "winning" cases per skill. Matches the routing in
        # :meth:`run_skill_iteration`.
        responsible_dims = {
            "paper_reader": ("content_accuracy",),
            "planner":      ("content_accuracy", "layout_quality"),
            "stylist":      ("aesthetic_quality", "role_match"),
            "drawer":       ("text_readability", "layout_quality"),
            "critic":       ("overall",),
            "retriever":    ("role_match",),
        }.get(skill_name, ("overall",))

        winning = _select_winning_cases(test_results, responsible_dims)
        if len(winning) < min_support:
            logger.info(f"[additive] {skill_name}: only {len(winning)} winning case(s); not enough support")
            return current_content

        # Determine the WEAKEST dim among this skill's responsibilities so
        # we can tell the extractor to target it.
        dim_scores: dict[str, list[float]] = {}
        for tc in test_results:
            for d in responsible_dims:
                v = tc.critique_scores.get(d, 0.0) or 0.0
                if isinstance(v, (int, float)) and v > 0:
                    dim_scores.setdefault(d, []).append(float(v))
        avg_by_dim = {d: sum(v) / len(v) for d, v in dim_scores.items() if v}
        weak_dim = min(avg_by_dim, key=avg_by_dim.get) if avg_by_dim else ""
        logger.info(f"[additive] {skill_name}: weak dim = {weak_dim}")

        proposed: list[tuple[str, str]] = []  # (raw_rule, source_id)
        for idx, tc in enumerate(winning):
            rules = _extract_rules_from_case(tc, skill_name, router, weak_dim=weak_dim)
            for r in rules:
                proposed.append((r, f"c{idx}"))

        if not proposed:
            return current_content

        # Filter banned
        filtered = [(r, src) for r, src in proposed if not _is_banned(r)]
        dropped = len(proposed) - len(filtered)
        if dropped:
            logger.info(f"[additive] {skill_name}: dropped {dropped} rule(s) containing banned phrases")

        # Cluster by hash of normalized text and require support
        buckets: dict[str, list[tuple[str, str]]] = {}
        for rule, src in filtered:
            key = _norm_key(rule)
            buckets.setdefault(key, []).append((rule, src))

        survivors: list[str] = []
        for key, items in buckets.items():
            unique_sources = {src for _, src in items}
            if len(unique_sources) >= min_support:
                chosen = min((r for r, _ in items), key=len)
                survivors.append(chosen.strip())

        # LLM-proposed rules rarely cluster (each call paraphrases
        # differently), so requiring ≥2 near-identical rules usually
        # yields zero survivors even when many good rules were proposed.
        # Fallback: when no clusters clear the support bar, take the
        # shortest rules from distinct source cases — preserves
        # "derived from multiple cases" without requiring verbatim
        # agreement across LLM calls.
        if not survivors and filtered:
            seen_sources: set[str] = set()
            singletons: list[str] = []
            for rule, src in sorted(filtered, key=lambda x: len(x[0])):
                if src in seen_sources:
                    continue
                seen_sources.add(src)
                singletons.append(rule.strip())
                if len(singletons) >= max_rules:
                    break
            if len(singletons) >= min_support:
                survivors = singletons
                logger.info(
                    f"[additive] {skill_name}: no rule cluster met support "
                    f"≥{min_support}; falling back to {len(survivors)} "
                    f"shortest rules from distinct cases"
                )

        survivors = survivors[:max_rules]
        if not survivors:
            logger.info(f"[additive] {skill_name}: no rule had support from ≥{min_support} cases")
            return current_content

        header = f"\n\n## Iterated rules (round {round_idx})\n"
        bullets = "\n".join(f"- {r}" for r in survivors)
        logger.info(f"[additive] {skill_name}: appended {len(survivors)} validated rule(s)")
        return current_content.rstrip() + header + bullets + "\n"

    # ── Orchestrator ──────────────────────────────────────────────

    def run_skill_iteration(
        self,
        current_skills: SkillSet,
        test_cases: list[SkillTestCase],
        router,
        round_idx: int,
        mode: Literal["rewrite", "additive"] = "additive",
    ) -> SkillSet:
        """Iterate weak-dimension skills and return a new :class:`SkillSet`.

        ``mode`` selects between wholesale rewrite and the
        additive rule-appending default.
        """
        new_skills = SkillSet(round_idx=round_idx)

        avg_scores: dict[str, list[float]] = {}
        for tc in test_cases:
            if not tc.critique_scores:
                continue
            for key, val in tc.critique_scores.items():
                if isinstance(val, (int, float)):
                    avg_scores.setdefault(key, []).append(val)
        avg = {k: sum(v) / len(v) for k, v in avg_scores.items() if v}

        skills_to_iterate: set[str] = set()
        if avg.get("content_accuracy", 10) < 7:
            skills_to_iterate.update(["paper_reader", "planner"])
        if avg.get("layout_quality", 10) < 7:
            skills_to_iterate.update(["planner", "drawer"])
        if avg.get("text_readability", 10) < 7:
            skills_to_iterate.update(["drawer", "stylist"])
        if avg.get("aesthetic_quality", 10) < 7:
            skills_to_iterate.update(["stylist", "drawer"])
        if avg.get("overall", 10) < 7:
            skills_to_iterate.update(["critic"])

        if not skills_to_iterate:
            skills_to_iterate = set(SKILL_NAMES)

        for name in SKILL_NAMES:
            current = current_skills.get(name)
            if name in skills_to_iterate and current:
                logger.info(f"Iterating skill ({mode}): {name}")
                if mode == "additive":
                    updated = self.propose_additive_rules(
                        skill_name=name,
                        current_content=current,
                        test_results=test_cases,
                        router=router,
                        round_idx=round_idx,
                    )
                else:
                    updated = self.iterate_skill(name, current, test_cases, router)
                setattr(new_skills, name, updated)
            else:
                setattr(new_skills, name, current)

        self.skill_history.append({
            "round": round_idx,
            "mode": mode,
            "avg_score": avg.get("overall", 0),
            "dim_scores": avg,
            "iterated_skills": list(skills_to_iterate),
        })

        return new_skills

    def save_history(self, output_dir: Optional[str] = None) -> None:
        save_dir = Path(output_dir) if output_dir else self.output_dir
        save_dir.mkdir(parents=True, exist_ok=True)
        history_path = save_dir / "skill_history.json"
        history_path.write_text(
            json.dumps(self.skill_history, indent=2, default=str),
            encoding="utf-8",
        )

    def load_history(self, output_dir: Optional[str] = None) -> None:
        load_dir = Path(output_dir) if output_dir else self.output_dir
        history_path = load_dir / "skill_history.json"
        if history_path.exists():
            self.skill_history = json.loads(history_path.read_text(encoding="utf-8"))


# ──────────────────────────────────────────────────────────────────────────────
# Helpers for additive rule extraction
# ──────────────────────────────────────────────────────────────────────────────


def _select_winning_cases(
    cases: list[SkillTestCase],
    responsible_dims: tuple[str, ...],
    top_frac: float = 0.40,
    min_dim_score: float = 6.0,
) -> list[SkillTestCase]:
    """Pick winning cases — adaptive:

    1. First preference: cases whose responsible dims strictly improved
       across refinement (``scores_after >= scores_before + 0.5``).
    2. Else: the top `top_frac` of cases ranked by the best responsible-dim
       score, restricted to cases that cleared `min_dim_score` on any
       responsible dim. This keeps some signal even when the model never
       hits overall>=7 (common with flash image models, session mean ≈ 6.2).
    """
    strict: list[SkillTestCase] = []
    for tc in cases:
        if tc.scores_before and tc.scores_after:
            for d in responsible_dims:
                before = tc.scores_before.get(d, 0.0) or 0.0
                after = tc.scores_after.get(d, 0.0) or 0.0
                if after - before >= 0.5:
                    strict.append(tc)
                    break
    if strict:
        return strict

    # Adaptive fallback: top-k by best responsible-dim score, over threshold.
    def _best_dim(tc: SkillTestCase) -> float:
        return max(
            (float(tc.critique_scores.get(d, 0.0) or 0.0) for d in responsible_dims),
            default=0.0,
        )
    ranked = sorted(cases, key=_best_dim, reverse=True)
    k = max(2, int(round(len(ranked) * top_frac)))
    survivors: list[SkillTestCase] = []
    for tc in ranked[:k]:
        if _best_dim(tc) >= min_dim_score:
            survivors.append(tc)
    return survivors


_RULE_EXTRACT_SYSTEM = """\
You extract PRACTICAL rules from a single success case of an academic figure
generation. A rule is a single sentence, ≤25 words, that names one CONCRETE
behaviour the agent should PREFER or ENFORCE.

## HARD CONSTRAINT: content preservation
Never suggest removing, dropping, or omitting COMPONENTS from the paper
(e.g. "skip the encoder", "reduce components", "fewer modules").
The figure must keep all components the paper describes.

## ENCOURAGED rule categories (choose appropriately for the weak dim):
- **Typography / readability**: "prefer 2-3 word labels", "use 14pt+ fonts",
  "keep label text concise", "avoid paragraph-length annotations inside
  boxes", "ensure high-contrast text on light background".
- **Layout**: "maintain generous whitespace between rows", "align
  components to a grid", "use left-to-right flow".
- **Content structuring**: "explicitly label sub-figures (A., B.)",
  "annotate arrows with 1-2 word actions", "group related components
  in a single panel".
- **Color & aesthetics**: "use a 3-color palette from paper venue",
  "reserve accent color for the main contribution".

## Pick rules targeted to the WEAK DIMENSION given in the prompt.
If the agent's weak dim is text_readability, favour typography rules.
If layout_quality, favour spacing/alignment rules. If content_accuracy,
favour structuring rules that clarify content without adding bulk.

Respond with JSON: `{"rules": ["…", "…"]}`. At most 2 rules. No prose."""


def _extract_rules_from_case(
    case: SkillTestCase,
    skill_name: str,
    router,
    weak_dim: str = "",
) -> list[str]:
    """Ask the LLM for up to 2 short rules targeted at `weak_dim`."""
    prompt_after = case.prompt_after or ""
    scores = case.critique_scores or {}
    # Name the weakest responsible dim explicitly so the extractor picks
    # typography rules when readability is weak, structuring rules when
    # content is weak, etc.
    dim_hint = f"\n## Weak dimension to target\n{weak_dim}\n" if weak_dim else ""
    user = (
        f"## Agent skill being improved\n{skill_name}\n"
        f"{dim_hint}"
        f"\n## Figure description\n{case.description[:300]}\n\n"
        f"## Winning prompt excerpt (first 2k chars)\n{prompt_after[:2000]}\n\n"
        f"## Final scores\ncontent={scores.get('content_accuracy', 0):.1f}, "
        f"layout={scores.get('layout_quality', 0):.1f}, text={scores.get('text_readability', 0):.1f}, "
        f"aesthetic={scores.get('aesthetic_quality', 0):.1f}, overall={scores.get('overall', 0):.1f}\n"
    )
    messages = [
        {"role": "system", "content": _RULE_EXTRACT_SYSTEM},
        {"role": "user", "content": user},
    ]
    try:
        data = router.chat_json(messages, temperature=0.3)
    except Exception as e:
        logger.debug(f"Rule extraction failed: {e}")
        return []
    rules = data.get("rules", []) if isinstance(data, dict) else []
    return [str(r).strip() for r in rules if str(r).strip()][:2]


def _is_banned(rule: str) -> bool:
    low = rule.lower()
    return any(b in low for b in _BANNED_PHRASES)


def _norm_key(rule: str) -> str:
    """Hash of lowercased, stopword-light text — for clustering near-duplicates."""
    text = rule.lower()
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    tokens = [t for t in text.split() if len(t) > 3]
    text = " ".join(sorted(set(tokens)))
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:10]


def _strip_markdown_fence(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        lines = s.split("\n")
        s = "\n".join(lines[1:])
        if s.endswith("```"):
            s = s[:-3]
        s = s.strip()
    return s
