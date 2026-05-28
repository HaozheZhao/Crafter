"""AgentRegistry: claw-code-inspired agent dispatch and orchestration.

Patterns adopted from claw-code:
- Registry-based agent dispatch (match figure type → agent chain)
- Structured TurnResult with metadata (stop_reason, usage, errors)
- Error-as-feedback (failures feed back as structured data, not exceptions)
- Context compaction (summarize long history to avoid overflow)
- Stop conditions (budget, max_turns, quality threshold, regression)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Stop Reasons (from claw-code pattern)
# ──────────────────────────────────────────────────────────────────────────────

class StopReason(Enum):
    """Why the agent loop stopped."""
    COMPLETED = "completed"                # Quality threshold met
    MAX_TURNS = "max_turns_reached"        # Hit iteration limit
    MAX_BUDGET = "max_budget_reached"      # Token/API budget exhausted
    REGRESSION = "regression_detected"     # Quality getting worse
    USER_STOP = "user_stopped"             # Interactive mode: user said 'q'
    ALL_FAILED = "all_generations_failed"  # No images generated
    ERROR = "error"                        # Unrecoverable error


@dataclass
class TurnResult:
    """Structured result from one agent turn (claw-code pattern).

    Every turn produces a TurnResult — even failures. This enables
    the orchestrator to reason about what happened and decide next steps.
    """
    agent_name: str = ""
    turn_idx: int = 0
    success: bool = True
    output: Any = None           # Agent-specific output
    error_message: str = ""      # If failed, what went wrong
    duration_seconds: float = 0.0
    stop_reason: Optional[StopReason] = None

    # Metadata for tracking
    model_used: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    prompt_length: int = 0
    image_generated: bool = False

    def as_feedback(self) -> str:
        """Convert to feedback string for next agent (error-as-feedback pattern)."""
        if self.success:
            return f"[{self.agent_name}] completed in {self.duration_seconds:.1f}s"
        return f"[{self.agent_name}] FAILED: {self.error_message}"


@dataclass
class AgentSpec:
    """Specification for a registered agent."""
    name: str
    description: str
    skill_name: str = ""        # Links to skills/X.md
    model_tier: str = "planner" # Which model to use
    required_input: list[str] = field(default_factory=list)  # What context it needs
    output_keys: list[str] = field(default_factory=list)     # What it produces


# ──────────────────────────────────────────────────────────────────────────────
# Agent Chains (figure type → ordered list of agents)
# ──────────────────────────────────────────────────────────────────────────────

AGENT_CHAINS: dict[str, list[str]] = {
    "method_pipeline": ["paper_reader", "planner", "stylist", "drawer", "critic"],
    "architecture": ["paper_reader", "planner", "stylist", "drawer", "critic"],
    "teaser": ["paper_reader", "planner", "stylist", "drawer", "critic"],
    "comparison": ["paper_reader", "planner", "drawer", "critic"],
    "result_chart": ["paper_reader", "planner", "drawer", "critic"],
    "overview": ["paper_reader", "planner", "stylist", "drawer", "critic"],
    "poster": ["paper_reader", "planner", "stylist", "drawer", "critic"],
    "custom": ["paper_reader", "planner", "stylist", "drawer", "critic"],
}

AGENT_SPECS: dict[str, AgentSpec] = {
    "paper_reader": AgentSpec(
        name="paper_reader",
        description="Extracts key methodology components from paper text",
        skill_name="paper_reader",
        model_tier="planner",
        required_input=["paper_text"],
        output_keys=["components", "data_flow", "figure_suggestion"],
    ),
    "retriever": AgentSpec(
        name="retriever",
        description="Finds relevant reference figures for style guidance",
        skill_name="retriever",
        model_tier="quick",
        required_input=["description", "venue"],
        output_keys=["reference_images", "reference_paths"],
    ),
    "planner": AgentSpec(
        name="planner",
        description="Creates detailed visual plan for the figure",
        skill_name="planner",
        model_tier="planner",
        required_input=["paper_text", "description", "components"],
        output_keys=["figure_description", "layout_plan"],
    ),
    "stylist": AgentSpec(
        name="stylist",
        description="Refines description with venue-specific aesthetic guidelines",
        skill_name="stylist",
        model_tier="quick",
        required_input=["figure_description", "venue"],
        output_keys=["styled_description"],
    ),
    "drawer": AgentSpec(
        name="drawer",
        description="Generates the actual image from styled description",
        skill_name="drawer",
        model_tier="generator",
        required_input=["styled_description", "reference_images"],
        output_keys=["image_bytes", "image_path"],
    ),
    "critic": AgentSpec(
        name="critic",
        description="Evaluates generated image with strict scoring",
        skill_name="critic",
        model_tier="critic",
        required_input=["image_path", "figure_description", "paper_text"],
        output_keys=["critique", "revised_description"],
    ),
}


# ──────────────────────────────────────────────────────────────────────────────
# Context Compaction (from claw-code transcript.py pattern)
# ──────────────────────────────────────────────────────────────────────────────

def compact_iteration_history(
    iterations: list[dict],
    keep_last: int = 3,
    max_chars: int = 2000,
) -> str:
    """Summarize old iterations to avoid context overflow.

    Keeps full detail for the last N iterations, summarizes older ones.
    This prevents the prompt from growing unboundedly across many refinement rounds.
    """
    if len(iterations) <= keep_last:
        # Short enough — return full detail
        parts = []
        for it in iterations:
            critique = it.get("critique")
            if critique:
                issues = "; ".join(critique.issues[:2]) if critique.issues else "none"
                parts.append(
                    f"Round {it.get('iteration', '?')}: "
                    f"score={critique.overall:.1f}, issues: {issues}"
                )
        return "\n".join(parts)

    # Compact older iterations into a summary
    old = iterations[:-keep_last]
    recent = iterations[-keep_last:]

    # Summarize old iterations
    old_scores = [it["critique"].overall for it in old if it.get("critique")]
    old_issues = []
    for it in old:
        if it.get("critique") and it["critique"].issues:
            old_issues.extend(it["critique"].issues[:1])

    summary_parts = [
        f"[Rounds 1-{len(old)} summary: "
        f"scores ranged {min(old_scores):.1f}-{max(old_scores):.1f}, "
        f"recurring issues: {'; '.join(set(old_issues[:5]))}]"
    ]

    # Full detail for recent
    for it in recent:
        critique = it.get("critique")
        if critique:
            issues = "; ".join(critique.issues[:3]) if critique.issues else "none"
            summary_parts.append(
                f"Round {it.get('iteration', '?')}: "
                f"score={critique.overall:.1f}, issues: {issues}"
            )

    result = "\n".join(summary_parts)
    if len(result) > max_chars:
        result = result[:max_chars] + "\n[...truncated]"
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Session History Log (from claw-code history.py pattern)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class HistoryEvent:
    """One event in the generation history."""
    timestamp: float = 0.0
    agent: str = ""
    action: str = ""
    detail: str = ""
    score: float = 0.0
    success: bool = True


class HistoryLog:
    """Structured audit trail for figure generation sessions."""

    def __init__(self):
        self.events: list[HistoryEvent] = []

    def add(self, agent: str, action: str, detail: str = "",
            score: float = 0.0, success: bool = True) -> None:
        self.events.append(HistoryEvent(
            timestamp=time.time(),
            agent=agent,
            action=action,
            detail=detail,
            score=score,
            success=success,
        ))

    def as_markdown(self) -> str:
        """Render as markdown audit trail."""
        lines = ["## Generation History", ""]
        for i, ev in enumerate(self.events):
            status = "ok" if ev.success else "FAILED"
            score_str = f" (score: {ev.score:.1f})" if ev.score > 0 else ""
            lines.append(
                f"{i+1}. **{ev.agent}** → {ev.action}{score_str} [{status}]"
            )
            if ev.detail:
                lines.append(f"   {ev.detail}")
        return "\n".join(lines)

    def get_bottleneck_agents(self) -> list[str]:
        """Identify agents that frequently fail or produce low scores."""
        agent_scores: dict[str, list[float]] = {}
        agent_failures: dict[str, int] = {}

        for ev in self.events:
            if ev.score > 0:
                agent_scores.setdefault(ev.agent, []).append(ev.score)
            if not ev.success:
                agent_failures[ev.agent] = agent_failures.get(ev.agent, 0) + 1

        bottlenecks = []
        for agent, scores in agent_scores.items():
            avg = sum(scores) / len(scores)
            if avg < 6.0:
                bottlenecks.append(agent)
        for agent, fails in agent_failures.items():
            if fails >= 2 and agent not in bottlenecks:
                bottlenecks.append(agent)

        return bottlenecks

    def to_json(self) -> str:
        """Serialize for persistence."""
        return json.dumps(
            [{"agent": e.agent, "action": e.action, "detail": e.detail,
              "score": e.score, "success": e.success, "timestamp": e.timestamp}
             for e in self.events],
            indent=2,
        )


def get_agent_chain(figure_type: str) -> list[str]:
    """Get the agent chain for a figure type."""
    return AGENT_CHAINS.get(figure_type, AGENT_CHAINS["method_pipeline"])


def get_agent_spec(agent_name: str) -> AgentSpec:
    """Get the spec for a named agent."""
    return AGENT_SPECS.get(agent_name, AgentSpec(name=agent_name, description="Unknown agent"))
