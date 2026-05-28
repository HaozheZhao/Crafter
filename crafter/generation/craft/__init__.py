"""PaperCraft Craft: agentic CLI for publication-quality figure generation.

Uses iterative generate → critique → refine loop with multi-model routing.
"""

from crafter.generation.craft.session import CraftSession, CraftInput, CraftResult
from crafter.generation.craft.skill_manager import SkillManager, SkillSet
from crafter.generation.craft.agent_registry import (
    AgentSpec, HistoryLog, StopReason, TurnResult,
    get_agent_chain, AGENT_CHAINS, AGENT_SPECS,
)

__all__ = ["CraftSession", "CraftInput", "CraftResult"]
