"""VisualGrounder — paper-specific concrete visual element extractor.

Faith failures under strict judging trace to a concretization gap: the planner
emits abstract boxes (`rounded_rect|circle|diamond|cylinder`) but the human
figure embeds CONCRETE visual evidence — anatomical icons, real-photo
exemplars, domain-specific drawings, hand-curated math notation — that is
*mentioned in the paper text* yet never makes it into the generated figure.

Symptom: faith is low across all
configs):
  - test_27: paper mentions "hippocampus" 6×, "Pattern Separation" 5×, "Rapid
    Binding" 4×; generated figure has 0 brain icons or analogy panels.
  - test_14: paper mentions "Mutation Subgraph" 9× and graph construction;
    generated has "Subgraph Extraction" as a text label, not a graph drawing.
  - test_18: paper mentions "ShiftPE" 8× with grid coordinate semantics;
    generated has a generic table, not a concrete grid example.

Fix: an LLM agent that reads the paper and produces a list of MUST-INCLUDE
concrete visual elements with realization hints. Wired before planner so
both planner and drawer see them.

This is paper-specific (not the keyword-based
`_suggest_domain_visuals`) and demands concrete realization (icon/drawing/
photo/notation), not the generic "use icons" suggestion.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from crafter.shared.model_router import ModelRouter

logger = logging.getLogger(__name__)


@dataclass
class VisualElement:
    """A concrete visual element that MUST appear in the generated figure."""
    name: str                  # e.g. "hippocampus brain icon"
    realization: str           # e.g. "anatomical brain illustration in pink"
    placement_hint: str        # e.g. "top-right corner as analogy panel"
    rationale: str             # why we believe this should appear
    mention_count: int = 0     # how often the underlying concept appears in text


@dataclass
class StructuralRelation:
    """VG-extracted structural relation between two visual elements.

    Currently unused at runtime — feeding LLM-extracted relations into
    the Designer prompt was noisy on long arxiv-style sources and
    tended to over-constrain the generator without lifting
    faithfulness. Kept for API compatibility with
    VisualGrounding.relations.
    """
    src: str
    dst: str
    kind: str
    label: str = ""


@dataclass
class VisualGrounding:
    elements: list[VisualElement] = field(default_factory=list)
    relations: list[StructuralRelation] = field(default_factory=list)

    def to_prompt_block(self, max_items: int = 8) -> str:
        if not self.elements:
            return ""
        lines = [
            "MANDATORY CONCRETE VISUAL ELEMENTS (these are NOT optional — every "
            "one must appear as a real drawing/icon/photo, NEVER as a text label "
            "inside a box):"
        ]
        for i, el in enumerate(self.elements[:max_items], 1):
            lines.append(
                f"  {i}. {el.name} — render as: {el.realization} "
                f"(placement: {el.placement_hint})"
            )
        lines.append(
            "If you cannot draw one, use a recognizable iconic stand-in. "
            "Putting these as text inside a rectangle counts as MISSING the element."
        )
        return "\n".join(lines)


_SYSTEM_PROMPT = """You are a scientific figure forensics agent. Your job is to read a paper's methodology section and predict which CONCRETE visual elements the paper's authors likely included in their key methodology figure.

Academic figure authors commonly include:
1. ICONS for analogies (brain for memory, atom for molecules, lock for security)
2. REAL-WORLD EXEMPLARS (sample photos, generated images, dataset thumbnails) when the paper does generation/recognition on real images
3. DOMAIN-SPECIFIC DRAWINGS (3D protein surfaces, graph diagrams, attention heatmaps) when the paper operates on that data type
4. SPECIFIC NOTATION (probability formulas P(X|Y), tensor names q_ij, k_jkl) when the paper has named-tensor math

Generic agent pipelines miss these because they only emit boxes-with-text. Your job: given the paper text, list the SPECIFIC concrete visual elements that should appear, NOT generic suggestions.

Rules:
- Only list elements GROUNDED in the paper text (mentioned by name OR strongly implied by the data type).
- Be SPECIFIC: not "an icon", but "a brain anatomy icon for the hippocampus analogy".
- Include the realization (drawing style) and placement hint.
- If the paper has no obvious concrete visuals (pure-math/abstract paper), return an empty list.
- 3-7 elements is ideal. Never more than 8.

Output STRICT JSON (no prose, no code fences):
{
  "elements": [
    {
      "name": "<short descriptive name>",
      "realization": "<how to draw it: icon|3D render|graph diagram|photo thumbnail|colored token bar|formula notation>",
      "placement_hint": "<where in the figure: top-right as analogy panel | inline next to box X | left column as exemplar>",
      "rationale": "<one short clause: which paper concept it grounds>"
    }
  ]
}
"""


class VisualGrounder:
    """Extracts paper-specific concrete visual elements via LLM."""

    def __init__(self, router: "ModelRouter", model: Optional[str] = None) -> None:
        self.router = router
        # Use the strong model — this is a one-shot per-session call and the
        # quality matters for downstream prompt construction.
        self.model = model or router.config.critic_model

    def extract(
        self, paper_text: str, caption: str, figure_type: str = ""
    ) -> VisualGrounding:
        """Run the grounder once. Returns empty grounding on any failure."""
        text_excerpt = (paper_text or "")[:8000]
        user = (
            f"Figure caption: {caption}\n\n"
            f"Figure type: {figure_type or 'method/architecture diagram'}\n\n"
            f"Paper methodology section (excerpt):\n```\n{text_excerpt}\n```\n\n"
            "List the concrete visual elements that should appear in the "
            "paper's main methodology figure."
        )
        try:
            data = self.router.chat_json(
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user},
                ],
                model=self.model,
                temperature=0.2,
            )
        except Exception as e:
            logger.warning(f"VisualGrounder LLM call failed: {e}")
            return VisualGrounding()

        raw = data.get("elements", []) if isinstance(data, dict) else []
        if not isinstance(raw, list):
            return VisualGrounding()

        text_lower = (paper_text or "").lower()
        out: list[VisualElement] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "")).strip()
            if not name:
                continue
            # Best-effort mention count: how often the head noun shows up
            head = name.split()[0].lower() if name.split() else ""
            mc = text_lower.count(head) if len(head) > 3 else 0
            out.append(VisualElement(
                name=name,
                realization=str(item.get("realization", "icon")).strip(),
                placement_hint=str(item.get("placement_hint", "as visual element")).strip(),
                rationale=str(item.get("rationale", "")).strip(),
                mention_count=mc,
            ))
        out = out[:8]
        logger.info(f"VisualGrounder extracted {len(out)} concrete elements")
        return VisualGrounding(elements=out)

    def extract_to_spec(
        self, paper_text: str, caption: str, figure_type: str = "",
    ) -> "list":
        """ return list[figure_spec.VisualElement] for the
        EvolvingFigureSpec architecture, instead of a free-text MANDATORY
        block. Each element gets a stable id (elem_0, elem_1, ...) so SE
        can reference them in structured edits."""
        from crafter.generation.craft.figure_spec import VisualElement as SpecElement
        grounding = self.extract(paper_text, caption, figure_type)
        spec_elements = []
        for i, e in enumerate(grounding.elements):
            spec_elements.append(SpecElement(
                id=f"elem_{i}",
                name=e.name,
                realization=e.realization,
                placement=e.placement_hint,
                must_have=True,  # SE may demote later
            ))
        return spec_elements
