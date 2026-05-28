r"""SketchAnalyzer: extract semantic + structural layout from a hand-drawn /
rough sketch so the figure can be generated FRESH (text-to-image only) rather
than image-conditioned on the sketch.

Motivation:
  Feeding the rough sketch into a chat-completions multimodal model as a
  reference image causes the generator to partially copy (trace) the
  sketch rather than rebuild it as a polished publication figure, even
  when the instruction explicitly says "do not copy".

Fix:
  Treat sketch task as text-only T2I generation. A VLM reads the sketch +
  paper text + caption and writes a structured layout description
  (panels / components / arrows / reading order). Downstream
  PromptRefiner consumes this text as the "Figure Description"; the
  generation backend never sees the sketch image, so it cannot copy.

Paper alignment (§4.2 4-role harness):
  Sketch handling stays inside the Designer $\mathcal{D}$ role — D reads
  (input, S_{t-1}) and proposes plans. For sketch task, "input" is the
  paper context + the EXTRACTED layout (via this analyzer). The sketch
  itself is metadata, not a generation conditioning signal.
"""
from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from crafter.shared.model_router import ModelRouter

logger = logging.getLogger(__name__)


_SKETCH_ANALYZER_SYSTEM = """\
You are a layout-extraction agent. Given a ROUGH SKETCH of a figure and the
paper context, write a CONCRETE STRUCTURED layout description that a clean
illustrator could use to draw a polished publication figure from scratch
WITHOUT seeing the sketch.

OUTPUT FORMAT
- Begin DIRECTLY with the layout description on the next line.
- 30-80 lines of concrete description.
- No preamble, no ``` fences, no JSON.

REQUIRED CONTENT (must include all)
1. **Overall structure**: panel count, primary flow direction (left-to-right /
   top-down / multi-column), figure aspect (wide landscape / tall portrait).
2. **Component list**: For each visible component in the sketch, in
   reading order, output a numbered line:
     N. {role/name} at {position} — {brief 1-line description of what it shows}
   Use EXACT paper terminology if the caption / paper context provides it.
3. **Connections**: Each arrow as one line: "Arrow from {source} to {target},
   labeled '{label or empty}', meaning {what flows}".
4. **Annotations / sub-labels**: Any text within or near components.
5. **Reading order**: A short paragraph stating the recommended reading order
   (top-left first / left-most column first / etc.).

DO NOT
- Describe colors, line styles, drawing media (the sketch is throwaway).
- Suggest "match the sketch's hand-drawn style" — the output should look
  PUBLICATION-CLEAN, not sketchy.
- Mention "the sketch shows..." or refer to the sketch object — the
  downstream illustrator never sees it.
- Output anything other than the layout description.
"""


def _b64_image(path: str) -> tuple[str, str]:
    p = Path(path)
    suffix = p.suffix.lower().lstrip(".")
    media = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "webp": "webp"}.get(suffix, "png")
    return base64.b64encode(p.read_bytes()).decode(), media


class SketchAnalyzer:
    """Extracts a text-only layout description from a rough sketch image."""

    def __init__(self, router: "ModelRouter"):
        self.router = router

    def analyze(
        self,
        *,
        sketch_path: str,
        caption: str,
        paper_text: str,
        instruction: Optional[str] = None,
    ) -> str:
        """Returns a layout-description string suitable for use as figure
        description in T2I gen. Returns '' on failure (caller falls back to
        whatever its default description is)."""
        if not sketch_path or not Path(sketch_path).exists():
            return ""

        try:
            b64, media = _b64_image(sketch_path)
        except Exception as e:
            logger.warning(f"SketchAnalyzer: failed to load sketch {sketch_path}: {e}")
            return ""

        user_blocks = [
            {"type": "text", "text": "## Paper context (excerpt)\n" + paper_text[:5000]},
            {"type": "text", "text": "## Caption / intent\n" + (caption or "")[:600]},
        ]
        if instruction:
            user_blocks.append(
                {"type": "text", "text": "## User instruction\n" + instruction[:500]}
            )
        user_blocks.append(
            {"type": "text", "text": "## Rough sketch (attached below)\n"}
        )
        user_blocks.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/{media};base64,{b64}"},
            }
        )
        user_blocks.append(
            {"type": "text",
             "text": "Now extract the structured layout description per the system rules."}
        )

        try:
            out = self.router._chat(
                messages=[
                    {"role": "system", "content": _SKETCH_ANALYZER_SYSTEM},
                    {"role": "user", "content": user_blocks},
                ],
                model=self.router.config.critic_model,  # VLM-capable
                temperature=0.3,
                max_tokens=2500,
            )
        except Exception as e:
            logger.warning(f"SketchAnalyzer LLM call failed: {e}")
            return ""

        out = (out or "").strip()
        if out.startswith("```"):
            lines = out.split("\n")
            out = "\n".join(lines[1:])
            if out.endswith("```"):
                out = out[:-3]
        out = out.strip()
        if len(out) < 100:
            logger.warning(f"SketchAnalyzer output too short ({len(out)}); discarding")
            return ""
        return out
