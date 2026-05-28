"""Judging utilities for the composition phase.

quick_judge — single vision-LLM call, returns numeric overall score plus
              per-aspect scores and a brief textual feedback. Used inside
              the refine loop to steer iteration cheaply.

For benchmark-grade scoring use the 3-VLM ensemble judge in
``crafter.editor.raster_to_svg.agents.judge.JudgeAgent``.
"""
from __future__ import annotations
import os

import base64
import io
import json
import logging
import re
import sys
from pathlib import Path

from PIL import Image

REPO = Path(os.environ.get("CRAFTER_HOME", str(Path(__file__).resolve().parents[2])))

from crafter.editor.approach.iter_svg_fix import call_model  # noqa: E402

logger = logging.getLogger("harness.judge")


QUICK_JUDGE_PROMPT = """Image 1 = ORIGINAL academic figure (ground truth).
Image 2 = my SVG rendering attempting to reproduce it.

Rate Image 2 on each aspect from 0 to 10 (decimals OK):
  - overall      — overall similarity / fidelity
  - position     — element positions / layout match
  - color        — colour fidelity
  - text         — text labels present and correct
  - icon         — icon / image content match
  - arrow        — arrows / connectors match
  - style        — visual style coherence

Then list at most 3 SPECIFIC fixes the next iteration should make
(short imperative bullets — what is missing, mis-placed, or wrong).

Return STRICT JSON:
{
  "overall": 7.2, "position": 7.0, "color": 6.8, "text": 8.0,
  "icon": 6.5, "arrow": 7.0, "style": 7.5,
  "fixes": ["add the dog photograph at top-left",
            "make the histogram bars taller",
            "fix overlapping Q/K/V text"]
}
Return ONLY the JSON.
"""


def _b64(path: Path, max_dim: int = 1500) -> str:
    img = Image.open(path).convert("RGB")
    if max(img.size) > max_dim:
        scale = max_dim / max(img.size)
        img = img.resize((int(img.width * scale), int(img.height * scale)),
                         Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88)
    return base64.b64encode(buf.getvalue()).decode()


def quick_judge(
    original_png: Path, preview_png: Path, model: str = "openai/gpt-5.5",
) -> dict:
    """Single VLM call: returns scores + fix suggestions."""
    msgs = [{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url":
                {"url": f"data:image/jpeg;base64,{_b64(original_png)}"}},
            {"type": "image_url", "image_url":
                {"url": f"data:image/jpeg;base64,{_b64(preview_png)}"}},
            {"type": "text", "text": QUICK_JUDGE_PROMPT},
        ],
    }]
    try:
        resp = call_model(msgs, model=model, max_tokens=2000,
                          label="stage6_quick_judge")
    except Exception as e:
        logger.warning("quick_judge failed: %s", e)
        return {"overall": 0.0, "fixes": [f"judge api error: {e}"]}

    text = resp.strip()
    m = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if m:
        text = m.group(1).strip()
    for s in range(len(text)):
        if text[s] == "{":
            depth = 0
            for e in range(s, len(text)):
                if text[e] == "{":
                    depth += 1
                elif text[e] == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(text[s:e + 1])
                        except json.JSONDecodeError:
                            break
    return {"overall": 0.0, "fixes": ["judge returned non-JSON"]}
