"""TextCritic — OCR-based text-issue detector for the directive critic.

Wraps a single VLM call that reads every visible text region in a
generated figure and flags problems (garbled glyphs, unreadable
rendering, words clearly off-topic from the paper, decorative mock
strings the paper does not mention).

Outputs text-only suggestions consumed by the directive critic
(§4.2.4 of the paper) — no pixel patching, no inpainting, no
regeneration. The suggestions feed into ``critic.issues`` /
``critic.suggestions`` so the existing skill_evolver converts them
into typed edits on the evolving spec ``S`` (§4.2.3) and the next
round's image-gen prompt naturally repairs the text.

Empty list on parse failure or empty response.
"""
from __future__ import annotations

import base64
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from crafter.shared.model_router import ModelRouter

logger = logging.getLogger(__name__)


@dataclass
class TextIssue:
    """A single text-rendering issue found by the OCR-based detector."""
    observed: str          # what OCR sees in the image
    location: str = ""     # natural-language position ("top-left", "near arrow")
    severity: str = "minor"  # "garbled" | "off_vocab" | "duplicate" | "minor"
    suggestion: str = ""   # one-line repair hint for the next-round prompt


_SYSTEM = (
    "You are a precise OCR-based text critic for a scientific-figure "
    "image-generation pipeline. Read EVERY visible text region in the "
    "image and flag rendering problems. Do NOT comment on layout / "
    "color / arrows — only TEXT issues.\n"
    "\n"
    "Flag a token as a problem only when one of these holds:\n"
    "  - garbled: contains broken / merged / partial glyphs, unicode "
    "look-alikes, or mojibake (e.g. 'Eɴcodɛr' for 'Encoder')\n"
    "  - off_vocab: a real-looking word that does NOT appear in the "
    "paper / caption AND looks like decorative filler ('Lorem ipsum', "
    "random Latin, made-up brand names)\n"
    "  - duplicate: the same paper title / figure number repeated when "
    "it should appear at most once\n"
    "  - minor: small kerning / spacing oddities that still read OK\n"
    "\n"
    "Do NOT flag correctly rendered field-standard tokens (Encoder, "
    "Loss, Attention(Q,K,V), section labels, etc.).\n"
    "\n"
    "Output STRICT JSON ARRAY ONLY — no preamble, no fences:\n"
    '[{"observed":"...","location":"...","severity":"garbled|off_vocab|'
    'duplicate|minor","suggestion":"render this label as ... in plain '
    'sans-serif"}]\n'
    "If no problems, return []."
)


_JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def _parse_json_array(raw: str) -> list[dict]:
    if not raw:
        return []
    cleaned = _JSON_FENCE_RE.sub("", raw).strip()
    # Some models prepend a comment; extract first JSON array
    m = re.search(r"\[\s*(?:\{.*?\}\s*,?\s*)*\]", cleaned, re.DOTALL)
    if m:
        cleaned = m.group(0)
    try:
        data = json.loads(cleaned)
    except Exception as e:
        logger.debug(f"TextCritic JSON parse failed: {e}; raw={raw[:200]}")
        return []
    return data if isinstance(data, list) else []


class TextCritic:
    """Single-call OCR text-issue detector. Stateless."""

    def __init__(self, router: "ModelRouter",
                 model: str = "gemini-3.1-pro-preview",
                 max_issues: int = 6) -> None:
        self.router = router
        self.model = model
        self.max_issues = max_issues

    def analyze(self, image_path: Path | str,
                 paper_text: str = "",
                 caption: str = "") -> list[TextIssue]:
        """One VLM call. Returns flagged text issues or []."""
        p = Path(image_path)
        if not p.exists():
            return []
        try:
            img_b64 = base64.b64encode(p.read_bytes()).decode()
        except Exception as e:
            logger.warning(f"TextCritic: failed to read {p}: {e}")
            return []

        # Compact paper context — helps the critic decide what's
        # "off_vocab" without sending 30K chars to the VLM.
        ctx_parts = []
        if caption.strip():
            ctx_parts.append(f"Figure caption: {caption[:600]}")
        if paper_text.strip():
            ctx_parts.append(f"Paper excerpt: {paper_text[:1500]}")
        ctx = "\n\n".join(ctx_parts) if ctx_parts else "(no paper context provided)"

        user_text = (
            f"{ctx}\n\n"
            f"Inspect the image. List up to {self.max_issues} of the "
            f"WORST text-rendering issues. Empty array if none."
        )

        content = [
            {"type": "image_url",
             "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
            {"type": "text", "text": user_text},
        ]

        try:
            resp = self.router._chat(
                [{"role": "system", "content": _SYSTEM},
                 {"role": "user", "content": content}],
                model=self.model,
                temperature=0.1,
                max_tokens=1500,
            )
        except Exception as e:
            logger.warning(f"TextCritic LLM call failed: {e}")
            return []

        items = _parse_json_array(resp or "")
        out: list[TextIssue] = []
        for it in items[: self.max_issues]:
            if not isinstance(it, dict):
                continue
            observed = str(it.get("observed", "")).strip()
            if not observed:
                continue
            out.append(TextIssue(
                observed=observed[:120],
                location=str(it.get("location", ""))[:60],
                severity=str(it.get("severity", "minor")).strip().lower()[:16],
                suggestion=str(it.get("suggestion", ""))[:240],
            ))
        return out

    @staticmethod
    def to_critique_lines(issues: list[TextIssue]) -> tuple[list[str], list[str]]:
        """Convert issues into (issues_lines, suggestions_lines) that
        can be merged into the directive critic's diagnostic. Each
        line is a single short string."""
        issues_lines: list[str] = []
        sugg_lines: list[str] = []
        for iss in issues:
            tag = f"[{iss.severity}]"
            loc = f" at {iss.location}" if iss.location else ""
            issues_lines.append(
                f"text {tag} observed '{iss.observed}'{loc}"
            )
            if iss.suggestion:
                sugg_lines.append(iss.suggestion)
        return issues_lines, sugg_lines
