"""Structural VLM critic — covers missing elements, structural layout,
text reflow, wrong content.

One gpt-5.5 vision call per refine iter. The model sees:
  - Image #1 — original
  - Image #2 — current SVG rendering
  - Paddle-OCR text list grounded on the original (so it knows exactly
    what text *should* be present and where)
  - labels.json digest (raster + vector placeholders by id, with bboxes)

It returns a strict JSON list of high-confidence mismatches, each
with a kind, what/where, severity, and an actionable fix sentence.

The prompt is engineered to MAXIMISE useful signal:
  - Few-shot examples show the difference between OK and BAD findings.
  - Hard guardrails forbid style/colour/font nitpicks (handled by other
    checkers / judge).
  - Capped at top-N by severity to keep the refine prompt readable.
  - Demands bbox or verbal anchor for every finding so the next refine
    iter can act on it deterministically.
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

logger = logging.getLogger("checkers.structural")


STRUCTURAL_PROMPT = """You are a STRICT visual diff critic. Compare two
academic-figure images and report only HIGH-CONFIDENCE structural
mismatches that the next render iteration can fix.

Image #1 — ORIGINAL figure (ground truth).
Image #2 — current SVG rendering attempt.

REFERENCE DATA (use these to be precise — do NOT invent content):
  • EXPECTED_TEXTS — Paddle OCR on the original. Every entry should
    appear in Image #2 at approximately the same location.
  • EXPECTED_ELEMENTS — raster icon slots and vector regions declared
    by the harness (with original-image bboxes). The preview should
    have content at each.

Categorise each issue into ONE of:
  1. "missing_text"     — a text in EXPECTED_TEXTS is not visible in #2
                           OR visible but in clearly wrong location.
  2. "missing_element"  — an EXPECTED_ELEMENTS slot is empty or has
                           clearly wrong content in #2.
  3. "structural"       — panel ordering / reading flow / overall
                           layout differs from original (e.g. groups
                           swapped, sections missing).
  4. "wrong_content"    — at a specific location, #2 shows something
                           clearly different from #1 (e.g. wrong icon
                           depicted, text content changed).
  5. "text_reflow"      — text content correct but spatial arrangement
                           wrong (was horizontal row → became vertical;
                           was left-of-B → became right-of-B).

For EACH issue, return:
  • "kind"        — one of the 5 above
  • "what"        — 1 concise sentence (no fluff)
  • "where"       — bbox [x1,y1,x2,y2] in ORIGINAL-image coords if
                    pinpointable, else short verbal anchor
                    ("top-left", "below the green panel", etc.)
  • "severity"    — "high" / "medium". Skip "low" entirely.
  • "fix"         — 1 imperative sentence the refine LLM will execute
                    ("add X at [bbox]", "move A so it sits left of B",
                    "replace icon at [bbox] with a magnifying-glass")

CRITICAL — bboxes are VISUAL ANCHORS, not absolute targets:
  • The bbox numbers in EXPECTED_TEXTS / EXPECTED_ELEMENTS came from
    OCR / SAM3 — they are noisy and may be off by 20-50 px. Use them
    only to LOCATE the visual region. Final correctness must be judged
    by what you see in Image #1.
  • In every "where" / "fix", say "near [bbox]" or describe the
    region verbally; never demand pixel-exact placement.
  • Refine LLM will use Image #1 as the source of truth — your fix
    sentence should help it FIND the spot, not lock it to numbers.

GUARDRAILS — these are critical:
  ✗ Do NOT flag pure-stylistic issues (colour, font family, gradient,
    shadow, opacity). Other checkers handle those.
  ✗ Do NOT flag minor pixel offsets (< 50 px). Position checker handles.
  ✗ Do NOT report the same issue twice with different wording.
  ✗ Do NOT invent content not visible in either image.
  ✗ Do NOT include findings you are < 70% confident about.
  ✓ DO ground every finding in something visible in BOTH images
    (or visibly missing from #2).
  ✓ DO use the EXPECTED_TEXTS / EXPECTED_ELEMENTS reference whenever
    relevant — citing an expected entry strengthens the finding.

CAP: max 6 findings, sorted highest severity first. If the preview
is essentially correct (no high/medium issues), return {"issues": []}.

GOOD EXAMPLE (specific, grounded, actionable):
  {"kind":"missing_text","what":"Original has the section title
  'Pressure Scoring' above the pink panel; preview lacks it.",
  "where":[480,72,640,98],"severity":"high",
  "fix":"add a <text> 'Pressure Scoring' centred at (560, 88) with
  font-size 16 bold dark-grey."}

BAD EXAMPLE (vague, unactionable):
  {"kind":"structural","what":"layout looks a bit off","severity":
  "medium","fix":"clean up the layout"}

REFERENCE DATA:
  EXPECTED_TEXTS (Paddle, top {NTEXT}, original-image coords):
{TEXT_LIST}

  EXPECTED_ELEMENTS (raster + vector placeholders):
{ELEM_LIST}

Return STRICT JSON only:
{"issues": [ {...}, {...}, ... ]}
"""


def _b64(path: Path, max_dim: int = 1100) -> str:
    """Encode image to base64 JPEG. Two images are sent per call so
    keep max_dim modest to stay under OpenRouter's vision-token budget and
    keep response time reasonable (~10-20s per call vs ~60s+ at 1500)."""
    img = Image.open(path).convert("RGB")
    if max(img.size) > max_dim:
        scale = max_dim / max(img.size)
        img = img.resize((int(img.width * scale), int(img.height * scale)),
                         Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def _bbox_of_paddle_entry(t: dict) -> list[float]:
    """Paddle entries are either {bbox:[x1,y1,x2,y2], ...} or the
    harness cache shape {x, y, w, h, ...}. Normalise."""
    if t.get("bbox"):
        bb = t["bbox"]
        return [float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])]
    if {"x", "y", "w", "h"} <= set(t.keys()):
        x, y, w, h = float(t["x"]), float(t["y"]), float(t["w"]), float(t["h"])
        return [x, y, x + w, y + h]
    return [0.0, 0.0, 0.0, 0.0]


def _format_texts(paddle_texts: list[dict] | None, cap: int = 30) -> str:
    if not paddle_texts:
        return "  (no Paddle text reference available)"
    lines = []
    for t in paddle_texts[:cap]:
        bb = _bbox_of_paddle_entry(t)
        bb_str = f"[{int(bb[0])},{int(bb[1])},{int(bb[2])},{int(bb[3])}]"
        lines.append(f"    - '{t.get('text','').strip()[:40]}' @ {bb_str}")
    if len(paddle_texts) > cap:
        lines.append(f"    ... and {len(paddle_texts) - cap} more")
    return "\n".join(lines)


def _format_elements(labels: dict | None) -> str:
    if not labels:
        return "  (no labels.json reference available)"
    lines = []
    for r in labels.get("raster", [])[:20]:
        bb = r.get("bbox") or [0, 0, 0, 0]
        bb_str = f"[{int(bb[0])},{int(bb[1])},{int(bb[2])},{int(bb[3])}]"
        kind = r.get("kind", "?")
        desc = (r.get("desc") or r.get("simple_desc") or "")[:40]
        lines.append(f"    - {r.get('label','RAS_?')} ({kind}) @ {bb_str}"
                     f" — {desc}")
    for v in labels.get("vector", [])[:10]:
        bb = v.get("bbox") or [0, 0, 0, 0]
        bb_str = f"[{int(bb[0])},{int(bb[1])},{int(bb[2])},{int(bb[3])}]"
        kind = v.get("kind", "?")
        desc = (v.get("desc") or "")[:40]
        lines.append(f"    - {v.get('label','VEC_?')} ({kind}) @ {bb_str}"
                     f" — {desc}")
    return "\n".join(lines) if lines else "  (no labelled elements)"


def _extract_json(text: str) -> dict | None:
    text = text.strip()
    m = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if m:
        text = m.group(1).strip()
    for s in range(len(text)):
        if text[s] != "{":
            continue
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
    return None


def check(
    original_png: Path,
    preview_png: Path,
    paddle_texts: list[dict] | None = None,
    labels: dict | None = None,
    model: str = "openai/gpt-5.5",
    cache_path: Path | None = None,
) -> list[dict]:
    """Run the structural VLM critic.

    Returns a list of fix items shaped like the other checkers:
      {"kind": "missing_text"|..., "fix": "...", "where": [...],
       "severity": "high"|"medium", "what": "..."}
    """
    if cache_path and cache_path.exists():
        try:
            return json.loads(cache_path.read_text())
        except Exception:
            pass

    prompt = (STRUCTURAL_PROMPT
              .replace("{NTEXT}", str(min(30, len(paddle_texts or []))))
              .replace("{TEXT_LIST}", _format_texts(paddle_texts))
              .replace("{ELEM_LIST}", _format_elements(labels)))
    msgs = [{
        "role": "user",
        "content": [
            {"type": "text", "text": "Image #1 — ORIGINAL figure:"},
            {"type": "image_url", "image_url":
                {"url": f"data:image/jpeg;base64,{_b64(original_png)}"}},
            {"type": "text", "text": "Image #2 — CURRENT preview:"},
            {"type": "image_url", "image_url":
                {"url": f"data:image/jpeg;base64,{_b64(preview_png)}"}},
            {"type": "text", "text": prompt},
        ],
    }]
    try:
        # max_tokens=2000 is plenty (output is ≤ 6 issues × ~50 tokens)
        # and limits worst-case generation time.
        resp = call_model(msgs, model=model, max_tokens=2000,
                          label="stage6_structural_critic")
    except Exception as e:
        logger.warning("structural critic api failed: %s", e)
        return []

    parsed = _extract_json(resp)
    if not parsed or "issues" not in parsed:
        logger.warning("structural critic returned non-JSON or no issues key")
        return []

    fixes = []
    for it in parsed.get("issues", [])[:6]:
        kind = it.get("kind", "structural")
        sev = it.get("severity", "medium").lower()
        if sev not in ("high", "medium"):
            continue
        fixes.append({
            "kind": kind,
            "severity": sev,
            "what": it.get("what", "")[:200],
            "where": it.get("where"),
            "fix": it.get("fix", "")[:240],
        })

    if cache_path:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(fixes, indent=2))
        except Exception:
            pass

    return fixes
