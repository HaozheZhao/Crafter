"""Text-recovery — VLM-based fix for Paddle OCR errors on
math symbols, Greek letters, calligraphy, scripts, rotated text.

Pipeline:
  1. Heuristically flag suspicious Paddle entries
        - text length ≤ 2 chars (likely single Greek/math symbol)
        - matches ambiguous singleton: l L I O Q 1 0 . , -
        - bbox aspect ratio > 4 or < 0.25 (probably rotated)
        - bbox area < 600 px (tiny — easy to misread)
        - text contains '?' or empty after strip
  2. For each suspicious entry, crop the region from the ORIGINAL
     and ask gpt-5.5: "what is this actually?" + style hint.
  3. Strong-replace the Paddle entry with the corrected text +
     attach a style_hint dict.
  4. Cache the recovered list per-image.

Output schema (each replacement):
  {
    "text": "λ",                         # corrected text (unicode)
    "style_hint": "italic_math",          # render hint
    "font_family": "Cambria Math",        # font suggestion
    "category": "greek" | "math_symbol" | "ascii" | "cursive" | "other",
    "confidence": 0.92,                   # VLM confidence
    "original_paddle_text": "L",          # what Paddle returned
  }

The corrected list is then used by SKELETON_PROMPT (compose.py)
in place of the raw Paddle output.
"""
from __future__ import annotations
import os

import base64
import io
import json
import logging
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image

REPO = Path(os.environ.get("CRAFTER_HOME", str(Path(__file__).resolve().parents[2])))

from crafter.editor.approach.iter_svg_fix import call_model  # noqa: E402

logger = logging.getLogger("checkers.text_recovery")


RECOVERY_PROMPT = """Look at the cropped region from an academic figure.
Paddle OCR read this region as: "{ORIG_TEXT}"

Your task:
1. State what character / word / symbol this region ACTUALLY contains.
2. Pay special attention to:
   • Greek letters (α β γ δ λ μ π σ θ φ ω ψ ε ζ η ξ ρ τ υ χ Σ Ω Λ Φ Π Δ Θ Γ ...)
   • Math symbols (∑ ∏ ∫ ∂ ∇ ≤ ≥ ≠ ≈ ∞ ± × ÷ → ← ↑ ↓ √ ∈ ∉ ⊆ ⊕ ⊗ ...)
   • LaTeX-style notation (subscripts, superscripts, italic var names)
   • Cursive / italic / bold styling
3. If Paddle was already correct, return the same text.
4. If the region is empty / unreadable / pure background, return null.

Return STRICT JSON:
{
  "text": "λ",                           // the actual character / word (unicode if symbol)
  "category": "greek",                    // one of: ascii, greek, math_symbol, cursive, mixed, other
  "style_hint": "italic_math",            // one of: normal, italic, bold, italic_math, cursive, math_symbol, null
  "font_family": "Cambria Math",          // suggested font (or null)
  "confidence": 0.92                      // 0..1 self-rating
}

GUARDRAILS:
  • Be CONSERVATIVE — if uncertain whether Paddle was right, return Paddle's text.
  • Never invent text not visible in the crop.
  • For Greek/math, use the unicode codepoint (λ not "lambda").
  • style_hint of "italic_math" means italic Latin/Greek var (e.g. v, x, λ, μ).
"""


SUSPICIOUS_AMBIGUOUS = set("lLIO0Q1.,-_:;'\"`'‘’")


def _bbox_xywh(t: dict) -> tuple[float, float, float, float]:
    """Return (x, y, w, h) regardless of paddle entry shape."""
    if {"x", "y", "w", "h"} <= set(t.keys()):
        return float(t["x"]), float(t["y"]), float(t["w"]), float(t["h"])
    if t.get("bbox"):
        bb = t["bbox"]
        return (float(bb[0]), float(bb[1]),
                float(bb[2]) - float(bb[0]),
                float(bb[3]) - float(bb[1]))
    return 0.0, 0.0, 0.0, 0.0


def _is_suspicious(t: dict) -> tuple[bool, str]:
    """Heuristic: should this Paddle entry be VLM-verified?"""
    txt = (t.get("text") or "").strip()
    if not txt:
        return False, "empty"
    x, y, w, h = _bbox_xywh(t)
    area = w * h
    ar = w / h if h > 0 else 0

    # 1) very short text → very likely Greek/math symbol misread
    if len(txt) <= 2:
        return True, f"short_text len={len(txt)}"
    # 2) ambiguous singleton characters
    if len(txt) == 1 and txt in SUSPICIOUS_AMBIGUOUS:
        return True, f"ambiguous_singleton '{txt}'"
    # 3) anomalous aspect ratio (rotated or very thin/wide)
    if ar > 0 and (ar > 4 or ar < 0.25):
        return True, f"anomalous_aspect ar={ar:.2f}"
    # 4) tiny bbox
    if 0 < area < 600:
        return True, f"tiny_bbox area={area:.0f}"
    # 5) contains math-symbol markers
    if re.search(r"[\\$°°{}\[\]<>]", txt):
        return True, f"contains_math_marker"
    # 6) explicit unsure marker
    if "?" in txt:
        return True, "contains_question_mark"
    return False, "ok"


def _b64_crop(img: Image.Image, bbox: tuple, pad: int = 6,
              max_dim: int = 256) -> str:
    x, y, w, h = bbox
    L = max(0, int(x - pad))
    T = max(0, int(y - pad))
    R = min(img.width, int(x + w + pad))
    B = min(img.height, int(y + h + pad))
    crop = img.crop((L, T, R, B)).convert("RGB")
    if max(crop.size) > max_dim:
        scale = max_dim / max(crop.size)
        crop = crop.resize(
            (int(crop.width * scale), int(crop.height * scale)),
            Image.LANCZOS)
    # Upsample tiny crops so the symbol is visible
    if max(crop.size) < 64:
        s = 64 / max(crop.size)
        crop = crop.resize(
            (int(crop.width * s), int(crop.height * s)),
            Image.NEAREST)
    buf = io.BytesIO()
    crop.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


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


def _verify_one(img: Image.Image, paddle_entry: dict,
                model: str = "openai/gpt-5.5") -> dict | None:
    """Run one VLM call to verify a single suspicious entry."""
    bbox = _bbox_xywh(paddle_entry)
    orig_text = (paddle_entry.get("text") or "").strip()
    try:
        b64 = _b64_crop(img, bbox)
    except Exception as e:
        logger.warning("text_recovery crop failed: %s", e)
        return None
    prompt = RECOVERY_PROMPT.replace("{ORIG_TEXT}",
                                     orig_text.replace('"', '\\"'))
    msgs = [{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url":
                {"url": f"data:image/png;base64,{b64}"}},
            {"type": "text", "text": prompt},
        ],
    }]
    try:
        # text-only output → small max_tokens, fast
        resp = call_model(msgs, model=model, max_tokens=400,
                          label="stage6_text_recovery")
    except Exception as e:
        logger.warning("text_recovery api failed: %s", e)
        return None
    parsed = _extract_json(resp) or {}
    if not parsed.get("text"):
        return None
    return {
        "text": str(parsed["text"]),
        "category": str(parsed.get("category", "other"))[:30],
        "style_hint": str(parsed.get("style_hint", "normal"))[:30],
        "font_family": parsed.get("font_family"),
        "confidence": float(parsed.get("confidence", 0.0)),
        "original_paddle_text": orig_text,
    }


def recover(
    original_png: Path,
    paddle_texts: list[dict],
    cache_path: Path | None = None,
    model: str = "openai/gpt-5.5",
    max_workers: int = 4,
    apply_threshold: float = 0.85,
) -> dict:
    """Verify suspicious Paddle entries, return:
      {
        "corrected": [paddle_entry, ...],   # full list with corrections applied
        "changes":   [{idx, old, new}, ...], # what was changed
        "skipped":   [{idx, reason}, ...],   # not verified
      }

    `apply_threshold` — only apply correction if VLM confidence ≥ this.
    """
    if cache_path and cache_path.exists():
        try:
            return json.loads(cache_path.read_text())
        except Exception:
            pass

    if not paddle_texts:
        return {"corrected": [], "changes": [], "skipped": []}

    # 1) Flag suspicious entries
    suspicious = []
    for i, t in enumerate(paddle_texts):
        sus, reason = _is_suspicious(t)
        if sus:
            suspicious.append((i, reason))

    if not suspicious:
        out = {"corrected": list(paddle_texts), "changes": [], "skipped": []}
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(out, indent=2))
        return out

    logger.info("text_recovery: %d/%d suspicious entries to verify",
                len(suspicious), len(paddle_texts))

    img = Image.open(original_png).convert("RGB")
    corrected = list(paddle_texts)
    changes: list[dict] = []
    skipped: list[dict] = []

    # 2) Verify in parallel — OpenRouter can handle ~4 concurrent
    def _job(idx_reason):
        idx, reason = idx_reason
        verdict = _verify_one(img, paddle_texts[idx], model=model)
        return idx, reason, verdict

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(_job, sr) for sr in suspicious]
        for fut in as_completed(futs):
            idx, reason, verdict = fut.result()
            if verdict is None:
                skipped.append({"idx": idx, "reason": reason,
                                "why": "no verdict"})
                continue
            old = (paddle_texts[idx].get("text") or "").strip()
            new = verdict["text"].strip()
            if verdict["confidence"] < apply_threshold:
                skipped.append({"idx": idx, "reason": reason,
                                "why": f"low_conf {verdict['confidence']}"})
                continue
            if old == new and not verdict.get("style_hint"):
                # Paddle was correct, no style change needed
                continue
            # Apply correction (strong replace)
            corrected[idx] = {
                **paddle_texts[idx],
                "text": new,
                "style_hint": verdict.get("style_hint"),
                "font_family": verdict.get("font_family"),
                "category": verdict.get("category"),
                "recovery_confidence": verdict.get("confidence"),
                "original_paddle_text": old,
            }
            changes.append({
                "idx": idx,
                "old": old,
                "new": new,
                "category": verdict.get("category"),
                "style_hint": verdict.get("style_hint"),
                "font_family": verdict.get("font_family"),
                "confidence": verdict.get("confidence"),
                "reason_flagged": reason,
            })

    out = {"corrected": corrected, "changes": changes, "skipped": skipped}
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(out, indent=2))
    logger.info("text_recovery: applied %d corrections, %d skipped",
                len(changes), len(skipped))
    return out
