"""Iterative SVG fix harness.

Tests two strategies for using a vision-capable LLM (default gpt-5.5)
to iteratively repair an existing R2E-generated SVG:

  Mode B (end-to-end, simpler):
     Give model: original PNG + current preview PNG + current SVG.
     Ask it to output a fixed SVG. One round per iteration.

  Mode A (grounded, focused):
     Step 1 — error listing: ask model for a JSON list of [bbox, kind,
              description] per visible problem, comparing originals
              vs preview.
     Step 2 — mark each error bbox on a copy of the preview using OpenCV.
     Step 3 — ask model to fix the marked errors only, given the
              marked-error preview + original + current SVG.

Both modes can run multiple iterations; each iteration:
  - reads the current SVG file,
  - renders it to a preview PNG (cairosvg or firefox-headless if avail),
  - calls the chosen model,
  - extracts the new SVG from the response,
  - validates it parses as XML; if not, falls back to previous,
  - writes it as iter_{n}.svg + iter_{n}.png in the output dir.

Usage:
  python iter_svg_fix.py mode_b  --orig PATH --svg PATH --out DIR --iters 2
  python iter_svg_fix.py mode_a  --orig PATH --svg PATH --out DIR --iters 2
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

import requests
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


GATEWAY = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "openai/gpt-5.5"


def encode_image_b64(path: str, max_dim: int = 1500) -> str:
    img = Image.open(path).convert("RGB")
    if max(img.size) > max_dim:
        ratio = max_dim / max(img.size)
        img = img.resize((int(img.width * ratio), int(img.height * ratio)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def render_svg(svg_path: str, png_path: str, w: int, h: int) -> bool:
    """Render SVG -> PNG. Returns True on success."""
    # Try Firefox headless first (best fidelity)
    import subprocess
    abs_svg = str(Path(svg_path).resolve())
    abs_png = str(Path(png_path).resolve())
    try:
        r = subprocess.run(
            ["firefox", "--headless",
             f"--screenshot={abs_png}",
             f"--window-size={w},{h}",
             f"file://{abs_svg}"],
            capture_output=True, timeout=30,
        )
        if r.returncode == 0 and Path(abs_png).exists():
            # Resize to exact dims if needed
            img = Image.open(abs_png)
            if img.size != (w, h):
                img = img.resize((w, h), Image.LANCZOS)
                img.save(abs_png)
            return True
    except FileNotFoundError:
        pass
    except Exception as exc:
        logger.debug("Firefox render failed: %s", exc)
    # cairosvg fallback
    try:
        import cairosvg
        cairosvg.svg2png(url=abs_svg, write_to=abs_png, output_width=w, output_height=h)
        return Path(abs_png).exists()
    except Exception as exc:
        logger.warning("cairosvg failed: %s", exc)
        return False


_HTML_ONLY_ENTITIES = {
    # Whitespace + dashes + quotes that LLMs sometimes emit as HTML
    # entities (HTML knows them, SVG/XML do not — cairosvg refuses to
    # parse and drops the SVG).
    "&nbsp;": " ",
    "&ensp;": " ",
    "&emsp;": " ",
    "&thinsp;": " ",
    "&ndash;": "–",
    "&mdash;": "—",
    "&hellip;": "…",
    "&middot;": "·",
    "&bull;": "•",
    "&copy;": "©",
    "&reg;": "®",
    "&trade;": "™",
    "&times;": "×",
    "&divide;": "÷",
    "&deg;": "°",
    "&plusmn;": "±",
    "&micro;": "µ",
    "&para;": "¶",
    "&sect;": "§",
    "&laquo;": "«",
    "&raquo;": "»",
    "&lsquo;": "‘",
    "&rsquo;": "’",
    "&ldquo;": "“",
    "&rdquo;": "”",
    "&prime;": "′",
    "&Prime;": "″",
    "&larr;": "←",
    "&uarr;": "↑",
    "&rarr;": "→",
    "&darr;": "↓",
    "&harr;": "↔",
    "&infin;": "∞",
    "&asymp;": "≈",
    "&ne;": "≠",
    "&le;": "≤",
    "&ge;": "≥",
}


def _sanitize_svg_entities(svg: str) -> str:
    """Defensive cleanup: replace HTML-only named entities with their
    Unicode equivalents. SVG/XML only natively define
    {amp, lt, gt, quot, apos}; LLMs occasionally emit named-entity
    spellings like &nbsp; that crash cairosvg with
    "undefined entity" and silently drop the candidate.
    """
    for ent, ch in _HTML_ONLY_ENTITIES.items():
        if ent in svg:
            svg = svg.replace(ent, ch)
    return svg


def extract_svg_from_response(text: str) -> Optional[str]:
    text = text.strip()
    # Strip markdown fences
    m = re.search(r"```(?:svg|xml|html)?\s*\n?(.*?)```", text, re.DOTALL)
    if m:
        text = m.group(1).strip()
    # Find <svg ...>...</svg>
    m = re.search(r"(<svg[\s\S]*?</svg>)", text)
    if m:
        return _sanitize_svg_entities(m.group(1))
    return None


# Token usage instrumentation. Module-global accumulator that callers
# inspect via get_usage_log() at end-of-run and dump alongside outputs.
_USAGE_LOG: list[dict] = []


def reset_usage_log() -> None:
    """Clear the usage log (call at start of a per-image run)."""
    _USAGE_LOG.clear()


def get_usage_log() -> list[dict]:
    """Return a copy of the current usage log."""
    return list(_USAGE_LOG)


def _record_usage(model: str, usage: dict | None, elapsed: float,
                  label: str | None = None) -> None:
    """Append one LLM call's token usage to the global log."""
    if not usage:
        return
    _USAGE_LOG.append({
        "model": model,
        "label": label or "unlabeled",
        "elapsed_s": round(elapsed, 2),
        "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
        "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
        "total_tokens": int(usage.get("total_tokens", 0) or 0),
    })


def call_model(messages: list, model: str = DEFAULT_MODEL,
               max_tokens: int = 32000,
               temperature: float | None = None,
               max_retries: int = 3,
               retry_backoff: float = 8.0,
               label: str | None = None) -> str:
    """Call LLM chat completion. Now delegates to editor.providers.llm.

    Backward-compat shim: existing callers pass kwargs unchanged. The actual
    HTTP call goes through editor.providers.llm.get_default_llm() which is
    pluggable (OpenRouter by default; OpenAI / custom via set_default_llm).

    `label` tags this call for token-usage attribution in
    phase2_summary.json (recorded into module-global _USAGE_LOG).
    """
    # Lazy import to avoid editor depending on this module at import time.
    # Ensure package root is on sys.path (subprocess callers may not have it).
    import sys as _sys
    from pathlib import Path as _Path
    _root = _Path(__file__).resolve().parents[2]
    if str(_root) not in _sys.path:
        _sys.path.insert(0, str(_root))
    from crafter.shared.providers.llm import get_default_llm  # noqa: E402
    provider = get_default_llm()
    resp = provider.chat(messages, model=model, max_tokens=max_tokens,
                          temperature=temperature, label=label)
    # Mirror provider's usage into the module-global log
    _record_usage(resp.model, {
        "prompt_tokens": resp.prompt_tokens,
        "completion_tokens": resp.completion_tokens,
        "total_tokens": resp.total_tokens,
    }, resp.elapsed_s, label)
    return resp.text


def parse_json_response(text: str) -> Optional[dict]:
    text = text.strip()
    m = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if m:
        text = m.group(1).strip()
    # Try the whole thing
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Find first { ... } that parses
    for start in range(len(text)):
        if text[start] == '{':
            depth = 0
            for end in range(start, len(text)):
                if text[end] == '{':
                    depth += 1
                elif text[end] == '}':
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(text[start:end+1])
                        except json.JSONDecodeError:
                            break
    return None


# ============================================================
# MODE B: End-to-end fix from (original + current preview + SVG)
# ============================================================

MODE_B_PROMPT = """You will receive:
  • image #1 — the ORIGINAL academic figure to reproduce
  • image #2 — the CURRENT rendered preview of an SVG attempt
  • the CURRENT SVG source text below

Your task: find the visible differences between #1 and #2 and produce a
CORRECTED SVG that brings #2 closer to #1 — preserving everything that
already matches and modifying ONLY what needs to change.

CRITICAL — DO NOT TOUCH <image> ELEMENTS:
  • Every <image> element references a raster icon already extracted from
    the original figure. PRESERVE every <image> element verbatim including
    its `href` attribute.
  • You MAY move an <image> by changing x/y/width/height, but NEVER
    remove or modify the href, NEVER replace it with <rect> or vector
    shapes, NEVER drop the element.
  • The href value may appear as a short token like `__ICON_REF_007__` —
    that is a stash marker; treat it as opaque and leave it intact.

PRIORITY of fixes (apply roughly in this order):
  1. Missing or incorrect text labels (especially subscripts, superscripts,
     Greek letters, and tiny labels like ×L, Concat, Q/K/V, digits in grids).
  2. Misplaced or overlapping text — snap text to the right anchor.
  3. <rect>s that should have been <image>s (the input may already have the
     <image>; if not, leave a placeholder comment, do NOT invent base64).
  4. Wrong arrow endpoints / wrong colors / missing arrows.
  5. Misaligned shapes (>5px off).
  6. Style polish (font weights, gradients, drop-shadows) ONLY if cheap.

OUTPUT: a single complete SVG document, well-formed XML, starting with
`<svg ...>` and ending with `</svg>`. No prose, no markdown fences in
the output, no commentary. Preserve all element ids you can. Preserve
every <image> element as described above.

CURRENT SVG:
```svg
{svg}
```
"""


_DATA_URI_RE = re.compile(
    r'(href|xlink:href)="(data:image/[a-z+]+;base64,[A-Za-z0-9+/=]+)"')


def _stash_data_uris(svg: str):
    """Replace each long data: URI in <image href=...> with a short token.

    LLMs cannot reliably preserve 100KB+ base64 strings in their output —
    they truncate or drop them, deleting all icons from the SVG. We swap
    each URI for a short __ICON_REF_NNN__ token before sending to the
    model, then restore by token after the response."""
    stash = {}
    def repl(m):
        attr, uri = m.group(1), m.group(2)
        token = f"__ICON_REF_{len(stash):03d}__"
        stash[token] = uri
        return f'{attr}="{token}"'
    return _DATA_URI_RE.sub(repl, svg), stash


def _restore_data_uris(svg: str, stash: dict) -> str:
    for token, uri in stash.items():
        svg = svg.replace(token, uri)
    return svg


def mode_b_iter(original_path: str, svg_path: str, output_dir: str,
                model: str, n_iters: int, w: int, h: int) -> dict:
    """Run mode B for n_iters iterations. Returns dict with per-iter info."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    record: list[dict] = []

    current_svg = Path(svg_path).read_text(encoding="utf-8")
    # Initial preview
    init_png = out / "iter_0_preview.png"
    if not Path(svg_path).with_suffix(".png").exists():
        render_svg(svg_path, str(init_png), w, h)
    else:
        init_png = Path(svg_path).with_name("preview_r0.png")
        if init_png.exists():
            from shutil import copy
            copy(init_png, out / "iter_0_preview.png")
            init_png = out / "iter_0_preview.png"

    for i in range(1, n_iters + 1):
        prev_png = out / f"iter_{i-1}_preview.png"
        if not prev_png.exists():
            # render
            cur_svg_path = out / f"iter_{i-1}.svg"
            cur_svg_path.write_text(current_svg, encoding="utf-8")
            render_svg(str(cur_svg_path), str(prev_png), w, h)

        logger.info("MODE_B iter %d — calling %s", i, model)
        t0 = time.time()
        b64_orig = encode_image_b64(original_path)
        b64_prev = encode_image_b64(str(prev_png))
        # Stash long base64 data: URIs to short tokens so the LLM can fit the
        # SVG in its context window AND so it doesn't drop/truncate the URIs
        # in its response (which deletes all the icons).
        compact_svg, stash = _stash_data_uris(current_svg)
        if stash:
            logger.info("  stashed %d data URIs", len(stash))
        prompt = MODE_B_PROMPT.format(svg=compact_svg)
        if len(prompt) > 110000:
            logger.warning("prompt large: %d chars; SVG body is %d", len(prompt), len(compact_svg))
        msgs = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "Image #1 — ORIGINAL:"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_orig}"}},
                {"type": "text", "text": "Image #2 — CURRENT preview of the SVG:"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_prev}"}},
                {"type": "text", "text": prompt},
            ],
        }]
        try:
            response = call_model(msgs, model=model, max_tokens=64000)
        except Exception as exc:
            logger.error("API call failed: %s", exc)
            record.append({"iter": i, "status": "api_error", "msg": str(exc)})
            break
        elapsed = time.time() - t0

        new_svg = extract_svg_from_response(response)
        if new_svg and stash:
            new_svg = _restore_data_uris(new_svg, stash)
        if new_svg:
            # Safety guards: the LLM sometimes "cheats" by replacing all
            # vector elements with cropped raster tiles. Reject if any of:
            #   • image count drops > 50%
            #   • text count drops > 30%
            #   • rect count drops > 50%
            old_img = current_svg.count("<image")
            new_img = new_svg.count("<image")
            old_txt = current_svg.count("<text")
            new_txt = new_svg.count("<text")
            old_rect = current_svg.count("<rect")
            new_rect = new_svg.count("<rect")
            if old_img > 0 and new_img < old_img * 0.5:
                logger.warning("iter %d: image count dropped %d → %d (>50%%); "
                               "rejecting this iter output", i, old_img, new_img)
                new_svg = None
            elif old_txt > 0 and new_txt < old_txt * 0.7:
                logger.warning("iter %d: text count dropped %d → %d (>30%%); "
                               "model is replacing vector with raster — "
                               "rejecting this iter", i, old_txt, new_txt)
                new_svg = None
            elif old_rect > 0 and new_rect < old_rect * 0.5:
                logger.warning("iter %d: rect count dropped %d → %d (>50%%); "
                               "rejecting this iter", i, old_rect, new_rect)
                new_svg = None
        if not new_svg:
            logger.warning("iter %d: model didn't return a parseable SVG", i)
            record.append({"iter": i, "status": "no_svg", "elapsed": elapsed,
                          "response_chars": len(response)})
            (out / f"iter_{i}_response.txt").write_text(response, encoding="utf-8")
            break
        # Validate XML
        try:
            ET.fromstring(new_svg)
            valid = True
        except ET.ParseError as exc:
            logger.warning("iter %d: SVG malformed: %s", i, exc)
            valid = False

        if not valid:
            record.append({"iter": i, "status": "malformed", "elapsed": elapsed,
                          "response_chars": len(response)})
            (out / f"iter_{i}_bad.svg").write_text(new_svg, encoding="utf-8")
            (out / f"iter_{i}_response.txt").write_text(response, encoding="utf-8")
            continue  # try next iter starting from same SVG

        svg_out = out / f"iter_{i}.svg"
        svg_out.write_text(new_svg, encoding="utf-8")
        png_out = out / f"iter_{i}_preview.png"
        if render_svg(str(svg_out), str(png_out), w, h):
            logger.info("iter %d: rendered to %s", i, png_out)
        record.append({"iter": i, "status": "ok", "elapsed": elapsed,
                      "svg_chars": len(new_svg), "svg_path": str(svg_out),
                      "png_path": str(png_out)})
        current_svg = new_svg

    summary_path = out / "mode_b_summary.json"
    summary_path.write_text(json.dumps({"mode": "B", "model": model, "iters": record},
                                       indent=2))
    logger.info("MODE_B done: %d iters; summary at %s", len(record), summary_path)
    return {"mode": "B", "iters": record}


# ============================================================
# MODE A: Error grounding -> mark -> per-region fix
# ============================================================

ERROR_LIST_PROMPT = """You will see two images:
  • image #1 — the ORIGINAL academic figure to reproduce
  • image #2 — a CURRENT rendered preview of an SVG attempt

Your task: list the VISIBLE problems in image #2 vs image #1, with each
problem grounded to a bounding box in image-coordinates of image #2.

Problem categories to use (one per entry):
  • "missing_text"     — text in original is absent in preview
  • "wrong_text"       — text content differs (incl. subscripts/Greek)
  • "overlap_text"     — two texts overlap or ghost
  • "missing_icon"     — embedded raster (heatmap, grid, photo, plot) is absent
  • "wrong_arrow"      — arrow endpoint / direction wrong
  • "wrong_color"      — fill color clearly wrong
  • "wrong_shape"      — shape geometry clearly wrong
  • "alignment_drift"  — element shifted >10px from original

Return STRICT JSON:
{{
  "image_size": [W, H],         // dimensions of image #2 you observed
  "problems": [
    {{
      "id": "p1",
      "category": "missing_icon",
      "bbox": [x1, y1, x2, y2],  // pixel coords in image #2
      "description": "one sentence",
      "fix_hint": "what should be there based on image #1"
    }},
    ...
  ]
}}

Hard limits:
  • At most 12 problems. Pick the ones that affect visual fidelity most.
  • Bboxes must be within the image bounds.
  • Return ONLY the JSON object, no prose, no markdown fences.
"""


PER_REGION_FIX_PROMPT = """You will see:
  • image #1 — the ORIGINAL academic figure
  • image #2 — the CURRENT preview, with PROBLEM REGIONS marked by
    coloured bboxes labelled p1, p2, …

Below the images is the CURRENT SVG plus a JSON list of the marked
problems with their categories and fix-hints.

Your task: produce a CORRECTED SVG that fixes EACH problem in the list
while leaving the rest of the SVG unchanged.

OUTPUT RULES:
  • A single, complete, well-formed SVG document.
  • Same root attributes (width / height / viewBox).
  • For "missing_icon" problems: replace the flat <rect> at that bbox
    with an <image href=""> placeholder (the composer will inject the
    raster crop later).
  • For "missing_text" / "wrong_text": insert/replace the <text> element
    at the correct position with the correct content.
  • Do NOT delete elements outside the marked problem bboxes.
  • No commentary, no markdown fences. Just the SVG.

PROBLEMS:
```json
{problems}
```

CURRENT SVG:
```svg
{svg}
```
"""


def mark_errors_on_image(image_path: str, problems: list, output_path: str) -> bool:
    """Draw labeled bboxes on the image for each problem."""
    try:
        import cv2
        import numpy as np
    except ImportError:
        return False
    img = cv2.imread(image_path)
    if img is None:
        return False
    H, W = img.shape[:2]
    colours = {
        "missing_text":     (0, 0, 255),       # red
        "wrong_text":       (0, 100, 255),     # orange
        "overlap_text":     (0, 200, 255),     # yellow
        "missing_icon":     (255, 0, 255),     # magenta
        "wrong_arrow":      (255, 0, 0),       # blue
        "wrong_color":      (0, 255, 255),     # cyan
        "wrong_shape":      (0, 255, 0),       # green
        "alignment_drift":  (255, 255, 0),     # cyan-light
    }
    for p in problems:
        try:
            x1, y1, x2, y2 = [int(v) for v in p.get("bbox", [0, 0, 0, 0])]
        except (ValueError, TypeError):
            continue
        x1 = max(0, min(W - 1, x1))
        y1 = max(0, min(H - 1, y1))
        x2 = max(0, min(W, x2))
        y2 = max(0, min(H, y2))
        col = colours.get(p.get("category", ""), (200, 200, 200))
        cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
        label = f"{p.get('id','')}-{p.get('category','')[:6]}"
        # Background for label
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ly = max(0, y1 - 4)
        cv2.rectangle(img, (x1, ly - th - 2), (x1 + tw + 4, ly + 2), col, -1)
        cv2.putText(img, label, (x1 + 2, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(output_path, img)
    return True


def mode_a_iter(original_path: str, svg_path: str, output_dir: str,
                model: str, n_iters: int, w: int, h: int) -> dict:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    record: list[dict] = []

    current_svg = Path(svg_path).read_text(encoding="utf-8")

    init_png = out / "iter_0_preview.png"
    if Path(svg_path).with_name("preview_r0.png").exists():
        from shutil import copy
        copy(Path(svg_path).with_name("preview_r0.png"), init_png)
    else:
        render_svg(svg_path, str(init_png), w, h)

    for i in range(1, n_iters + 1):
        prev_png = out / f"iter_{i-1}_preview.png"
        if not prev_png.exists():
            cur_svg_path = out / f"iter_{i-1}.svg"
            cur_svg_path.write_text(current_svg, encoding="utf-8")
            render_svg(str(cur_svg_path), str(prev_png), w, h)

        # Step 1 — error listing
        logger.info("MODE_A iter %d step 1: error listing", i)
        b64_orig = encode_image_b64(original_path)
        b64_prev = encode_image_b64(str(prev_png))
        msgs = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "Image #1 — ORIGINAL:"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_orig}"}},
                {"type": "text", "text": "Image #2 — CURRENT preview:"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_prev}"}},
                {"type": "text", "text": ERROR_LIST_PROMPT},
            ],
        }]
        try:
            t0 = time.time()
            err_resp = call_model(msgs, model=model, max_tokens=8000)
            err_elapsed = time.time() - t0
        except Exception as exc:
            record.append({"iter": i, "status": "api_error_step1", "msg": str(exc)})
            break
        problems_data = parse_json_response(err_resp) or {}
        problems = problems_data.get("problems", []) or []
        (out / f"iter_{i}_problems.json").write_text(json.dumps(problems_data, indent=2))
        logger.info("  found %d problems in %.1fs", len(problems), err_elapsed)
        if not problems:
            logger.info("  no problems reported; stopping")
            record.append({"iter": i, "status": "no_problems", "elapsed_step1": err_elapsed})
            break

        # Step 2 — mark errors on preview
        marked_png = out / f"iter_{i}_marked.png"
        ok = mark_errors_on_image(str(prev_png), problems, str(marked_png))
        if not ok:
            logger.warning("  marking failed; using unmarked preview")
            from shutil import copy
            copy(prev_png, marked_png)

        # Step 3 — per-region fix
        logger.info("MODE_A iter %d step 3: per-region fix", i)
        b64_marked = encode_image_b64(str(marked_png))
        fix_prompt = PER_REGION_FIX_PROMPT.format(
            problems=json.dumps(problems, indent=2),
            svg=current_svg,
        )
        msgs = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "Image #1 — ORIGINAL:"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_orig}"}},
                {"type": "text", "text": "Image #2 — MARKED preview (problems labelled p1, p2, ...):"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_marked}"}},
                {"type": "text", "text": fix_prompt},
            ],
        }]
        try:
            t1 = time.time()
            fix_resp = call_model(msgs, model=model, max_tokens=64000)
            fix_elapsed = time.time() - t1
        except Exception as exc:
            record.append({"iter": i, "status": "api_error_step3", "msg": str(exc),
                          "n_problems": len(problems)})
            break
        new_svg = extract_svg_from_response(fix_resp)
        if not new_svg:
            (out / f"iter_{i}_fix_response.txt").write_text(fix_resp, encoding="utf-8")
            record.append({"iter": i, "status": "no_svg_step3",
                          "n_problems": len(problems),
                          "elapsed_step1": err_elapsed, "elapsed_step3": fix_elapsed})
            break
        # Validate
        try:
            ET.fromstring(new_svg)
            valid = True
        except ET.ParseError as exc:
            valid = False
            logger.warning("  iter %d malformed SVG: %s", i, exc)
        if not valid:
            (out / f"iter_{i}_bad.svg").write_text(new_svg, encoding="utf-8")
            record.append({"iter": i, "status": "malformed",
                          "n_problems": len(problems),
                          "elapsed_step1": err_elapsed, "elapsed_step3": fix_elapsed})
            continue

        svg_out = out / f"iter_{i}.svg"
        svg_out.write_text(new_svg, encoding="utf-8")
        png_out = out / f"iter_{i}_preview.png"
        render_svg(str(svg_out), str(png_out), w, h)
        record.append({"iter": i, "status": "ok",
                      "n_problems": len(problems),
                      "elapsed_step1": err_elapsed, "elapsed_step3": fix_elapsed,
                      "svg_chars": len(new_svg)})
        current_svg = new_svg

    summary_path = out / "mode_a_summary.json"
    summary_path.write_text(json.dumps({"mode": "A", "model": model, "iters": record},
                                       indent=2))
    logger.info("MODE_A done: %d iters; summary at %s", len(record), summary_path)
    return {"mode": "A", "iters": record}


# ============================================================
# CLI
# ============================================================

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["mode_a", "mode_b"])
    ap.add_argument("--orig", required=True, help="original PNG")
    ap.add_argument("--svg", required=True, help="current SVG to repair")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--iters", type=int, default=2)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--w", type=int, default=0, help="render width (0 = from input)")
    ap.add_argument("--h", type=int, default=0)
    args = ap.parse_args()

    if args.w == 0 or args.h == 0:
        img = Image.open(args.orig)
        args.w = args.w or img.width
        args.h = args.h or img.height

    if args.mode == "mode_b":
        mode_b_iter(args.orig, args.svg, args.out, args.model, args.iters,
                    args.w, args.h)
    else:
        mode_a_iter(args.orig, args.svg, args.out, args.model, args.iters,
                    args.w, args.h)
    return 0


if __name__ == "__main__":
    sys.exit(main())
