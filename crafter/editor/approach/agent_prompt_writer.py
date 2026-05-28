"""Agent-driven gpt-image-2 prompt writer.

Instead of a fixed prompt, a vision LLM inspects the figure first,
identifies what visual assets are present, then writes a custom
KEEP/DELETE prompt tailored to this specific figure. The custom prompt
is then sent to gpt-image-2 for extraction.

Pipeline:
  1. Agent looks at original figure → enumerates visible icons/assets
  2. Agent drafts a focused extraction prompt (positive list of what
     this figure has, explicit about what to KEEP and what to DELETE)
  3. The drafted prompt is fed to gpt-image-2
  4. Output: cleaned PNG + the agent's drafted prompt + meta
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
from pathlib import Path

import requests
from PIL import Image, ImageOps

REPO = Path(os.environ.get("CRAFTER_HOME", str(Path(__file__).resolve().parents[2])))

from crafter.editor.approach.iter_svg_fix import call_model  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  agent_pw  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("agent_pw")


# Image-edit credentials are resolved inside the shared image_edit
# provider (crafter.shared.providers.image_edit) when call_gpt_image2()
# instantiates it. Missing env raises there, not at import time, so
# unrelated modules (docs, tests) can import this module freely.


# ---- Stage A: agent drafts the extraction prompt -----------------------

PROMPT_DESIGNER = """You are designing an image-edit instruction for an
academic figure (or poster / infographic). Goal: produce a "cleaned"
version that REMOVES caption-style text, arrows, connector lines,
panel outlines, and flat-coloured panel backgrounds, while KEEPING
every visual asset (icon, photo, chart, illustration, decorative
mark, AND real logos/wordmarks) at exact original size + position,
completely UNCHANGED.

Step 1 — INSPECT the image and ENUMERATE every visible visual asset.
For each, give a short concrete description (e.g. "a small blue
folder icon at top-left", "a lock icon in the centre", "the Snap Inc
wordmark logo at top-right", "the UC Merced shield logo"). This is
the KEEP LIST.

  IMPORTANT — LOGO RULE.  Real logos / wordmarks / brand marks /
  institution crests / conference badges are VISUAL ASSETS, not
  text. KEEP them intact even if they consist of, or contain,
  stylized text — for example: "Snap Inc", "Google", "CVPR 2024",
  university shields, company name in a custom typeface. The text
  inside a real logo IS part of the logo and must survive.

  This is DIFFERENT from a generic text label that happens to sit
  near an icon (e.g. the word "Encoder" written next to a box icon,
  or a bullet caption under a figure panel) — those are caption-style
  text and should be DELETED.

  Heuristic for telling them apart:
    • LOGO  → has its own stylized typography / mark / unique font /
              colour scheme; would be recognised as a brand or
              institution; usually compact and self-contained.
    • LABEL → uses the figure's body font; describes or names a
              nearby element; is part of the figure's annotation
              layer, not a brand asset.
  When in doubt about a text-bearing element, KEEP IT.

Step 2 — ENUMERATE delete categories visible in this figure (e.g.
"horizontal arrows between modules", "section title text in body
font", "bullet captions under panels", "outline boxes around
modules"). This is the DELETE LIST. Do NOT put logos / wordmarks /
brand marks here.

Step 3 — DRAFT the extraction prompt. Write it as a focused
instruction for an image-edit model. Use this template, filled with
specifics from your inspection:

  EDIT THIS IMAGE. Output a copy with the following changes:

  KEEP (do not modify, move, resize, recolour) all of these:
  - <specific list from your inspection, 3-20 items, be specific
    about location and appearance>
  - any real logo / wordmark / brand mark / institution crest /
    conference badge that appears in the figure, even if it contains
    stylized text (the text is part of the logo and must survive)

  DELETE (replace with WHITE pixels) all of these:
  - <specific list of caption-style text labels, arrows, panel
    borders etc.>
  - DO NOT delete logos or wordmarks even if they look text-y.

  Rules:
  - Every kept element stays at EXACT original position and size.
  - Background of deleted regions: pure WHITE.
  - If unsure whether to keep, KEEP IT.

Return STRICT JSON:
{
  "keep_list": ["short description 1", "short description 2", ...],
  "delete_list": ["text labels", "arrows", ...],
  "logos_detected": [
    {
      "desc":"the Snap Inc wordmark",
      "bbox":[x1, y1, x2, y2]    // pixel coords in the ORIGINAL image
    },
    {
      "desc":"the UC Merced shield",
      "bbox":[x1, y1, x2, y2]
    }
  ],                               // empty list [] if no real logos visible
  "extraction_prompt": "the full filled-in prompt above"
}

NOTE on logo bboxes: be conservative — give a TIGHT bounding box that
covers the whole logo (mark + any stylized text) but no surrounding
padding. These bboxes are used as a fallback crop when downstream SAM3
grounding fails to localize a logo, so accuracy matters. If you can't
estimate a tight bbox, omit that logo from logos_detected (do NOT
guess loose coords — better no entry than a wrong one).

Return ONLY this JSON, no preamble.
"""


def _b64_image(path: Path, max_dim: int = 1500) -> str:
    img = Image.open(path).convert("RGB")
    if max(img.size) > max_dim:
        scale = max_dim / max(img.size)
        img = img.resize(
            (int(img.width * scale), int(img.height * scale)),
            Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88)
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


def design_prompt(image_path: Path,
                  model: str = "openai/gpt-5.5") -> dict:
    """Stage A: agent inspects figure + drafts custom extraction prompt."""
    msgs = [{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url":
                {"url": f"data:image/jpeg;base64,{_b64_image(image_path)}"}},
            {"type": "text", "text": PROMPT_DESIGNER},
        ],
    }]
    t0 = time.time()
    try:
        resp = call_model(msgs, model=model, max_tokens=4000,
                          label="stage1a_agent_design")
    except Exception as e:
        return {"error": f"design call failed: {e}",
                "elapsed": time.time() - t0}
    parsed = _extract_json(resp)
    if not parsed:
        return {"error": "agent returned non-JSON",
                "raw": resp[:500],
                "elapsed": time.time() - t0}
    parsed["elapsed"] = round(time.time() - t0, 1)
    return parsed


# ---- Stage B: gpt-image-2 with the agent's drafted prompt --------------

def ceil_to_16(v):
    return ((v + 15) // 16) * 16


def call_gpt_image2(orig_path: Path, prompt: str,
                    out_png: Path, quality: str = "high") -> dict:
    """Edit image via image-edit provider. Returns dict format
    {status, elapsed[, error]}.

    Backward-compat shim: existing callers unchanged. The actual HTTP call
    goes through editor.providers.image_edit.get_default_image_edit() —
    pluggable via set_default_image_edit().
    """
    import sys as _sys
    from pathlib import Path as _Path
    _root = _Path(__file__).resolve().parents[2]
    if str(_root) not in _sys.path:
        _sys.path.insert(0, str(_root))
    from crafter.shared.providers.image_edit import get_default_image_edit  # noqa: E402
    provider = get_default_image_edit()
    resp = provider.edit(orig_path, prompt, out_png, quality=quality)
    out = {"status": resp.status, "elapsed": resp.elapsed_s}
    if resp.error is not None:
        out["error"] = resp.error
    return out


# ---- Stage C: optional verifier — agent checks output --------------------

VERIFIER_PROMPT = """You designed an extraction prompt that was fed to
an image-edit model. Now compare the ORIGINAL (Image #1) and the
CLEANED OUTPUT (Image #2) and report briefly:
  - missing_assets: list of icons/photos that were in #1 but
                     vanished from #2
  - distorted_assets: list of icons/photos that are in #2 but visibly
                       changed (recoloured, reshaped, smaller, etc.)
  - kept_correctly: count of icons that survived intact

Return STRICT JSON only:
{
  "missing_assets": ["...", "..."],
  "distorted_assets": ["...", "..."],
  "kept_correctly": 8,
  "overall_quality": "good" | "ok" | "poor",
  "suggested_prompt_change": "1 sentence of advice for redrafting"
}
"""


def verify(original_path: Path, cleaned_path: Path,
           model: str = "openai/gpt-5.5") -> dict:
    msgs = [{
        "role": "user",
        "content": [
            {"type": "text", "text": "Image #1 — ORIGINAL:"},
            {"type": "image_url", "image_url":
                {"url": f"data:image/jpeg;base64,{_b64_image(original_path)}"}},
            {"type": "text", "text": "Image #2 — CLEANED OUTPUT:"},
            {"type": "image_url", "image_url":
                {"url": f"data:image/jpeg;base64,{_b64_image(cleaned_path)}"}},
            {"type": "text", "text": VERIFIER_PROMPT},
        ],
    }]
    t0 = time.time()
    try:
        resp = call_model(msgs, model=model, max_tokens=2000,
                          label="stage1a_agent_verify")
    except Exception as e:
        return {"error": str(e)}
    parsed = _extract_json(resp) or {}
    parsed["elapsed"] = round(time.time() - t0, 1)
    return parsed


# ---- Comparison image --------------------------------------------------

def make_compare(orig_path: Path, cleaned_path: Path, out_path: Path):
    a = Image.open(orig_path).convert("RGB")
    b = Image.open(cleaned_path).convert("RGB")
    if a.size != b.size:
        b = b.resize(a.size, Image.LANCZOS)
    gap = 8
    canvas = Image.new("RGB", (a.width * 2 + gap, a.height),
                        (200, 200, 200))
    canvas.paste(a, (0, 0))
    canvas.paste(b, (a.width + gap, 0))
    canvas.save(out_path)


# ---- Orchestrator -------------------------------------------------------

def run_one(image_path: Path, out_dir: Path, do_verify: bool = True,
            quality: str = "high"):
    img_name = image_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    # A. Design
    logger.info("[%s] designing custom prompt...", img_name)
    design = design_prompt(image_path)
    (out_dir / f"{img_name}_design.json").write_text(
        json.dumps(design, indent=2, ensure_ascii=False))
    if "extraction_prompt" not in design:
        logger.error("[%s] design failed: %s", img_name,
                     design.get("error", "no prompt"))
        return {"image": img_name, "status": "design_failed"}
    prompt = design["extraction_prompt"]
    logger.info("[%s] design done in %.1fs (keep=%d delete=%d "
                "prompt=%dch)", img_name, design.get("elapsed", 0),
                len(design.get("keep_list", [])),
                len(design.get("delete_list", [])), len(prompt))

    # B. gpt-image-2
    cleaned = out_dir / f"{img_name}_agent_cleaned.png"
    logger.info("[%s] calling gpt-image-2...", img_name)
    extract = call_gpt_image2(image_path, prompt, cleaned, quality=quality)
    (out_dir / f"{img_name}_extract_meta.json").write_text(
        json.dumps(extract, indent=2))
    if extract.get("status") != "ok":
        logger.error("[%s] extract failed: %s", img_name, extract)
        return {"image": img_name, "status": "extract_failed",
                "design": design, "extract": extract}
    logger.info("[%s] extract done in %.1fs", img_name,
                 extract.get("elapsed", 0))

    # Compare image
    compare = out_dir / f"{img_name}_compare.png"
    try:
        make_compare(image_path, cleaned, compare)
    except Exception as e:
        logger.warning("compare failed: %s", e)

    # C. Verify (optional)
    verify_out = None
    if do_verify:
        logger.info("[%s] verifier...", img_name)
        verify_out = verify(image_path, cleaned)
        (out_dir / f"{img_name}_verify.json").write_text(
            json.dumps(verify_out, indent=2, ensure_ascii=False))
        logger.info("[%s] verify: missing=%d distorted=%d kept=%d "
                    "quality=%s",
                    img_name, len(verify_out.get("missing_assets", [])),
                    len(verify_out.get("distorted_assets", [])),
                    verify_out.get("kept_correctly", 0),
                    verify_out.get("overall_quality", "?"))

    return {"image": img_name, "status": "ok",
            "design": design, "extract": extract, "verify": verify_out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", help="single image path")
    ap.add_argument("--examples", action="store_true",
                    help="run on canonical example list")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-verify", action="store_true",
                    help="skip post-extract verifier (saves ~30s/img)")
    ap.add_argument("--quality", default="high",
                    choices=["low", "medium", "high"])
    args = ap.parse_args()

    if not args.image and not args.examples:
        ap.error("Need --image or --examples")

    if args.examples:
        names = ["img1", "img2", "img3", "img4",
                 "img5", "img8", "img10", "img12"]
        images = [REPO / "external_comparison/original" / f"{n}.png"
                  for n in names]
    else:
        images = [Path(args.image)]

    out_dir = Path(args.out)
    summary = []
    for img in images:
        if not img.exists():
            logger.warning("missing %s", img)
            continue
        info = run_one(img, out_dir, do_verify=not args.no_verify,
                        quality=args.quality)
        summary.append(info)
    (out_dir / "summary.json").write_text(json.dumps(
        [{k: v for k, v in s.items() if k not in ("design", "extract", "verify")}
         for s in summary], indent=2))
    logger.info("=== ALL DONE ===  %d/%d ok",
                sum(1 for s in summary if s.get("status") == "ok"),
                len(summary))


if __name__ == "__main__":
    main()
