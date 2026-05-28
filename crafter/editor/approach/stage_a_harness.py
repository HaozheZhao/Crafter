"""Stage A harness — iterate cleaning prompt to maximize asset preservation.

Loop:
  iter 1:
    prompt = design_prompt(orig, model=ITER_MODEL)
    cleaned = gpt-image-2(orig, prompt)
    verdict = verify(orig, cleaned, model=ITER_MODEL)
  iter 2..N (only if verdict.overall_quality != "good"
              AND verdict.missing_assets/distorted_assets non-empty):
    prompt = revise_prompt(orig, prev_prompt, verdict, model=ITER_MODEL)
    cleaned = gpt-image-2(orig, prompt)
    verdict = verify(...)
  break when quality == "good" OR iter == MAX_ITERS

Output (drop-in replacement for stage1_cleaned single-shot):
  out_dir/icons_only/cleaned.png         — final cleaned image
  out_dir/agent_extraction_prompt.txt    — final iter's extraction prompt
  out_dir/agent_design.json              — final iter's design dict
                                            (Fix A's logo-bbox bypass reads this)
  out_dir/stage_a_harness_history.json   — per-iter prompt/verdict/elapsed/cost

The ITER steps (design/verify/revise) use ITER_MODEL (default gemini-3.1-pro
-preview, cheap). The actual gpt-image-2 generation is independent of the
prompt-writer model.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

REPO = Path(os.environ.get("CRAFTER_HOME", str(Path(__file__).resolve().parents[2])))
sys.path.insert(0, str(REPO))

from crafter.editor.approach.agent_prompt_writer import (
    PROMPT_DESIGNER, VERIFIER_PROMPT,
    _b64_image, _extract_json, call_gpt_image2,
)
from crafter.editor.approach.iter_svg_fix import call_model

logger = logging.getLogger("stage_a_harness")


# ---- Revise prompt -----------------------------------------------------

REVISE_PROMPT = """You previously designed an image-edit instruction to
clean an academic figure (or poster / infographic). The cleaned output
was reviewed and the following problems were found:

  missing_assets:    {missing}
  distorted_assets:  {distorted}
  kept_correctly:    {kept_correctly}
  overall_quality:   {quality}
  verifier_advice:   {advice}

Your previous KEEP list was:
{prev_keep}

Your previous DELETE list was:
{prev_delete}

Your previous extraction_prompt was:
---
{prev_prompt}
---

REVISE the design to FIX the missing/distorted assets while still
deleting the originally targeted text/arrows/borders. Specifically:

1. Add any missed assets to the KEEP list with MORE EXPLICIT
   descriptions (location, colour, shape).
2. If a logo was deleted, re-emphasise the LOGO RULE — real
   logos/wordmarks/brand marks must be KEPT even if they look text-y.
3. Tighten language around DELETE so a borderline element is no
   longer mistaken for a target.
4. Re-write the full extraction_prompt to incorporate these fixes.

Use the exact same template structure (KEEP / DELETE / Rules
sections) as before.

Return STRICT JSON, same schema as before:
{{
  "keep_list": [...],
  "delete_list": [...],
  "logos_detected": [{{"desc":"...", "bbox":[x1,y1,x2,y2]}}, ...],
  "extraction_prompt": "the revised full prompt"
}}

Return ONLY this JSON, no preamble.
"""


def design_prompt_iter(image_path: Path, model: str, label: str) -> dict:
    """Design phase, with explicit label for instrumentation."""
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
        resp = call_model(msgs, model=model, max_tokens=4000, label=label)
    except Exception as e:
        return {"error": f"design call failed: {e}",
                "elapsed": round(time.time() - t0, 1)}
    parsed = _extract_json(resp)
    if not parsed:
        return {"error": "agent returned non-JSON",
                "raw": resp[:500],
                "elapsed": round(time.time() - t0, 1)}
    parsed["elapsed"] = round(time.time() - t0, 1)
    return parsed


def verify_iter(orig_path: Path, cleaned_path: Path,
                model: str, label: str) -> dict:
    """Verify phase, with explicit label for instrumentation."""
    msgs = [{
        "role": "user",
        "content": [
            {"type": "text", "text": "Image #1 — ORIGINAL:"},
            {"type": "image_url", "image_url":
                {"url": f"data:image/jpeg;base64,{_b64_image(orig_path)}"}},
            {"type": "text", "text": "Image #2 — CLEANED OUTPUT:"},
            {"type": "image_url", "image_url":
                {"url": f"data:image/jpeg;base64,{_b64_image(cleaned_path)}"}},
            {"type": "text", "text": VERIFIER_PROMPT},
        ],
    }]
    t0 = time.time()
    try:
        resp = call_model(msgs, model=model, max_tokens=2000, label=label)
    except Exception as e:
        return {"error": str(e),
                "elapsed": round(time.time() - t0, 1)}
    parsed = _extract_json(resp) or {}
    parsed["elapsed"] = round(time.time() - t0, 1)
    return parsed



def revise_prompt(image_path: Path, prev_design: dict, verdict: dict,
                  model: str, label: str) -> dict:
    """Revise phase: produce a new design dict using verdict."""
    fmt = REVISE_PROMPT.format(
        missing=json.dumps(verdict.get("missing_assets", []),
                            ensure_ascii=False),
        distorted=json.dumps(verdict.get("distorted_assets", []),
                              ensure_ascii=False),
        kept_correctly=verdict.get("kept_correctly", "?"),
        quality=verdict.get("overall_quality", "?"),
        advice=verdict.get("suggested_prompt_change", ""),
        prev_keep=json.dumps(prev_design.get("keep_list", []),
                              ensure_ascii=False, indent=2),
        prev_delete=json.dumps(prev_design.get("delete_list", []),
                                ensure_ascii=False, indent=2),
        prev_prompt=prev_design.get("extraction_prompt", "")[:3000],
    )
    msgs = [{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url":
                {"url": f"data:image/jpeg;base64,{_b64_image(image_path)}"}},
            {"type": "text", "text": fmt},
        ],
    }]
    t0 = time.time()
    try:
        resp = call_model(msgs, model=model, max_tokens=4000, label=label)
    except Exception as e:
        return {"error": f"revise call failed: {e}",
                "elapsed": round(time.time() - t0, 1)}
    parsed = _extract_json(resp)
    if not parsed:
        return {"error": "revise returned non-JSON",
                "raw": resp[:500],
                "elapsed": round(time.time() - t0, 1)}
    parsed["elapsed"] = round(time.time() - t0, 1)
    return parsed


# ---- Main orchestrator -------------------------------------------------

def _should_continue(verdict: dict) -> bool:
    """Decide whether to iterate again."""
    if verdict.get("error"):
        return False  # verifier failed — stop, ship current
    quality = verdict.get("overall_quality", "?")
    if quality == "good":
        return False
    missing = verdict.get("missing_assets", []) or []
    distorted = verdict.get("distorted_assets", []) or []
    if not missing and not distorted:
        return False  # nothing to fix
    return True


def run(img: str, orig_path: Path, out_dir: Path,
        iter_model: str, max_iters: int) -> dict:
    """Run iterative Stage A. Writes cleaned.png + agent_design.json
    + stage_a_harness_history.json in out_dir.

    Returns: dict with summary (n_iters, final_quality, total_elapsed, etc.)
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    icons_dir = out_dir / "icons_only"
    icons_dir.mkdir(parents=True, exist_ok=True)
    cleaned_final = icons_dir / "cleaned.png"
    history_path = out_dir / "stage_a_harness_history.json"

    history: list = []
    t_total = time.time()

    design = None
    verdict = None
    quality = "?"
    last_ok_design = None  # last design that produced a successful iter

    for i in range(1, max_iters + 1):
        iter_t0 = time.time()
        iter_rec: dict = {"iter": i}

        # 1. Design (or revise)
        if i == 1:
            logger.info("[%s] stage_a iter %d: design (model=%s) ...",
                        img, i, iter_model)
            design = design_prompt_iter(orig_path, iter_model,
                                         label=f"stage_a_iter{i}_design")
        else:
            logger.info("[%s] stage_a iter %d: revise (model=%s) ...",
                        img, i, iter_model)
            design = revise_prompt(orig_path, design, verdict,
                                    iter_model,
                                    label=f"stage_a_iter{i}_revise")
        iter_rec["design"] = {
            "elapsed": design.get("elapsed", 0),
            "n_keep": len(design.get("keep_list", [])),
            "n_delete": len(design.get("delete_list", [])),
            "n_logos": len(design.get("logos_detected", [])),
            "prompt_len": len(design.get("extraction_prompt", "")),
            "error": design.get("error"),
        }
        if "_diff_meta" in design:
            iter_rec["design"]["diff_meta"] = design["_diff_meta"]
        if "extraction_prompt" not in design:
            iter_rec["status"] = "design_failed"
            history.append(iter_rec)
            history_path.write_text(json.dumps(history, indent=2,
                                                ensure_ascii=False))
            logger.error("[%s] iter %d design failed: %s",
                          img, i, design.get("error"))
            # If iter 1 failed, no cleaned image yet — re-raise.
            if i == 1:
                raise RuntimeError(f"[{img}] stage_a iter 1 design "
                                   f"failed: {design.get('error')}")
            # Else keep prior cleaned + design; stop loop.
            break

        # Save full design json for this iter (audit)
        (out_dir / f"stage_a_iter{i}_design.json").write_text(
            json.dumps(design, indent=2, ensure_ascii=False))

        # 2. gpt-image-2 (always uses default model — independent of iter_model)
        cleaned_iter = icons_dir / f"cleaned_iter{i}.png"
        logger.info("[%s] stage_a iter %d: gpt-image-2 ...", img, i)
        extract = call_gpt_image2(orig_path, design["extraction_prompt"],
                                   cleaned_iter, quality="high")
        iter_rec["extract"] = extract
        if extract.get("status") != "ok":
            logger.error("[%s] iter %d extract failed: %s", img, i, extract)
            iter_rec["status"] = "extract_failed"
            history.append(iter_rec)
            history_path.write_text(json.dumps(history, indent=2,
                                                ensure_ascii=False))
            if i == 1:
                raise RuntimeError(f"[{img}] stage_a iter 1 gpt-image-2 "
                                   f"failed: {extract}")
            break

        # 3. Verify
        logger.info("[%s] stage_a iter %d: verify (model=%s) ...",
                    img, i, iter_model)
        verdict = verify_iter(orig_path, cleaned_iter, iter_model,
                               label=f"stage_a_iter{i}_verify")
        iter_rec["verify"] = verdict
        quality = verdict.get("overall_quality", "?")
        n_miss = len(verdict.get("missing_assets", []) or [])
        n_dist = len(verdict.get("distorted_assets", []) or [])
        n_kept = verdict.get("kept_correctly", "?")
        iter_rec["iter_elapsed"] = round(time.time() - iter_t0, 1)
        iter_rec["status"] = "ok"
        last_ok_design = design  # remember the design that produced this OK iter
        history.append(iter_rec)
        history_path.write_text(json.dumps(history, indent=2,
                                            ensure_ascii=False))

        logger.info("[%s] iter %d done %.1fs: quality=%s missing=%d "
                    "distorted=%d kept=%s",
                    img, i, iter_rec["iter_elapsed"], quality, n_miss,
                    n_dist, n_kept)

        # 4. Decide continue
        if not _should_continue(verdict):
            logger.info("[%s] stage_a converged at iter %d (quality=%s)",
                        img, i, quality)
            break

    # ---- FINALIZE ----
    # Find the best iter to use as the FINAL cleaned.png. Use the LAST
    # successful iter (we treat revisions as monotonic improvements; if
    # they're not, that's a model failure, not our problem here).
    last_ok = None
    for rec in history:
        if rec.get("status") == "ok":
            last_ok = rec["iter"]
    if last_ok is None:
        raise RuntimeError(f"[{img}] no successful stage_a iter")

    src_cleaned = icons_dir / f"cleaned_iter{last_ok}.png"
    if not src_cleaned.exists():
        raise RuntimeError(f"[{img}] cleaned_iter{last_ok}.png missing")
    # Atomic copy to canonical location
    cleaned_final.write_bytes(src_cleaned.read_bytes())

    # Save final design for downstream (Fix A reads agent_design.json).
    # Use the last design that produced a successful iter — `design` may be
    # a failed-revise dict (no extraction_prompt) if the final iter's revise
    # call returned non-JSON.
    final_design = last_ok_design if last_ok_design is not None else design
    (out_dir / "agent_design.json").write_text(
        json.dumps(final_design, indent=2, ensure_ascii=False))
    if "extraction_prompt" in final_design:
        (out_dir / "agent_extraction_prompt.txt").write_text(
            final_design["extraction_prompt"])

    total = time.time() - t_total
    summary = {
        "img": img,
        "n_iters": len(history),
        "n_iters_ok": sum(1 for r in history if r.get("status") == "ok"),
        "final_iter_used": last_ok,
        "final_quality": quality,
        "iter_model": iter_model,
        "max_iters": max_iters,
        "total_elapsed_s": round(total, 1),
        "history": history,
    }
    # Append summary back into history file
    history_path.write_text(json.dumps({
        "summary": {k: v for k, v in summary.items() if k != "history"},
        "history": history,
    }, indent=2, ensure_ascii=False))

    logger.info("[%s] === stage_a harness done === iters=%d quality=%s "
                "total=%.1fs cleaned=%s",
                img, len(history), quality, total, cleaned_final)
    return summary
