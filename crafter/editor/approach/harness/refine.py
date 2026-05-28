"""Step 5 — agentic refinement loop.

Each iteration:
  1. Render current SVG → preview PNG.
  2. Call quick_judge for score + fix list.
  3. Stop if score >= STOP_THRESHOLD or iter_count >= MAX_ITER.
  4. Call gpt-5.5 with (original, preview, current SVG, fix list)
     → corrected SVG. Data-URI stash protects long base64.
  5. Safety guard: reject the iter if the new SVG drops <text> by
     > 30% or <image> by > 50% or <rect> by > 50%.
  6. Accept (overwrite current_svg) or reject (keep current_svg).
"""
from __future__ import annotations
import os

import logging
import re
import sys
from pathlib import Path

REPO = Path(os.environ.get("CRAFTER_HOME", str(Path(__file__).resolve().parents[2])))

from crafter.editor.approach.iter_svg_fix import (  # noqa: E402
    encode_image_b64, render_svg, call_model, extract_svg_from_response,
    _stash_data_uris, _restore_data_uris,
)

from .judge import quick_judge  # noqa: E402
from .checkers import text_overflow as _chk_text  # noqa: E402
from .checkers import arrow as _chk_arrow  # noqa: E402
from .checkers import style as _chk_style  # noqa: E402
from .checkers import position as _chk_position  # noqa: E402
from .checkers import overlap as _chk_overlap  # noqa: E402
from .checkers import missing_text as _chk_missing  # noqa: E402
from .checkers import structural as _chk_struct  # noqa: E402
from .checkers import text_size as _chk_text_size  # noqa: E402
from . import prompt_evolver as _evolver  # noqa: E402
from . import raster_cleanup as _raster_cleanup  # noqa: E402

logger = logging.getLogger("harness.refine")


REFINE_PROMPT = """You will receive:
  • Image #1 — the ORIGINAL academic figure to reproduce
  • Image #2 — the CURRENT rendered preview of an SVG attempt
  • a JUDGE FIX LIST (visual feedback) and a CHECKER FIX LIST
    (automated audits — overflow, arrow, style, position, overlap,
    missing text, structural)
  • the CURRENT SVG source text below

Your task: produce a CORRECTED SVG that addresses EVERY checker fix
item AND every judge suggestion, while preserving everything that
already matches.

SOURCE OF TRUTH = Image #1 (visual). Bboxes / coordinates appearing
in the FIX lists are VISUAL ANCHORS produced by noisy automated tools
(OCR, segmentation). Use them to FIND the right region in Image #1,
then place / move / restyle elements to match what you actually see.
NEVER snap an element to a numeric bbox if it visibly disagrees with
Image #1. If a fix sentence says "add at [bbox]", read it as
"add NEAR [bbox], finalised by visual matching".

CRITICAL — DO NOT TOUCH <image> ELEMENTS:
  • Every <image> references a SPECIFIC icon already extracted from
    the original — each `href` value (or `__ICON_REF_NNN__` stash
    marker) belongs to ONE caption. KEEP each existing <image> tag
    verbatim including its href.
  • You MAY move an <image> by changing x/y/width/height, but NEVER
    drop it, NEVER replace it with <rect> or vector shapes.
  • **NEVER duplicate an <image>**: each `href` value (or stash
    marker) must appear AT MOST ONCE in your output SVG. Do NOT clone
    an <image> tag to multiple positions to fake a tile / grid /
    filmstrip layout, even if the original figure shows repeated
    visual elements. If a region needs visually-repeated structure,
    draw it with empty <rect> placeholders or vector primitives —
    never copy the same href to multiple <image> tags.
  • Do NOT introduce NEW <image> tags. Only the <image> tags already
    present in the input SVG are valid; if you need to "add" a panel,
    add a <rect> placeholder instead, never a new <image>.

RASTER-IMAGE OVERLAY RULES — be SMART, not strict:
  Each <image> element renders a real raster icon at its bbox. Avoid
  the visible "double-drawing" bug, but keep legitimate overlays.

  REMOVE these (they cause the "灰底 + duplicate icon" bug):
    • a non-transparent <rect> whose bbox is near-identical (IoU≥0.5)
      to an <image> bbox — that's a redundant background under the icon
    • a <path>/<circle>/<ellipse>/<polygon> that visibly redraws the
      same icon shape underneath — duplicate vector copy of the raster
    • another <image> with the same bbox

  KEEP these (they are legitimate, original-figure overlays):
    • small <text> labels INSIDE or near an image bbox (callouts, tags)
    • small decorative <circle>/<rect> markers on top of a panel image
      (graph nodes, histogram tick marks, status dots) — these are
      authored intentionally on top
    • <line>/<polyline> annotation arrows ENDING at an image bbox
    • a vector primitive whose bbox is < 30% of the image bbox area —
      these are visibly distinct on top, not redundant underlays

  Rule of thumb: if the vector visibly REPRODUCES what the raster icon
  already shows → REMOVE. If the vector ADDS something the raster
  doesn't show (a label, an annotation, a small marker) → KEEP.
  When in doubt, KEEP.

EXPLICIT REPAIR RULES (act on every relevant kind):
  • text_overflow → SHRINK font-size (down to 9px min) or wrap with
    <tspan x="..." dy="1em"> to break into multi-line.
  • text_collision → MOVE one of the colliding texts OR shrink both.
  • text_too_small → INCREASE the font-size of the named text element
    to match the original character height (the fix line gives a
    target font-size). Do NOT change x/y unless the text now overflows
    its bbox; in that case allow a small (≤8px) horizontal nudge.
  • text_too_large → DECREASE the font-size of the named text element
    to match the original. Same — keep position fixed.
  • oversized_arrow → set markerWidth=8 markerHeight=6 max.
  • dangling_arrow → connect each endpoint to the bbox of an actual
    nearby element, OR delete the arrow.
  • style_drift → REMOVE feDropShadow, linearGradient, radialGradient,
    feGaussianBlur, and any filter= attribute on simple boxes.
    Academic figures use FLAT colours.
  • misplaced → MOVE the element x/y to the expected bbox; do NOT
    just restyle in place.
  • overlap → MOVE the smaller element to nearby empty space (offset
    30+ px) so the two no longer cover each other. NEVER delete to
    fix overlap.
  • missing_text → ADD a new <text> with the exact content stated in
    the fix line, anchored at the given bbox. Choose font-size that
    fits the bbox height (height × 0.7 is a good default).
  • missing_element → ADD the missing region as a <rect> placeholder
    (NEVER a new <image> tag — see the no-clone rule above) at the
    given bbox. Match the description in the fix line.
  • structural → restructure as instructed (panel order, flow
    direction). PRESERVE every existing element while restructuring.
  • text_reflow → MOVE the named text so its position relative to the
    referenced neighbour matches the fix instruction.
  • wrong_content → REPLACE the labelled element with the corrected
    content as stated.

POSITION FIRST: if an element is in the wrong place relative to its
neighbours, MOVE IT. Don't try to "fix" it by restyling.

OUTPUT: a single complete SVG document, well-formed XML, starting
with <svg, ending with </svg>. No prose, no markdown.

JUDGE FIX LIST (high-level visual cues):
{JUDGE_FIXES}

CHECKER FIX LIST (automated audits):
{CHECKER_FIXES}

CURRENT SVG:
```svg
{SVG}
```
"""


def _grep_counts(svg: str) -> dict:
    return {
        "text": len(re.findall(r"<text[\s>]", svg)),
        "rect": len(re.findall(r"<rect[\s>]", svg)),
        "image": len(re.findall(r"<image[\s>]", svg)),
        "path": len(re.findall(r"<path[\s>]", svg)),
    }


def _dedupe_image_hrefs(svg: str) -> tuple[str, int]:
    """Defensive: drop any <image> tags whose href value has already
    appeared. The refine LLM occasionally clones a single href across
    many positions to fake a tile / filmstrip layout, producing a
    cluttered output (root cause traced from img18 — score collapse
    from one crop of "Lossless Decompression" being placed 25× across
    the architecture region). Caller logs the count.
    """
    img_pat = re.compile(
        r'<image\b[^>]*?(?:href|xlink:href)\s*=\s*"([^"]+)"[^>]*?/>',
        re.IGNORECASE,
    )
    seen: set[str] = set()
    n_dropped = 0

    def repl(m):
        nonlocal n_dropped
        href = m.group(1)
        if href in seen:
            n_dropped += 1
            return ""
        seen.add(href)
        return m.group(0)

    new_svg = img_pat.sub(repl, svg)
    return new_svg, n_dropped


def _safety_guards_pass(old_counts: dict, new_counts: dict) -> tuple[bool, str]:
    if old_counts["image"] > 0 and new_counts["image"] < old_counts["image"] * 0.5:
        return False, (f"image dropped {old_counts['image']}→{new_counts['image']} "
                       f"(>50%) — reject")
    if old_counts["text"] > 0 and new_counts["text"] < old_counts["text"] * 0.7:
        return False, (f"text dropped {old_counts['text']}→{new_counts['text']} "
                       f"(>30%) — reject")
    if old_counts["rect"] > 0 and new_counts["rect"] < old_counts["rect"] * 0.5:
        return False, (f"rect dropped {old_counts['rect']}→{new_counts['rect']} "
                       f"(>50%) — reject")
    return True, "ok"


def _format_checker_fixes(fix_list: list[dict]) -> str:
    if not fix_list:
        return "  (no checker issues)"
    lines = []
    for f in fix_list:
        kind = f.get("kind", "?")
        lines.append(f"  - [{kind}] {f.get('fix','')}")
    return "\n".join(lines)


def _run_checkers(svg_path: Path, preview_png: Path,
                  labels_json: Path | None = None,
                  original_png: Path | None = None,
                  paddle_texts: list[dict] | None = None,
                  labels: dict | None = None,
                  iter_idx: int = 0,
                  cache_dir: Path | None = None,
                  enable_structural: bool = True,
                  model: str = "openai/gpt-5.5") -> dict:
    """Run all checkers; return dict of fix lists keyed by category."""
    out = {"text": [], "arrow": [], "style": [], "position": [],
           "overlap": [], "missing": [], "structural": []}
    try:
        out["text"] = _chk_text.check(svg_path, preview_png)
    except Exception as e:
        logger.warning("  text checker failed: %s", e)
    try:
        out["arrow"] = _chk_arrow.check(svg_path)
    except Exception as e:
        logger.warning("  arrow checker failed: %s", e)
    try:
        out["style"] = _chk_style.check(svg_path)
    except Exception as e:
        logger.warning("  style checker failed: %s", e)
    if labels_json and labels_json.exists():
        try:
            out["position"] = _chk_position.check(svg_path, labels_json)
        except Exception as e:
            logger.warning("  position checker failed: %s", e)
    try:
        out["overlap"] = _chk_overlap.check(svg_path, labels_json)
    except Exception as e:
        logger.warning("  overlap checker failed: %s", e)
    if paddle_texts:
        cache = (cache_dir / f"missing_iter{iter_idx}.json"
                 if cache_dir else None)
        try:
            out["missing"] = _chk_missing.check(
                preview_png, paddle_texts, cache_path=cache)
        except Exception as e:
            logger.warning("  missing_text checker failed: %s", e)
        # text_size_drift critic — flag <text> rendering at the wrong
        # font-size relative to the character height in the original.
        # Reuses paddle output on the preview (cached above).
        ts_cache = (cache_dir / f"text_size_iter{iter_idx}.json"
                    if cache_dir else None)
        try:
            out["text_size"] = _chk_text_size.check(
                preview_png, paddle_texts, cache_path=ts_cache)
        except Exception as e:
            logger.warning("  text_size checker failed: %s", e)
    if enable_structural and original_png is not None:
        cache = (cache_dir / f"structural_iter{iter_idx}.json"
                 if cache_dir else None)
        try:
            out["structural"] = _chk_struct.check(
                original_png, preview_png,
                paddle_texts=paddle_texts, labels=labels,
                model=model, cache_path=cache)
        except Exception as e:
            logger.warning("  structural critic failed: %s", e)
    return out


def _should_stop(history: list[dict], max_iter: int) -> tuple[bool, str]:
    """Convergence detector — Item 6.

    Always run iter 1 (first refine usually big).
    Stop if last 2 iters' total Δ < 0.3 (catches stalls but allows
      a single-iter dip-then-rise like img3 iter 2→3).
    Hard cap at max_iter.
    """
    completed = len([h for h in history if h.get("iter", 0) > 0])
    if completed >= max_iter:
        return True, f"hit hard cap max_iter={max_iter}"
    if completed < 2:
        return False, "need at least 2 iters before considering stop"
    if completed >= 3:
        recent_gain = history[-1]["score"] - history[-3]["score"]
        if recent_gain < 0.3:
            return True, (f"converged: last 2 iters total Δ={recent_gain:+.2f} "
                          f"< 0.3")
    return False, "still improving"


def refine_loop(
    original_png: Path,
    initial_svg_path: Path,
    out_dir: Path,
    w: int, h: int,
    model: str = "openai/gpt-5.5",
    max_iter: int = 4,
    stop_threshold: float = 8.5,
    labels_json: Path | None = None,
    paddle_texts: list[dict] | None = None,
    enable_structural_critic: bool = True,
    # Regression guard: fall back from last-accepted to best-so-far
    # when last_score < best_score - this. 0.3 sits just above the
    # judge noise floor while still catching single-iter regressions.
    bsf_regression_guard: float = 0.3,
    enable_self_evolving_prompt: bool = True,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    current_svg = initial_svg_path.read_text(encoding="utf-8")
    history = []
    # current_* = state of the most recent accepted iter (loop input).
    # best_*    = best-by-score across all judged iters (loop output).
    # Splitting these lets the loop keep refining on top of the latest
    # SVG even after a regression, while shipping the high-water mark.
    current_svg_path = initial_svg_path
    current_png_path = initial_svg_path.with_suffix(".png").with_name(
        initial_svg_path.stem + "_preview.png")

    # Load labels.json once for the structural critic / overlap checker
    labels_data = None
    if labels_json and labels_json.exists():
        try:
            import json as _json
            labels_data = _json.loads(labels_json.read_text())
        except Exception:
            labels_data = None

    # Cache dir for missing/structural per-iter outputs
    chk_cache_dir = out_dir / "checker_cache"
    chk_cache_dir.mkdir(parents=True, exist_ok=True)

    # Initial render + judge + checkers
    if not current_png_path.exists():
        render_svg(str(initial_svg_path), str(current_png_path), w, h)
    j0 = quick_judge(original_png, current_png_path, model=model)
    chk0 = _run_checkers(initial_svg_path, current_png_path, labels_json,
                         original_png=original_png,
                         paddle_texts=paddle_texts, labels=labels_data,
                         iter_idx=0, cache_dir=chk_cache_dir,
                         enable_structural=enable_structural_critic,
                         model=model)
    n_chk0 = sum(len(v) for v in chk0.values())
    history.append({"iter": 0, "score": j0.get("overall", 0.0),
                    "fixes": j0.get("fixes", []), "scores": j0,
                    "checker_issues": n_chk0, "checkers": chk0,
                    "svg": str(current_svg_path),
                    "preview": str(current_png_path),
                    "status": "initial"})
    logger.info("  iter 0: score=%.2f, checker_issues=%d "
                "(txt=%d arr=%d sty=%d pos=%d ovr=%d mis=%d str=%d)",
                history[-1]["score"], n_chk0,
                len(chk0["text"]), len(chk0["arrow"]),
                len(chk0["style"]), len(chk0["position"]),
                len(chk0["overlap"]), len(chk0["missing"]),
                len(chk0["structural"]))

    # Best-so-far tracker (initialised to iter 0).
    best_iter = 0
    best_score = j0.get("overall", 0.0)
    best_chk = n_chk0
    best_svg_path = current_svg_path
    best_png_path = current_png_path

    # Self-evolving prompt — per-case trajectory + lessons accumulator.
    evolve_trajectory: list[dict] = []
    evolve_lessons: list[str] = []
    # Seed iter 0 in trajectory (no fixes attempted yet, just baseline).
    evolve_trajectory.append({
        "iter": 0, "status": "initial",
        "score_before": None, "score_after": j0.get("overall", 0.0),
        "checker_changes": {}, "judge_fixes_addressed": [],
        "notable": f"baseline_score={j0.get('overall',0.0):.2f} chk={n_chk0}",
    })

    # C (prompt slim): track (kind, fix) tuples from prev iter's checker
    # report. Persistent issues (same tuple seen last iter) get collapsed
    # to a one-line marker in the prompt to save ~5–10% of input tokens.
    prev_chk_fix_set: set[tuple[str, str]] = set()

    for i in range(1, max_iter + 1):
        if history[-1]["score"] >= stop_threshold and \
                history[-1]["checker_issues"] == 0:
            logger.info("  stopping at iter %d (score=%.2f >= %.2f, "
                        "no checker issues)",
                        i - 1, history[-1]["score"], stop_threshold)
            break
        stop, why = _should_stop(history, max_iter)
        if stop:
            logger.info("  early-stop at iter %d: %s", i - 1, why)
            break

        # Aggregate checker fixes from the LAST ACCEPTED iter (rejected
        # iters don't carry checker output — find the last one that does).
        last_chk_state = next(
            (h for h in reversed(history) if "checkers" in h),
            None,
        )
        if last_chk_state is None:
            all_chk = []
        else:
            chks = last_chk_state["checkers"]
            all_chk = (chks.get("text", []) + chks.get("arrow", [])
                       + chks.get("style", []) + chks.get("position", [])
                       + chks.get("overlap", []) + chks.get("missing", [])
                       + chks.get("text_size", [])
                       + chks.get("structural", []))

        # Self-evolving prompt: reflect on trajectory before iter ≥ 2.
        lessons_block = ""
        if enable_self_evolving_prompt and len(evolve_trajectory) >= 2:
            try:
                lesson = _evolver.evolve(evolve_trajectory, model=model)
                if lesson:
                    evolve_lessons.append({"iter": i, "lesson": lesson})
                    lessons_block = _evolver.format_for_refine(lesson)
                    if lessons_block:
                        logger.info("  iter %d lessons: %s", i, lesson[:140])
            except Exception as e:
                logger.warning("  prompt_evolver iter %d failed: %s", i, e)

        # Build refine prompt
        compact_svg, stash = _stash_data_uris(current_svg)
        # Preview + judge fixes come from the most recent ACCEPTED /
        # initial iter — rejected/api_error/no_svg/malformed entries
        # don't carry preview/fixes forward.
        last_with_preview = next(
            (h for h in reversed(history) if "preview" in h), None)
        prev_png = Path(last_with_preview["preview"]
                        if last_with_preview else current_png_path)
        last_fixes = (last_with_preview.get("fixes", [])
                      if last_with_preview else [])
        # C (prompt slim): split checker fixes into NEW (first-seen this
        # iter) and PERSISTENT (also present in prev iter). NEW go into
        # the body in full; PERSISTENT collapse to one summary line. The
        # LLM is told to "continue addressing them" so signal is preserved.
        cur_chk_keys = [(f.get("kind", "?"), f.get("fix", "")) for f in all_chk]
        new_chk_items = [f for f, k in zip(all_chk, cur_chk_keys)
                          if k not in prev_chk_fix_set]
        n_persistent = len(all_chk) - len(new_chk_items)
        checker_block = _format_checker_fixes(new_chk_items)
        if n_persistent > 0:
            checker_block += (
                f"\n  + {n_persistent} persistent issue(s) carried over "
                f"from the previous iter — keep addressing them."
            )
        prev_chk_fix_set = set(cur_chk_keys)

        # Use plain replace — SVG and texts may contain `{...}` which
        # str.format() would mistake for placeholders.
        prompt = lessons_block + (REFINE_PROMPT
                  .replace("{JUDGE_FIXES}",
                           "\n".join(f"  - {f}" for f in last_fixes))
                  .replace("{CHECKER_FIXES}", checker_block)
                  .replace("{SVG}", compact_svg))
        msgs = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "Image #1 — ORIGINAL:"},
                {"type": "image_url", "image_url":
                    {"url": f"data:image/jpeg;base64,{encode_image_b64(str(original_png))}"}},
                {"type": "text", "text": "Image #2 — CURRENT preview:"},
                {"type": "image_url", "image_url":
                    {"url": f"data:image/jpeg;base64,{encode_image_b64(str(prev_png))}"}},
                {"type": "text", "text": prompt},
            ],
        }]

        try:
            resp = call_model(msgs, model=model, max_tokens=64000,
                              label=f"stage6_refine_iter{i}")
        except Exception as e:
            logger.warning("  iter %d refine api failed: %s", i, e)
            history.append({"iter": i, "status": "api_error",
                            "score": history[-1]["score"]})
            break

        # Carry forward state from last accepted iter so the loop can
        # keep running even when an iter is rejected.
        prev_score = history[-1].get("score", 0.0)
        prev_chk_issues = history[-1].get("checker_issues", 0)

        new_svg = extract_svg_from_response(resp)
        if new_svg and stash:
            new_svg = _restore_data_uris(new_svg, stash)
        if new_svg:
            new_svg, n_dups = _dedupe_image_hrefs(new_svg)
            if n_dups:
                logger.info("  iter %d: removed %d duplicated <image> "
                            "tags (LLM cloned hrefs)", i, n_dups)
        if not new_svg:
            logger.warning("  iter %d: no SVG extracted", i)
            history.append({"iter": i, "status": "no_svg",
                            "score": prev_score,
                            "checker_issues": prev_chk_issues})
            continue

        # safety guards
        old_c, new_c = _grep_counts(current_svg), _grep_counts(new_svg)
        ok, why = _safety_guards_pass(old_c, new_c)
        if not ok:
            logger.warning("  iter %d REJECTED: %s", i, why)
            history.append({"iter": i, "status": f"rejected:{why}",
                            "before": old_c, "after": new_c,
                            "score": prev_score,
                            "checker_issues": prev_chk_issues})
            continue

        # Validate XML
        import xml.etree.ElementTree as ET
        try:
            ET.fromstring(new_svg)
        except ET.ParseError as e:
            logger.warning("  iter %d malformed XML: %s", i, e)
            (out_dir / f"iter_{i}_bad.svg").write_text(new_svg)
            history.append({"iter": i, "status": "malformed_xml",
                            "score": prev_score,
                            "checker_issues": prev_chk_issues})
            continue

        # NOTE: raster cleanup disabled by default (see compose.py
        # comment) — empirically over-aggressive when underlying
        # raster icons have quality issues. The REFINE_PROMPT now
        # explicitly forbids drawing vector primitives over raster
        # bboxes, which prevents NEW double-drawing without removing
        # potentially-helpful "compensating" vectors from older runs.

        # Accept
        sp = out_dir / f"iter_{i}.svg"
        pp = out_dir / f"iter_{i}_preview.png"
        sp.write_text(new_svg, encoding="utf-8")
        render_svg(str(sp), str(pp), w, h)
        ji = quick_judge(original_png, pp, model=model)
        chk_i = _run_checkers(sp, pp, labels_json,
                              original_png=original_png,
                              paddle_texts=paddle_texts,
                              labels=labels_data,
                              iter_idx=i, cache_dir=chk_cache_dir,
                              enable_structural=enable_structural_critic,
                              model=model)
        n_chk_i = sum(len(v) for v in chk_i.values())
        current_svg = new_svg
        current_svg_path = sp
        current_png_path = pp
        new_score = ji.get("overall", 0.0)

        # Best-so-far update: max score, tie-break = fewer checker issues.
        is_best = (new_score > best_score) or (
            new_score == best_score and n_chk_i < best_chk)
        if is_best:
            best_iter = i
            best_score = new_score
            best_chk = n_chk_i
            best_svg_path = sp
            best_png_path = pp

        history.append({
            "iter": i, "status": "accepted",
            "score": new_score,
            "fixes": ji.get("fixes", []), "scores": ji,
            "checker_issues": n_chk_i, "checkers": chk_i,
            "svg": str(sp), "preview": str(pp),
            "before": old_c, "after": new_c,
        })

        # Record into the evolving-prompt trajectory.
        chk_changes = {}
        if last_chk_state is not None:
            old_chks = last_chk_state.get("checkers", {})
            for kind in ("text", "arrow", "style", "position",
                         "overlap", "missing", "structural"):
                old_n = len(old_chks.get(kind, []))
                new_n = len(chk_i.get(kind, []))
                chk_changes[kind] = new_n - old_n
        evolve_trajectory.append({
            "iter": i, "status": "accepted",
            "score_before": history[-2]["score"],
            "score_after": new_score,
            "checker_changes": chk_changes,
            # crude: mark each judge fix as "addressed" if score went up
            "judge_fixes_addressed": [new_score > history[-2]["score"]] *
                                     len(history[-2].get("fixes", [])),
            "notable": (f"counts: text {old_c['text']}→{new_c['text']}, "
                        f"image {old_c['image']}→{new_c['image']}, "
                        f"rect {old_c['rect']}→{new_c['rect']}"),
        })
        logger.info("  iter %d: score %.2f → %.2f  text=%d→%d image=%d→%d "
                    "rect=%d→%d  checker_issues %d→%d "
                    "(txt=%d arr=%d sty=%d pos=%d ovr=%d mis=%d str=%d)%s",
                    i, history[-2]["score"], new_score,
                    old_c["text"], new_c["text"],
                    old_c["image"], new_c["image"],
                    old_c["rect"], new_c["rect"],
                    history[-2]["checker_issues"], n_chk_i,
                    len(chk_i["text"]), len(chk_i["arrow"]),
                    len(chk_i["style"]), len(chk_i["position"]),
                    len(chk_i["overlap"]), len(chk_i["missing"]),
                    len(chk_i["structural"]),
                    "  ★ NEW BEST" if is_best else "")

    last_iter_num = history[-1].get("iter", 0)
    last_score = history[-1].get("score", 0.0)
    last_chk = history[-1].get("checker_issues", 0)
    regression = best_score - last_score

    if regression >= bsf_regression_guard:
        # Pathological dip — last iter is significantly worse than the
        # historical high. Fall back to the best-so-far snapshot.
        final_svg_path = best_svg_path
        final_png_path = best_png_path
        final_score = best_score
        final_chk = best_chk
        selection = "bsf_fallback"
    else:
        # Normal case. The last accepted iter is within noise of the
        # high-water mark, so prefer it (same behaviour as pre-bsf v3a).
        final_svg_path = current_svg_path
        final_png_path = current_png_path
        final_score = last_score
        final_chk = last_chk
        selection = "last_accepted"

    logger.info("  best iter %d (score=%.2f, chk=%d) — last iter %d "
                "(score=%.2f, chk=%d) — regression=%.2f → selection=%s",
                best_iter, best_score, best_chk,
                last_iter_num, last_score, last_chk,
                regression, selection)

    return {
        "history": history,
        "final_svg": str(final_svg_path),
        "final_preview": str(final_png_path),
        "final_score": final_score,
        "final_checker_issues": final_chk,
        "selection": selection,
        "regression": regression,
        "regression_guard": bsf_regression_guard,
        "best_iter": best_iter,
        "best_score": best_score,
        "best_checker_issues": best_chk,
        "last_iter": last_iter_num,
        "last_iter_score": last_score,
        "last_iter_checker_issues": last_chk,
        "evolve_trajectory": evolve_trajectory,
        "evolve_lessons": evolve_lessons,
    }
