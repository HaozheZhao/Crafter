"""Composition phase — iterative SVG assembly with hybrid critic.

Paper §3.3.2 (\\Editor: Composition).

Roles:
  D (designer)  — language model drafts two SVG skeletons at
                  temperatures 0.20 and 0.45; the convergence judge
                  picks the better.
  E (executor)  — splices extracted assets into the placeholders of
                  the selected skeleton (raster <image href="data:..."/>
                  + vector primitives for vector-codeable elements).
  V (verifier)  — hybrid critic: (1) VLM quick-judge for global layout
                  fidelity + semantic correspondence; (2) programmatic
                  checkers (text overflow, arrow endpoints, text size
                  drift, overlap, missing-text) for structural audits.
  R (reviser)   — language model rewrites the SVG given V's diagnostic.

Loop runs for at most T=4 refinement rounds, with best-so-far
reversion guarding against non-monotonic regressions.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from shutil import copy

from PIL import Image as PILImage

from .config import Config

logger = logging.getLogger("editor_v2.composition")

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "crafter" / "editor" / "approach"))


@dataclass
class CompositionResult:
    img_id: str
    out_dir: Path
    final_svg: Path
    final_png: Path
    skeleton_picked_score: float
    composed_score: float
    refine_history: list
    refine_final_score: float


def run(img_id: str, original_png: Path, processing_pairs_json: Path,
        out_root: Path, config: Config) -> CompositionResult:
    """Run the composition harness on one figure.

    Implements D→E→V→R loop + visual polish using production modules
    from `crafter/editor/approach/harness/`.
    """
    from harness import build_marked, compose, refine
    from harness.judge import quick_judge
    from harness.checkers import paddle_text as _paddle
    from harness.checkers import text_recovery as _text_recovery
    from crafter.editor.approach import style_analyzer

    out_dir = out_root / img_id / "compose"
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = json.loads(processing_pairs_json.read_text())
    crops_dir = processing_pairs_json.parent / "crops"
    W, H = PILImage.open(original_png).size
    t0 = time.time()

    cc = config.composition
    logger.info("[composition] %s ▸ %dx%d  designer=%s reviser=%s "
                "skeleton_T=%s refine_T=%d",
                img_id, W, H,
                cc.designer_model, cc.reviser_model,
                cc.skeleton_temperatures, cc.refine_max_iters)

    # ---- Setup: build placeholder-marked image + OCR + style profile ----
    labels = build_marked.build(original_png, pairs, out_dir)
    logger.info("[composition]   placeholders: %d raster + %d vector",
                len(labels["raster"]), len(labels["vector"]))

    paddle_texts = _paddle.extract(original_png,
                                     out_dir / "paddle_texts.json")
    rec = _text_recovery.recover(
        original_png, paddle_texts,
        cache_path=out_dir / "text_recovery.json",
        model=cc.designer_model,
    )
    paddle_texts = rec["corrected"]

    style_profile = pairs.get("style_profile") if isinstance(
        pairs, dict) else None
    if not style_profile or not isinstance(
            style_profile.get("font_hints"), dict):
        try:
            style_profile = style_analyzer.analyze(
                original_png,
                cache_path=out_dir / "style_profile.json",
                model=cc.designer_model,
            )
        except Exception as e:
            logger.warning("[composition]   style_analyzer failed: %s", e)
            style_profile = None

    # ---- D role: sequential best-of-N skeleton + early-adopt ----
    #   Sequential (not parallel) to avoid API quota contention. The first
    #   candidate to clear `skeleton_early_adopt_threshold` is accepted
    #   and the remaining temperatures are skipped.
    paddle_for_skel = paddle_texts if len(paddle_texts) >= 50 else None
    best_svg, best_score, best_idx = None, -1.0, -1
    n_attempted = 0
    for i, t in enumerate(cc.skeleton_temperatures):
        one = compose.generate_skeleton(
            original_png, out_dir / "marked.png", labels, W, H,
            paddle_texts=paddle_for_skel,
            style_profile=style_profile,
            model=cc.designer_model,
            temperatures=[t],
        )
        if not one:
            logger.warning("[composition]   skeleton T=%.2f returned no svg", t)
            continue
        svg = one[0]
        n_attempted += 1
        cand_svg = out_dir / f"candidate_{i}.svg"
        cand_png = out_dir / f"candidate_{i}.png"
        cand_svg.write_text(svg, encoding="utf-8")
        try:
            compose.render_to_png(cand_svg, cand_png, W, H)
        except Exception as e:
            logger.warning(
                "[composition]   candidate %d (T=%.2f) render failed (%s); skipping",
                i, t, e,
            )
            continue
        if not cand_png.exists():
            logger.warning(
                "[composition]   candidate %d (T=%.2f) rendered no PNG; skipping",
                i, t,
            )
            continue
        j = quick_judge(original_png, cand_png, model=cc.quick_judge_model)
        s = j.get("overall", 0.0)
        logger.info("[composition]   skeleton cand %d (T=%.2f) score=%.2f",
                    i, t, s)
        if s > best_score:
            best_svg, best_score, best_idx = svg, s, i
        if best_score >= cc.skeleton_early_adopt_threshold:
            logger.info(
                "[composition]   skeleton early-adopt cand %d (score=%.2f >= %.2f); "
                "skipping %d remaining temperature(s)",
                best_idx, best_score, cc.skeleton_early_adopt_threshold,
                len(cc.skeleton_temperatures) - i - 1,
            )
            break
    if best_svg is None:
        raise RuntimeError(
            f"[composition] all {n_attempted} skeleton candidates "
            f"failed to render for {img_id}"
        )
    skeleton_svg_path = out_dir / "skeleton.svg"
    skeleton_svg_path.write_text(best_svg, encoding="utf-8")
    skeleton_png_path = out_dir / "skeleton.png"
    compose.render_to_png(skeleton_svg_path, skeleton_png_path, W, H)
    logger.info("[composition]   skeleton: picked cand %d score=%.2f",
                best_idx, best_score)

    # ---- E role: inject raster crops + vector primitives ----
    svg_after_raster = compose.inject_raster(
        best_svg, labels["raster"], crops_dir,
        canvas_w=W, canvas_h=H,
    )
    svg_after_vector, _vec_info = compose.inject_vector(
        svg_after_raster, labels["vector"], original_png,
        model=cc.designer_model,
    )
    composed_svg = out_dir / "composed.svg"
    composed_png = out_dir / "composed.png"
    composed_svg.write_text(svg_after_vector, encoding="utf-8")
    compose.render_to_png(composed_svg, composed_png, W, H)
    j_composed = quick_judge(original_png, composed_png,
                              model=cc.quick_judge_model)
    composed_score = j_composed.get("overall", 0.0)
    logger.info("[composition]   composed score=%.2f", composed_score)

    # ---- V + R roles: hybrid-critic refine with best-so-far ----
    refine_dir = out_dir / "refine"
    rinfo = refine.refine_loop(
        original_png, composed_svg, refine_dir, W, H,
        model=cc.reviser_model,
        max_iter=cc.refine_max_iters,
        labels_json=out_dir / "labels.json",
        paddle_texts=paddle_texts,
        enable_structural_critic=cc.enable_structural_critic,
    )
    logger.info("[composition]   refine: best score=%.2f at iter=%s",
                rinfo.get("final_score", 0.0),
                rinfo.get("best_iter", "?"))

    # ---- Materialize final.svg + final.png ----
    final_svg = out_dir / "final.svg"
    final_png = out_dir / "final.png"
    copy(rinfo["final_svg"], final_svg)
    if Path(rinfo["final_preview"]).exists():
        copy(rinfo["final_preview"], final_png)

    logger.info("[composition]   done in %.1fs — final score %.2f",
                time.time() - t0, rinfo.get("final_score", 0.0))

    return CompositionResult(
        img_id=img_id,
        out_dir=out_dir,
        final_svg=final_svg,
        final_png=final_png,
        skeleton_picked_score=best_score,
        composed_score=composed_score,
        refine_history=rinfo.get("history", []),
        refine_final_score=rinfo.get("final_score", 0.0),
    )
