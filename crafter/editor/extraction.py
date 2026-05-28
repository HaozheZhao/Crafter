"""Extraction phase — instruction-driven canvas cleaning.

Paper §3.3.1 (\\Editor: Harness for Raster-to-Vector Conversion,
Extraction).

Roles:
  D (designer)  — a vision-language agent inspects $a^*$ and authors a
                  per-figure keep/delete plan.
  E (executor)  — an instructable image editor (gpt-image-2) applies
                  the plan at the pixel level.
  V (verifier)  — a lightweight VLM inspects each candidate clean
                  canvas; returns either "ok" or a directive diagnostic.
  R (reviser)   — given the prior plan and the diagnostic, the
                  designer revises (closes the loop).

Loop runs for at most T=3 iterations. Empirically 47% converge at
round 1, 46% at round 2, 7% at round 3.

Output: cleaned canvas + per-element raster crops (with hallucination
filter removing blank / text-only / mis-extracted crops).
"""
from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path

from .config import Config

logger = logging.getLogger("editor_v2.extraction")

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "crafter" / "editor" / "approach"))


@dataclass
class ExtractionResult:
    img_id: str
    cleaned_png: Path
    out_dir: Path
    history: list   # one entry per verify-then-refine iteration
    final_quality: str    # "ok" | "good" | "iterating" | "error"
    n_iters: int


def run(img_id: str, original_png: Path, out_root: Path,
        config: Config) -> ExtractionResult:
    """Run the extraction harness on one raster figure.

    Implements the D→E→V→R loop directly using the production
    components from `crafter/editor/approach/stage_a_harness.py`.

    Args:
      img_id:        symbolic id for this run (used in cache paths)
      original_png:  absolute path to the input raster
      out_root:      directory to write cleaned.png + history under
                     out_root / img_id / extract
      config:        production config
    """
    from stage_a_harness import run as _stage_a_run

    out_dir = out_root / img_id / "extract"
    out_dir.mkdir(parents=True, exist_ok=True)

    cleaned_png = out_dir / "icons_only" / "cleaned.png"
    cleaned_png.parent.mkdir(parents=True, exist_ok=True)

    # SAM-only branch: pass the original raster straight to processing.
    if not getattr(config.extraction, "use_gpt_image2", True):
        import shutil
        shutil.copy(original_png, cleaned_png)
        logger.info("[extraction] %s ▸ SAM-only (gpt-image-2 skipped)", img_id)
        return ExtractionResult(
            img_id=img_id, cleaned_png=cleaned_png, out_dir=out_dir,
            history=[{"mode": "sam_only"}], final_quality="skipped", n_iters=0,
        )

    logger.info("[extraction] %s ▸ designer=%s executor=%s verifier=%s T=%d",
                img_id,
                config.extraction.designer_model,
                config.extraction.executor_model,
                config.extraction.verifier_model,
                config.extraction.max_iters)

    summary = _stage_a_run(
        img=img_id,
        orig_path=original_png,
        out_dir=out_dir,
        iter_model=config.extraction.designer_model,
        max_iters=config.extraction.max_iters,
    )
    if not cleaned_png.exists():
        raise RuntimeError(
            f"[extraction] cleaned.png missing for {img_id}: "
            f"{cleaned_png}"
        )

    return ExtractionResult(
        img_id=img_id,
        cleaned_png=cleaned_png,
        out_dir=out_dir,
        history=summary.get("history", []),
        final_quality=summary.get("final_quality", "?"),
        n_iters=summary.get("n_iters", 0),
    )
