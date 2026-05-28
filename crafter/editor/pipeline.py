"""Editor orchestrator — runs extraction → processing → composition.

Mirrors paper §3.3 (\\Editor: Harness for Raster-to-Vector Conversion).
Single public entry: `Editor.run(input_image, out_dir)`.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from shutil import copy
from typing import Any

from . import composition, extraction, processing
from .config import DEFAULT_CONFIG, Config

logger = logging.getLogger("editor_v2")


_REPO = Path(__file__).resolve().parents[2]
_CANONICAL_INPUT_DIR = _REPO / "external_comparison" / "original"


@dataclass
class RunOutputs:
    """Result of one Editor.run() call."""
    img_id: str
    out_dir: Path
    final_svg: Path
    final_png: Path
    score: float                  # composition refine final score
    elapsed_s: float
    extraction: Any
    processing: Any
    composition: Any
    raw_summary: dict = field(default_factory=dict)


class Editor:
    """Convert a raster academic figure into an editable SVG.

    Three phases run in sequence:
      1. extraction  — instruction-driven canvas cleaning (D→E→V→R, T=3)
      2. processing  — caption / ground / classify / extract
      3. composition — skeleton best-of-N → inject → refine (D→E→V→R, T=4)

    All settings come from `editor_v2.config.Config` (production defaults).
    """

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or DEFAULT_CONFIG

    def run(self, input_image: Path | str, out_dir: Path | str,
            img_id: str | None = None) -> RunOutputs:
        input_image = Path(input_image).resolve()
        out_dir = Path(out_dir).resolve()
        if not input_image.exists():
            raise FileNotFoundError(f"input image missing: {input_image}")
        out_dir.mkdir(parents=True, exist_ok=True)
        img_id = img_id or input_image.stem

        # The internal modules (caption / classify / extract / harness)
        # read their input from external_comparison/original/<img>.png.
        # Copy the user input there under img_id so callers can supply
        # arbitrary paths. We use copy rather than symlink because
        # concurrent symlink creation in the same directory triggers
        # spurious ENOSPC errors on distributed filesystems (BeeGFS-style
        # metadata contention).
        canonical = _CANONICAL_INPUT_DIR / f"{img_id}.png"
        canonical.parent.mkdir(parents=True, exist_ok=True)
        if canonical.is_symlink() or canonical.exists():
            try:
                if not canonical.is_symlink() and canonical.resolve() == input_image.resolve():
                    pass  # already a copy of the same source
                else:
                    canonical.unlink()
            except FileNotFoundError:
                pass
        if not canonical.exists():
            copy(input_image, canonical)

        logger.info("editor_v2 ▸ %s ← %s", img_id, input_image)
        t0 = time.time()

        # --- Phase 1: extraction (instruction-driven canvas cleaning) ---
        ex_result = extraction.run(img_id, canonical, out_dir, self.config)

        # --- Phase 2: processing (caption / ground / classify / extract) ---
        pr_result = processing.run(img_id, canonical, ex_result.cleaned_png,
                                     out_dir, self.config)

        # --- Phase 3: composition (skeleton → inject → refine → polish) ---
        co_result = composition.run(img_id, canonical, pr_result.pairs_json,
                                      out_dir, self.config)

        # Materialize phase-3 outputs at the run root for easy discovery
        final_svg = out_dir / img_id / "final.svg"
        final_png = out_dir / img_id / "final.png"
        copy(co_result.final_svg, final_svg)
        copy(co_result.final_png, final_png)

        elapsed = round(time.time() - t0, 1)
        summary = {
            "img_id": img_id,
            "input": str(input_image),
            "out_dir": str(out_dir / img_id),
            "final_svg": str(final_svg),
            "final_png": str(final_png),
            "score": co_result.refine_final_score,
            "elapsed_s": elapsed,
            "extraction": {
                "n_iters": ex_result.n_iters,
                "final_quality": ex_result.final_quality,
            },
            "processing": {
                "n_total_captions": pr_result.n_total_captions,
                "n_raster_ok": pr_result.n_raster_ok,
                "n_vector_descriptors": pr_result.n_vector_descriptors,
                "n_af_supplemental": pr_result.n_af_supplemental,
            },
            "composition": {
                "skeleton_picked_score": co_result.skeleton_picked_score,
                "composed_score": co_result.composed_score,
                "refine_final_score": co_result.refine_final_score,
            },
        }
        (out_dir / img_id / "summary.json").write_text(
            json.dumps(summary, indent=2)
        )
        logger.info("editor_v2 ▸ DONE %s  score=%.2f  wall=%.1fs",
                    img_id, co_result.refine_final_score, elapsed)

        return RunOutputs(
            img_id=img_id,
            out_dir=out_dir / img_id,
            final_svg=final_svg,
            final_png=final_png,
            score=co_result.refine_final_score,
            elapsed_s=elapsed,
            extraction=ex_result,
            processing=pr_result,
            composition=co_result,
            raw_summary=summary,
        )
