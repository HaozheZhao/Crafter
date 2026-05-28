"""CraftEditor CLI.

Usage:
  crafter edit --img path/to/figure.png --out-dir /tmp/out
  # or: python -m crafter.editor.cli --img ... --out-dir ...
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from . import Editor
from .config import Config, DEFAULT_CONFIG


def main():
    ap = argparse.ArgumentParser(
        description="Editor — raster academic figure → editable SVG",
    )
    ap.add_argument("--img", required=True, help="input raster image")
    ap.add_argument("--out-dir", required=True, help="output directory")
    ap.add_argument("--img-id", default=None,
                    help="symbolic id (default: input filename stem)")
    ap.add_argument("--config", default=None,
                    help="path to YAML config (default: configs/default.yaml; "
                         "override with $CRAFTER_CONFIG)")
    ap.add_argument("--sam-only", action="store_true",
                    help="skip the gpt-image-2 extraction phase; pass the "
                         "original raster straight to the SAM3 grounding pass")
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    try:
        config = Config.from_yaml(args.config)
    except Exception as e:
        logger.warning("[cli] YAML config load failed (%s); falling back", e)
        config = DEFAULT_CONFIG
    if args.sam_only:
        from dataclasses import replace
        config = replace(config, extraction=replace(config.extraction, use_gpt_image2=False))
    logger.info(
        "[cli] T_extract=%d T_refine=%d gpt-image-2=%s",
        config.extraction.max_iters, config.composition.refine_max_iters,
        config.extraction.use_gpt_image2,
    )

    out = Editor(config=config).run(args.img, args.out_dir, img_id=args.img_id)
    print(f"\nFinal SVG: {out.final_svg}")
    print(f"Final PNG: {out.final_png}")
    print(f"Score:     {out.score:.2f}")
    print(f"Wall:      {out.elapsed_s}s")


if __name__ == "__main__":
    main()
