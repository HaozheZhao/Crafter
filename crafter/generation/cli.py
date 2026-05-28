"""Crafter CLI — text + paper context → AI raster figure.

This is the single-shot user-facing CLI. For benchmark batch runs
across a benchmark, see ``inference.py`` which
wraps this same pipeline over a sample manifest.

Usage:
    crafter generate --caption "Figure 1: ..." --paper-text "..." --out raster.png
    CRAFTER_CONFIG=configs/openrouter.yaml crafter generate ...
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="crafter generate",
        description="figCraft (§4.2) — generate an AI raster academic figure from text",
    )
    parser.add_argument("--caption", required=True,
                        help="Figure caption / brief")
    parser.add_argument("--paper-text", default="",
                        help="Paper methodology text")
    parser.add_argument("--paper-text-file", default="",
                        help="Path to text file with paper context")
    parser.add_argument("--out", default="raster.png",
                        help="Output raster path (default: raster.png)")
    parser.add_argument("--out-dir", default="",
                        help="Directory for run artifacts (default: same dir as --out)")
    parser.add_argument("--figure-type", default="method_pipeline",
                        choices=["method_pipeline", "t2i", "inpaint",
                                 "keyelems", "sketch", "poster", "infographic"])
    parser.add_argument("--venue", default="neurips")
    parser.add_argument("--role", default="academic",
                        choices=["academic", "poster", "infographic"])
    parser.add_argument("--reference", action="append", default=[],
                        help="Optional reference image (repeatable)")
    parser.add_argument("--config", default="",
                        help="Config path (else $CRAFTER_CONFIG or default.yaml)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    paper_text = args.paper_text
    if args.paper_text_file:
        ptf = Path(args.paper_text_file)
        if not ptf.exists():
            sys.exit(f"--paper-text-file not found: {ptf}")
        paper_text = ptf.read_text(encoding="utf-8")

    out_path = Path(args.out).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    from crafter.pipeline import Pipeline
    pipe = Pipeline(config_path=args.config or None)
    result = pipe.run(
        caption=args.caption,
        paper_text=paper_text,
        out_dir=str(out_dir),
        figure_type=args.figure_type,
        venue=args.venue,
        role=args.role,
        reference_paths=args.reference,
        skip_editor=True,
    )

    # Move raster to user-specified path
    if result.raster_png and Path(result.raster_png) != out_path:
        shutil.copyfile(result.raster_png, out_path)
        result.raster_png = str(out_path)

    print(json.dumps({
        "raster_png": result.raster_png,
        "duration_s": round(result.duration_s, 1),
        "out_dir":    result.out_dir,
        "selected_style": result.raster_meta.get("selected_style", ""),
        "run_id":     result.raster_meta.get("run_id", ""),
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
