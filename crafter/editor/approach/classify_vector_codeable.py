"""Classify each existing-caption icon as vector_codeable or raster.

One VLM call per image (4 calls total, parallel ~30s wall). Takes the
existing stage1_describe.json icon list (IDs preserved) + the cleaned
extracted image, and asks the model to classify each icon by ID.

Output: crafter/editor/approach/runs/imgN/vector_codeable.json

Schema:
  {
    "image": "img1",
    "n_total": 21,
    "n_vector_codeable": 5,
    "n_raster": 16,
    "items": {
      "ic01": {"vector_codeable": false, "complexity": "high",
               "reason": "filmstrip with photo content — cannot be vector-coded"},
      ...
    }
  }
"""
from __future__ import annotations
import os

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(os.environ.get("CRAFTER_HOME", str(Path(__file__).resolve().parents[2])))

from crafter.editor.approach.test_text_referring_grounding import call_vlm  # noqa: E402

HASHES = {"img1": "1cf6416a", "img2": "1efa244a",
          "img3": "3d030d50", "img4": "66ef00c9"}


CLASSIFY_PROMPT = """The image below is a CLEANED variant of an academic figure \
where text/arrows/shapes have been removed (white-filled). Only the \
non-vectorisable raster icons remain.

Below is a list of icons that were previously identified in this \
image, each with an ID, kind label, and description. For EACH icon by \
ID, decide whether it can be FAITHFULLY reproduced using only basic \
SVG primitives (<rect>, <circle>, <ellipse>, <line>, <polygon>, \
<path> with a few control points, <text>) — this is a tight bar.

Mark vector_codeable=TRUE only for things like:
  - a flat single-colour rectangle / square / patch
  - a single coloured strip with no internal pattern
  - a single line, single arrow shaft, single dot
  - a generic empty container box
  - a single text label with no decoration

Mark vector_codeable=FALSE for everything else, including (but not \
limited to):
  - photographs, video frames, real images
  - dense heatmaps, colour grids with many cells, scatter plots, \
    histograms with many bars, bar charts
  - 3D renderings (rubik's cubes, gaussian splats, 3D scatter)
  - molecular graphs, neural-network spaghetti diagrams, brain icons
  - decorative iconography (snowflakes, clocks, stars, plus marks, \
    tag marks, axes glyphs, badges, reward marks, timer icons, \
    detailed logos)
  - composite tiles (letter circles with embedded glyphs, letter \
    tiles with patterns, button icons, message panels with internal \
    structure, process panels, diagram tiles, stack tiles)
  - any icon with multiple colours / gradients / shadows / specific \
    visual style that vector code would not match

When in doubt, prefer FALSE (raster). Losing icon visual content is \
worse than wasting one raster slot.

ICON LIST:
{ICON_LIST}

Return STRICT JSON:
{{
  "items": {{
    "ic01": {{"vector_codeable": false, "complexity": "high", \
"reason": "filmstrip with photo content"}},
    "ic02": {{"vector_codeable": true,  "complexity": "low",  \
"reason": "single flat green block"}}
  }}
}}

Return ONLY the JSON.
"""


def classify_one(img: str, captions_json: Path | None = None,
                 cleaned_png: Path | None = None) -> tuple[str, dict]:
    """Tag the call_vlm with a stage label for usage tracking."""
    if captions_json is not None:
        captions_path = Path(captions_json)
    else:
        captions_path = (REPO / "crafter" / "editor" / "_runs"
                         / "text_referring_grounding" / img
                         / "stage1_describe.json")
    # Prefer the resized (VLM-input) PNG that sits next to the captions
    # — full-size cleaned.png is the fallback.
    resized = captions_path.parent / "icons_only_resized.png"
    if resized.exists():
        cleaned = resized
    elif cleaned_png is not None:
        cleaned = Path(cleaned_png)
    else:
        cleaned = (REPO / "crafter" / "editor" / "_runs"
                   / img / "extract" / "icons_only" / "cleaned.png")

    stage1 = json.loads(captions_path.read_text())
    icons = json.loads(stage1["content"])["icons"]
    icon_list = "\n".join(
        f'  {{"id":"{ic["id"]}","kind":"{ic.get("kind","")}",'
        f'"simple_desc":"{ic.get("simple_desc","")[:80]}"}}'
        for ic in icons
    )
    prompt = CLASSIFY_PROMPT.format(ICON_LIST=icon_list)
    info = call_vlm(cleaned, prompt, label="stage4_classify_vector")
    return img, icons, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imgs", nargs="+",
                    default=["img1", "img2", "img3", "img4"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--captions", default="",
                    help="Absolute path to stage1_describe.json "
                         "(single-image runs only)")
    ap.add_argument("--cleaned", default="",
                    help="Absolute path to cleaned-icons PNG "
                         "(single-image runs only)")
    ap.add_argument("--out-root", default="",
                    help="Output root (default crafter/editor/approach/runs)")
    args = ap.parse_args()
    print(f"Classifying existing icons (vector_codeable) for "
          f"{len(args.imgs)} imgs ({args.workers} parallel)...")
    t0 = time.time()
    out_root = (Path(args.out_root) if args.out_root
                else REPO / "crafter" / "editor" / "approach" / "runs")
    captions_path = Path(args.captions) if args.captions else None
    cleaned_path = Path(args.cleaned) if args.cleaned else None
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(classify_one, img, captions_path, cleaned_path): img
                   for img in args.imgs}
        for f in as_completed(futures):
            img, icons, info = f.result()
            out_dir = out_root / img
            out_dir.mkdir(parents=True, exist_ok=True)
            data = info.get("data") or {}
            classifications = data.get("items", {})
            n_vc = sum(1 for v in classifications.values()
                       if v.get("vector_codeable"))
            summary = {
                "image": img,
                "n_total": len(icons),
                "n_classified": len(classifications),
                "n_vector_codeable": n_vc,
                "n_raster": len(icons) - n_vc,
                "items": classifications,
                "raw_response": info,
            }
            (out_dir / "vector_codeable.json").write_text(
                json.dumps(summary, indent=2, ensure_ascii=False))
            print(f"  ✓ {img}: {len(icons)} icons → "
                  f"{n_vc} vector_codeable, {len(icons) - n_vc} raster   "
                  f"({info.get('elapsed', '?')}s)")
            # Dump per-image usage to file — subprocess loses the
            # module-global log, so write it for the coordinator to collate.
            try:
                from crafter.editor.approach.test_text_referring_grounding import get_vlm_usage_log
                img_log = [r for r in get_vlm_usage_log()
                           if "stage4" in r.get("label", "")]
                (out_dir / "stage4_usage.json").write_text(
                    json.dumps(img_log, indent=2))
            except Exception as e:
                print(f"  warn: usage dump failed for {img}: {e}")
    print(f"Done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
