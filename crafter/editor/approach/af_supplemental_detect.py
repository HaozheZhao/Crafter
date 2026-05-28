"""AF-style supplemental SAM3 detection.

Diagnosed gap: our caption-driven Stage 3 grounding has 80-100% recall on
simple academic figures (≤17 icons) but 22-42% on dense posters/
infographics (≥32 icons). The failure mode is REFERRING EXPRESSION
DISAMBIGUATION:
  - VLM caption: "the first drag source portrait"
  - SAM3 with this expression: 0 or 1 boxes (cannot semantically rank
    multiple instances of the same icon class).

AF's approach handles this differently: it uses BROAD CLASS PROMPTS to
SAM3 ("icon", "person", "robot", "animal", "logo", "diagram", "chart")
and gets ALL boxes of each class in a single call. Then box-merge
threshold 0.9 dedups overlaps.

This module ADDS AF-style detection as a supplemental layer on top of
our existing Stage 3 output, NOT replacing it. The combined box set
preserves our caption-driven precision (specific icons → specific
captions for downstream) while gaining AF's recall on dense scenes.

Pipeline:
  1. Read existing pairs.json (Stage 3 output, caption-driven).
  2. Run SAM3 with AF_PROMPT_LIST on the cleaned image.
  3. For each AF-detected box:
       - if IoU >= 0.4 with any existing box → discard (already covered).
       - else → it's a NEW icon. Add to pairs with kind=<af_prompt>,
         simple_desc=f"the {af_prompt} at ({x},{y})".
  4. Write enriched pairs.json + supplemental_log.json.

Downstream (Stage 4 classify, Stage 5 extract) treats these new
items identically to caption-driven ones.

Cost: 1 extra SAM3 call (free, local server). 0 LLM calls per
supplemental icon — we use a templated caption rather than a VLM
caption to avoid extra spend; the kind is rich enough for downstream
classification.
"""
from __future__ import annotations
import os

import json
import logging
import sys
from pathlib import Path

REPO = Path(os.environ.get("CRAFTER_HOME", str(Path(__file__).resolve().parents[2])))
sys.path.insert(0, str(REPO))

from crafter.editor.raster_to_svg.tools.sam3_client import SAM3Client

logger = logging.getLogger("af_supplemental_detect")


# tightened prompt list + threshold after measurement.
# a prior measurement added all boxes at score >= 0.3 → 50 supplementals, of which
# the high-score (>=0.85) ones were real icons (portraits) but the
# low-score (<0.5) ones were noise (decorative shapes, text fragments,
# whole-panel-sized 'diagram' detections that wreck compose).
#
# Tightened defaults:
#  - min_score raised 0.3 → 0.5 (drops the noisy bottom half)
#  - removed catch-all "icon" prompt (generated 29 hits, mostly noise)
#  - removed "diagram" prompt (3 of 4 lowest-scoring detections were
#    huge "diagram" boxes that overlapped real panels by being just
#    inside their bounds, escaping the 0.4 IoU dedup)
#  - kept specific real-world classes
AF_PROMPT_LIST = [
    "person",        # portraits, faces, avatars (highest-precision class)
    "robot",         # robot icons, AI agents
    "animal",        # animal symbols
    "chart",         # bar/line charts
    "thumbnail",     # photo thumbnails
    "logo",          # institutional logos
]
AF_MIN_SCORE = 0.5
AF_IOU_DEDUP = 0.4   # boxes with IoU >= this vs existing → discard


def _iou(a: tuple, b: tuple) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def _existing_boxes(pairs: list[dict]) -> list[tuple]:
    """Extract bboxes from existing Stage-3 pairs entries."""
    out = []
    for p in pairs:
        bb = p.get("sam3_bbox")
        if bb and len(bb) == 4:
            out.append(tuple(bb))
    return out


def af_detect_and_merge(
    cleaned_png: Path,
    pairs_json: Path,
    sam3_client: SAM3Client,
    out_log_json: Path | None = None,
    captions_json: Path | None = None,
    prompt_list: list[str] = AF_PROMPT_LIST,
    min_score: float = AF_MIN_SCORE,
    iou_dedup: float = AF_IOU_DEDUP,
    max_supplemental: int = 50,
) -> dict:
    """Run AF-style supplemental SAM3 detection, merge into pairs.

    Mutates pairs.json in place: adds new pairs with status='ok_af_supplemental'.
    Returns a log dict with stats.
    """
    pairs_data = json.loads(pairs_json.read_text())
    pairs = pairs_data.get("pairs", pairs_data) if isinstance(pairs_data, dict) else pairs_data
    is_dict = isinstance(pairs_data, dict)
    existing = _existing_boxes(pairs)
    n_before = len(pairs)
    log = {"prompt_list": prompt_list, "min_score": min_score,
           "iou_dedup": iou_dedup, "n_before": n_before,
           "per_prompt": {}, "added": [], "errors": []}

    next_id = 1
    while any(p.get("id") == f"af{next_id:02d}" for p in pairs):
        next_id += 1

    # Single SAM3 call with all prompts comma-separated (matches AF style)
    try:
        all_results = sam3_client.segment_text(
            str(cleaned_png), prompt_list, min_score=min_score, return_masks=False,
        )
    except Exception as e:
        log["errors"].append(f"SAM3 segment_text failed: {type(e).__name__}: {e}")
        if out_log_json:
            out_log_json.write_text(json.dumps(log, indent=2))
        return log

    # Group by prompt for stats
    by_prompt: dict[str, list[dict]] = {}
    for r in all_results:
        by_prompt.setdefault(r.get("prompt", "?"), []).append(r)
    log["per_prompt"] = {k: len(v) for k, v in by_prompt.items()}

    # Dedup against existing + among themselves (highest score wins per region)
    af_boxes = sorted(all_results, key=lambda r: -float(r.get("score", 0)))
    # a prior measurement: per-prompt cap. Any single prompt class returning >15
    # boxes is suspicious — likely catching decorations rather than
    # real distinct icons. Cap at 15 per prompt to keep diversity.
    per_prompt_cap = 15
    per_prompt_count: dict[str, int] = {}
    accepted: list[tuple[tuple, str, float]] = []  # (bbox, prompt, score)
    for r in af_boxes:
        if len(accepted) >= max_supplemental:
            break
        bb = (int(r["x1"]), int(r["y1"]), int(r["x2"]), int(r["y2"]))
        prompt = r.get("prompt", "icon")
        # per-prompt cap
        if per_prompt_count.get(prompt, 0) >= per_prompt_cap:
            continue
        # vs existing pairs
        if any(_iou(bb, eb) >= iou_dedup for eb in existing):
            continue
        # vs already-accepted supplementals (avoid intra-AF duplicates)
        if any(_iou(bb, ab) >= 0.7 for ab, _, _ in accepted):
            continue
        # filter tiny boxes (likely false positives)
        w = bb[2] - bb[0]
        h = bb[3] - bb[1]
        if w < 12 or h < 12 or w * h < 250:
            continue
        # a prior measurement: filter HUGE boxes (whole-panel false positives that
        # don't quite reach 0.4 IoU with our small icons). Anything
        # > 25% of the canvas area is unlikely to be a real icon.
        # We don't have the canvas dims here cheaply; use absolute
        # area cap of 80,000 px² (= ~283×283).
        if w * h > 80_000:
            continue
        accepted.append((bb, prompt, float(r.get("score", 0))))
        per_prompt_count[prompt] = per_prompt_count.get(prompt, 0) + 1

    # Append accepted as new pairs entries with templated caption
    for bb, prompt, score in accepted:
        new_id = f"af{next_id:02d}"
        next_id += 1
        cx = (bb[0] + bb[2]) // 2
        cy = (bb[1] + bb[3]) // 2
        pairs.append({
            "id": new_id,
            "kind": prompt,
            "simple_desc": f"the {prompt} at ({cx},{cy})",
            "detailed_desc": (f"a {prompt} region detected by SAM3 broad-class "
                              f"grounding (score {score:.2f}) at bbox {list(bb)}"),
            "hint_bbox": list(bb),
            "sam3_bbox": list(bb),
            "score": score,
            "iou": 1.0,
            "status": "ok_af_supplemental",
            "candidates_simple": 1,
            "candidates_detailed": 0,
            "which_desc": "af_class_prompt",
            "vector_codeable": False,  # default to raster — extracted as <image>
        })
        log["added"].append({
            "id": new_id, "prompt": prompt, "score": score,
            "bbox": list(bb), "size": [bb[2] - bb[0], bb[3] - bb[1]],
        })

    log["n_added"] = len(accepted)
    log["n_after"] = len(pairs)

    # Write back pairs.json
    if is_dict:
        pairs_data["pairs"] = pairs
        pairs_json.write_text(json.dumps(pairs_data, indent=2))
    else:
        pairs_json.write_text(json.dumps(pairs, indent=2))

    # Also extend stage1_describe.json (Stage 2 captions output) so
    # extract_icons.py picks up the supplemental icons. Stage 5
    # iterates over `captions` from this file, not pairs directly.
    if captions_json and captions_json.exists() and accepted:
        try:
            cdata = json.loads(captions_json.read_text())
            content = json.loads(cdata["content"])
            icons = content.get("icons", [])
            existing_ids = {ic.get("id") for ic in icons}
            n_added_to_captions = 0
            for entry in pairs:
                if entry.get("status") != "ok_af_supplemental":
                    continue
                if entry["id"] in existing_ids:
                    continue
                bb = entry["sam3_bbox"]
                cx = (bb[0] + bb[2]) // 2
                cy = (bb[1] + bb[3]) // 2
                icons.append({
                    "id": entry["id"],
                    "kind": entry["kind"],
                    "simple_desc": entry["simple_desc"],
                    "detailed_desc": entry["detailed_desc"],
                    "simple_bbox": bb,
                    "detailed_bbox": bb,
                    "vector_codeable": False,
                    "_af_supplemental": True,
                })
                n_added_to_captions += 1
            content["icons"] = icons
            cdata["content"] = json.dumps(content)
            captions_json.write_text(json.dumps(cdata, indent=2))
            log["n_added_to_captions"] = n_added_to_captions
        except Exception as e:
            log["errors"].append(
                f"failed to update captions_json: {type(e).__name__}: {e}")

    if out_log_json:
        out_log_json.write_text(json.dumps(log, indent=2))

    logger.info("AF supplemental: +%d boxes (n_before=%d → n_after=%d)",
                len(accepted), n_before, len(pairs))
    return log


def main():
    """CLI entry: enrich an existing pairs.json with AF-supplemental boxes."""
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cleaned-png", required=True)
    ap.add_argument("--pairs-json", required=True)
    ap.add_argument("--sam3-url", default=None,
                    help="overrides .sam3_server_url file")
    ap.add_argument("--out-log", default=None)
    args = ap.parse_args()
    if args.sam3_url is None:
        crafter_home = Path(os.environ.get(
            "CRAFTER_HOME", str(Path(__file__).resolve().parents[2])))
        args.sam3_url = (crafter_home / ".sam3_server_url").read_text().strip()
    client = SAM3Client(server_url=args.sam3_url)
    log = af_detect_and_merge(
        Path(args.cleaned_png), Path(args.pairs_json), client,
        out_log_json=Path(args.out_log) if args.out_log else None,
    )
    print(json.dumps(log, indent=2))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
