"""§7.1 caption-driven SAM3 referring — build (clean icon, accurate bbox) pairs.

Input per image:
    original_png    external_comparison/original/imgN.png
    cleaned_png     crafter/editor/.runs/imgN/extract/icons_only/cleaned.png
    captions_json   crafter/editor/.runs/text_referring_grounding/imgN/stage1_describe.json
    hint_bbox_json  crafter/editor/.runs/text_referring_grounding/imgN/stage2_simple_bbox.json
                    (gpt-image-2 reported bboxes; used as IoU sanity hint, not authoritative)

For each captioned icon:
    1. SAM3.segment_text(original, [simple_desc])
    2. If 0 detections, retry with detailed_desc.
    3. From candidates, pick the one with highest IoU vs hint bbox; tie-break
       by score, then by area closeness to hint.
    4. If best IoU < MIN_IOU, mark status=offset and keep both for review.
    5. Crop the icon raster from cleaned_png with the SAM3 bbox.

Output per image (out_dir):
    pairs.json              [{id, sam3_bbox, hint_bbox, iou, score, status, cleaned_crop, original_crop}]
    crops/<id>_cleaned.png  RGBA crop from cleaned_png at SAM3 bbox
    crops/<id>_original.png RGB  crop from original_png at SAM3 bbox
    overlay.png             original + green=SAM3 bbox / red=hint bbox / yellow text=id
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = Path(os.environ.get("CRAFTER_HOME", str(Path(__file__).resolve().parents[2])))
sys.path.insert(0, str(REPO))

from crafter.editor.raster_to_svg.tools.sam3_client import SAM3Client  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("caption_sam3")

MIN_IOU = 0.10  # below this we tag status=offset for manual review


def iou(b1: list[float], b2: list[float]) -> float:
    ax1, ay1, ax2, ay2 = b1
    bx1, by1, bx2, by2 = b2
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    a1 = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    a2 = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def best_candidate(
    candidates: list[dict],
    hint: list[float] | None,
) -> tuple[Optional[dict], float]:
    """Pick the SAM3 box whose bbox best matches the hint.

    If no hint, return the highest-score candidate.
    Returns (chosen, iou_with_hint) — iou is 0 when no hint.
    """
    if not candidates:
        return None, 0.0
    if hint is None:
        chosen = max(candidates, key=lambda c: c.get("score", 0.0))
        return chosen, 0.0
    hx1, hy1, hx2, hy2 = hint
    h_area = max(0.0, hx2 - hx1) * max(0.0, hy2 - hy1)

    def key(c):
        b = [c["x1"], c["y1"], c["x2"], c["y2"]]
        i = iou(b, hint)
        # secondary: closeness in log-area (prefer same-scale match)
        c_area = max(1.0, (b[2] - b[0]) * (b[3] - b[1]))
        log_ratio = abs(np.log(c_area / max(1.0, h_area)))
        return (i, c.get("score", 0.0), -log_ratio)

    chosen = max(candidates, key=key)
    chosen_iou = iou(
        [chosen["x1"], chosen["y1"], chosen["x2"], chosen["y2"]], hint
    )
    return chosen, chosen_iou


def crop(img: Image.Image, bbox: list[float], pad: int = 0) -> Image.Image:
    W, H = img.size
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(W, x2 + pad)
    y2 = min(H, y2 + pad)
    return img.crop((x1, y1, x2, y2))


def cleaned_to_rgba(crop_img: Image.Image, white_thresh: int = 248) -> Image.Image:
    """Convert near-white pixels in a cleaned-icon crop to alpha=0.

    The cleaned PNG fills deleted areas with white, so a crop bounded by
    SAM3 still has a white halo around the icon. RMBG would do this better
    but is heavy; for sanity-check the white-threshold is enough.
    """
    arr = np.array(crop_img.convert("RGBA"))
    rgb = arr[:, :, :3]
    white = (rgb[:, :, 0] >= white_thresh) & \
            (rgb[:, :, 1] >= white_thresh) & \
            (rgb[:, :, 2] >= white_thresh)
    arr[:, :, 3] = np.where(white, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def render_overlay(
    original: Image.Image,
    pairs: list[dict],
    out_path: Path,
) -> None:
    img = original.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf", 14
        )
    except Exception:
        font = ImageFont.load_default()

    for p in pairs:
        if p["status"] == "missing":
            continue
        if p.get("hint_bbox"):
            x1, y1, x2, y2 = p["hint_bbox"]
            draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        if p.get("sam3_bbox"):
            x1, y1, x2, y2 = p["sam3_bbox"]
            if p["status"] == "ok":
                colour = "lime"
            elif p["status"] == "ok_bbox":
                colour = "cyan"
            else:
                colour = "orange"
            draw.rectangle([x1, y1, x2, y2], outline=colour, width=3)
            label = f"{p['id']} iou={p['iou']:.2f}"
            draw.text((x1 + 2, max(0, y1 - 16)), label,
                      fill=colour, font=font)
    img.save(out_path)


def process_image(
    img_name: str,
    original_png: Path,
    cleaned_png: Path,
    captions_json: Path,
    hint_bbox_json: Path | None,
    out_dir: Path,
    sam3: SAM3Client,
    min_score: float = 0.25,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    crops_dir = out_dir / "crops"
    crops_dir.mkdir(exist_ok=True)

    captions_data = json.loads(captions_json.read_text())
    captions = json.loads(captions_data["content"])["icons"]

    hints_by_id: dict[str, list[float]] = {}
    if hint_bbox_json and hint_bbox_json.exists():
        hd = json.loads(hint_bbox_json.read_text())
        # Hints are stored with the gpt grounding done on a div-16 resized
        # original. We rescale to the true original at process time.
        bbox_data = hd.get("data") or {}
        if not bbox_data and hd.get("content"):
            try:
                bbox_data = json.loads(hd["content"])
            except Exception:
                bbox_data = {}
        for ic in bbox_data.get("icons", []):
            if "bbox" in ic and len(ic["bbox"]) == 4:
                hints_by_id[ic["id"]] = list(ic["bbox"])

    original = Image.open(original_png).convert("RGB")
    cleaned = Image.open(cleaned_png).convert("RGB")

    # If sizes differ, resize the cleaned to original (gpt-image-2 sometimes
    # crops/pads).
    if cleaned.size != original.size:
        logger.warning(
            "  cleaned size %s != original %s — resizing cleaned",
            cleaned.size, original.size,
        )
        cleaned = cleaned.resize(original.size, Image.LANCZOS)

    pairs = []
    for cap in captions:
        cid = cap["id"]
        simple = cap.get("simple_desc") or cap.get("description", "")
        detailed = cap.get("detailed_desc", "")
        hint = hints_by_id.get(cid)

        record: dict = {
            "id": cid,
            "kind": cap.get("kind", ""),
            "simple_desc": simple,
            "detailed_desc": detailed,
            "hint_bbox": hint,
            "sam3_bbox": None,
            "score": 0.0,
            "iou": 0.0,
            "status": "missing",
            "candidates_simple": 0,
            "candidates_detailed": 0,
            "cleaned_crop": None,
            "original_crop": None,
        }

        for desc_kind, desc in [("simple", simple), ("detailed", detailed)]:
            if not desc:
                continue
            try:
                cands = sam3.segment_text(
                    str(original_png), [desc],
                    min_score=min_score, return_masks=False,
                )
            except Exception as e:
                logger.warning("  %s/%s SAM3 failed: %s", cid, desc_kind, e)
                cands = []
            record[f"candidates_{desc_kind}"] = len(cands)
            if not cands:
                continue
            chosen, ch_iou = best_candidate(cands, hint)
            if chosen is None:
                continue
            record["sam3_bbox"] = [
                chosen["x1"], chosen["y1"], chosen["x2"], chosen["y2"]
            ]
            record["score"] = float(chosen.get("score", 0.0))
            record["iou"] = ch_iou
            record["status"] = "ok" if (hint is None or ch_iou >= MIN_IOU) else "offset"
            record["which_desc"] = desc_kind
            break

        # Bbox-prompted SAM3 fallback (§7.2): when text grounding missed
        # or landed in the wrong place, refine the gpt-image-2 hint with
        # SAM3's bbox-prompted segmentation. Hint accuracy: ±0-32 px.
        if hint and record["status"] in ("missing", "offset"):
            try:
                refined = sam3.segment_bbox(str(original_png), [hint])
            except Exception as e:
                logger.warning("  %s bbox-fallback failed: %s", cid, e)
                refined = []
            if refined:
                r = refined[0]
                tb = r.get("tight_box") or hint
                bb_iou = iou(tb, hint)
                # Accept refined bbox when it stayed near the hint
                if bb_iou >= 0.30 or record["status"] == "missing":
                    record["sam3_bbox"] = list(tb)
                    record["score"] = float(r.get("confidence", 0.0))
                    record["iou"] = bb_iou
                    record["status"] = "ok_bbox"
                    record["which_desc"] = "bbox_hint"

        if record["sam3_bbox"]:
            bbox = record["sam3_bbox"]
            cleaned_crop = cleaned_to_rgba(crop(cleaned, bbox))
            original_crop = crop(original, bbox)
            cp = crops_dir / f"{cid}_cleaned.png"
            op = crops_dir / f"{cid}_original.png"
            cleaned_crop.save(cp)
            original_crop.save(op)
            record["cleaned_crop"] = str(cp.relative_to(out_dir))
            record["original_crop"] = str(op.relative_to(out_dir))

        logger.info(
            "  %s [%s] iou=%.2f status=%s cands(s/d)=%d/%d",
            cid, cap.get("kind", "")[:14], record["iou"],
            record["status"], record["candidates_simple"],
            record["candidates_detailed"],
        )
        pairs.append(record)

    render_overlay(original, pairs, out_dir / "overlay.png")

    summary = {
        "image": img_name,
        "n_total": len(pairs),
        "n_ok_text": sum(1 for p in pairs if p["status"] == "ok"),
        "n_ok_bbox": sum(1 for p in pairs if p["status"] == "ok_bbox"),
        "n_offset": sum(1 for p in pairs if p["status"] == "offset"),
        "n_missing": sum(1 for p in pairs if p["status"] == "missing"),
        "mean_iou_when_present": float(
            np.mean([p["iou"] for p in pairs if p["sam3_bbox"]] or [0])
        ),
        "pairs": pairs,
    }
    (out_dir / "pairs.json").write_text(json.dumps(summary, indent=2))
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--imgs", nargs="+", default=["img1", "img3"])
    p.add_argument("--out", default=str(REPO / "crafter" / "editor" /
                                       "caption_sam3_referring" / "runs"))
    p.add_argument("--sam3-url", default=os.environ.get(
        "SAM3_SERVER_URL", ""))
    p.add_argument("--min-score", type=float, default=0.25)
    # Explicit input paths — callers (crafter.editor.processing) pass
    # the per-image canonical / cleaned / captions paths so this script
    # does not need to know any project-specific layout.
    p.add_argument("--original", default="",
                   help="Absolute path to the original raster PNG")
    p.add_argument("--cleaned", default="",
                   help="Absolute path to the cleaned (icons-only) PNG")
    p.add_argument("--captions", default="",
                   help="Absolute path to stage1_describe.json")
    p.add_argument("--hints", default="",
                   help="Absolute path to a stage2_*_bbox.json (optional)")
    args = p.parse_args()

    sam3 = SAM3Client(args.sam3_url)
    if not sam3.wait_ready(max_wait=300, interval=5):
        sys.exit(f"SAM3 not ready at {args.sam3_url}")

    out_root = Path(args.out)
    summaries = []
    for img in args.imgs:
        if args.original:
            original = Path(args.original)
        else:
            original = REPO / "external_comparison" / "original" / f"{img}.png"
        if args.cleaned:
            cleaned = Path(args.cleaned)
        else:
            cleaned = (REPO / "crafter" / "editor" / "_runs"
                       / img / "extract" / "icons_only" / "cleaned.png")
        if args.captions:
            captions = Path(args.captions)
        else:
            captions = (REPO / "crafter" / "editor" / "_runs"
                        / "text_referring_grounding" / img / "stage1_describe.json")

        # Prefer simple_bbox; fall back to detailed_bbox if simple errored.
        hints = None
        if args.hints:
            cand = Path(args.hints)
            if cand.exists():
                try:
                    cd = json.loads(cand.read_text())
                except Exception:
                    cd = {}
                if cd.get("status") == 200 and cd.get("data", {}).get("icons"):
                    hints = cand
        else:
            for cand_name in ("stage2_simple_bbox.json", "stage2_detailed_bbox.json"):
                cand = (Path(captions).parent / cand_name)
                if cand.exists():
                    try:
                        cd = json.loads(cand.read_text())
                    except Exception:
                        cd = {}
                    if cd.get("status") == 200 and cd.get("data", {}).get("icons"):
                        hints = cand
                        if cand_name == "stage2_detailed_bbox.json":
                            logger.info(
                                "  using detailed_bbox hints (simple_bbox unavailable)"
                            )
                        break

        if not original.exists():
            logger.error("Missing original: %s", original); continue
        if not cleaned.exists():
            # No-gpt-image-2 ablation: Stage 1 cleaning was skipped, so
            # fall back to grounding on the original image. The "cleaned"
            # crops will simply be regions of the original image; Stage 5
            # later strips backgrounds via RMBG-2.0.
            logger.warning("Missing cleaned: %s — falling back to original",
                            cleaned)
            cleaned = original
        if not captions.exists():
            logger.error("Missing captions: %s — need to generate first",
                         captions); continue

        out_dir = out_root / img
        logger.info("=== %s ===", img)
        s = process_image(
            img, original, cleaned, captions,
            hints if (hints is not None and hints.exists()) else None,
            out_dir, sam3, args.min_score,
        )
        summaries.append(s)
        logger.info(
            "  summary: total=%d ok_text=%d ok_bbox=%d offset=%d missing=%d mean_iou=%.2f",
            s["n_total"], s["n_ok_text"], s["n_ok_bbox"],
            s["n_offset"], s["n_missing"], s["mean_iou_when_present"],
        )

    (out_root / "summary.json").write_text(json.dumps(
        {"runs": [{k: v for k, v in s.items() if k != "pairs"}
                  for s in summaries]}, indent=2))


if __name__ == "__main__":
    main()
