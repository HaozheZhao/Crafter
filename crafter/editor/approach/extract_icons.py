"""Approach 1 Step A — extract icons by running SAM3 on the gpt-image-2
extracted image (single coordinate system, fixes Bug A).

Inputs per image:
  - Original PNG       external_comparison/original/imgN.png
  - Cleaned PNG        crafter/editor/.runs/imgN/extract/icons_only/cleaned.png
  - VLM captions       crafter/editor/.runs/text_referring_grounding/imgN/stage1_describe.json
  - SAM3-on-original   crafter/editor/caption_sam3_referring/runs/imgN/pairs.json
                        (corrected variant if available — used ONLY for placement)

Per icon (one VLM caption):
  Step 1 — SAM3 segment_text on the CLEANED image with simple_desc.
           If 0 results, fall back to detailed_desc, then to kind alone.
           From candidates, pick highest score.
  Step 2 — bbox is in cleaned-image coords. Crop the CLEANED image at
           those coords → tight RGBA (white→alpha 0).
  Step 3 — placement bbox in ORIGINAL coords: take the matching record
           from the existing SAM3-on-original pairs.json (matched by
           icon id). gpt-image-2 has 0-30 px shift, but the original-
           grounded bbox correctly localises where the icon should
           render in the final SVG.

Output per image:
  crafter/editor/approach/runs/imgN/
    pairs.json       per-icon: id, kind, simple_desc,
                        sam3_extracted_bbox, sam3_orig_bbox_for_placement,
                        cleaned_crop_path, ext_score, status
    crops/icXX.png      tight RGBA crop from cleaned image
    overlay_extracted.png  cleaned image with new lime SAM3 bboxes
    overlay_original.png   original image with original-coords bboxes (placement)
    inspection.png      grid: original-crop | extracted-crop | RGBA crop
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = Path(os.environ.get("CRAFTER_HOME", str(Path(__file__).resolve().parents[2])))
sys.path.insert(0, str(REPO))

from crafter.editor.raster_to_svg.tools.sam3_client import SAM3Client  # noqa: E402
from crafter.editor.approach.post_classify_rules import classify_override  # noqa: E402
from crafter.editor.approach import style_analyzer  # noqa: E402
from crafter.editor.approach import prompt_selector  # noqa: E402
from crafter.editor.approach import hallucination_guard  # noqa: E402

# Classification uses a
# per-icon VLM classification produced by classify_vector_codeable.py
# and stored at crafter/editor/approach/runs/<img>/vector_codeable.json.
# A SMALL post_classify_rules.classify_override() flips a few specific
# patterns the VLM tends to mis-label (vectorable bars + circle-series
# token/neuron diagrams). It does NOT touch crop bboxes — col 3 and
# col 4 of the inspection grid stay aligned (both crop cleaned at the
# same SAM3-on-extracted bbox).
# Default to raster when an icon is not present in the classification
# (e.g. classification step failed for some ids).

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("approach1_step_a")


def find_font(size: int) -> ImageFont.ImageFont:
    for p in [
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def crop_rgb(img: Image.Image, bbox, pad=0):
    W, H = img.size
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
    x2, y2 = min(W, x2 + pad), min(H, y2 + pad)
    return img.crop((x1, y1, x2, y2))


def rmbg_crop(rgb_crop: Image.Image, rmbg_url: str) -> Image.Image:
    """Send an RGB crop to the RMBG-2.0 service and return RGBA with
    background alpha-stripped. Used by the --no-gpt-image2 fallback
    path so that crops from the *original* image still come out as
    background-free icons (vs. relying on gpt-image-2 having already
    white-filled the background).
    """
    import base64 as _b64, io as _io
    import requests as _req
    buf = _io.BytesIO()
    rgb_crop.convert("RGB").save(buf, format="PNG")
    payload = {"image": _b64.b64encode(buf.getvalue()).decode()}
    r = _req.post(f"{rmbg_url.rstrip('/')}/remove",
                  json=payload, timeout=60)
    r.raise_for_status()
    rgba_b64 = r.json()["image"]
    return Image.open(_io.BytesIO(_b64.b64decode(rgba_b64))).convert("RGBA")


def cleaned_to_rgba(rgb_crop: Image.Image, white_thresh=248,
                    bg_color_tolerance=18, corner_size=4):
    """Convert a cleaned-image RGB crop into an RGBA icon by stripping
    background to alpha=0.

    Naive variant: strips only pure-white pixels, leaves coloured
    panel backgrounds visible as a halo around the icon.

    Background-aware variant: samples corner patches to detect the dominant
    background colour (white / grey / coloured panel), then strips all
    pixels within bg_color_tolerance Euclidean distance of that colour.
    Falls back to white_thresh path if corner detection finds no
    consistent background.
    """
    arr = np.array(rgb_crop.convert("RGBA"))
    h, w = arr.shape[:2]
    rgb = arr[:, :, :3].astype(np.int16)

    # Sample 4 corner patches
    cs = max(2, min(corner_size, h // 4, w // 4))
    if cs < 2:
        # Crop too small for sampling — fall back to white threshold
        white = (rgb[:, :, 0] >= white_thresh) & \
                (rgb[:, :, 1] >= white_thresh) & \
                (rgb[:, :, 2] >= white_thresh)
        arr[:, :, 3] = np.where(white, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)

    corners = np.concatenate([
        rgb[:cs, :cs].reshape(-1, 3),
        rgb[:cs, -cs:].reshape(-1, 3),
        rgb[-cs:, :cs].reshape(-1, 3),
        rgb[-cs:, -cs:].reshape(-1, 3),
    ])
    # Use median to be robust to noise + a few foreground corner pixels
    bg = np.median(corners, axis=0).astype(np.int16)

    # If bg is far from any "clean" colour (i.e. not white/grey/etc),
    # corners likely landed on the icon itself — fall back to white.
    # Heuristic: keep using corner detection, but combine WITH white
    # detection so we never miss the white-background case.
    diff = np.linalg.norm(rgb - bg, axis=2)
    bg_mask = diff <= bg_color_tolerance

    # Also strip pure white as a safety net
    white = (rgb[:, :, 0] >= white_thresh) & \
            (rgb[:, :, 1] >= white_thresh) & \
            (rgb[:, :, 2] >= white_thresh)

    transparent_mask = bg_mask | white
    arr[:, :, 3] = np.where(transparent_mask, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


MAX_PLACEMENT_OFFSET = 80   # px — gpt-image-2 shift bound + slack


_LOGO_STOPWORDS = {"the", "a", "an", "logo", "wordmark", "mark", "shield",
                   "seal", "crest", "icon", "graphic", "of"}


def _logo_match(caption_simple_desc: str,
                agent_logos: list[dict]) -> dict | None:
    """Return the agent-logo entry whose desc shares the most non-stop
    tokens with the caption (≥1 brand token overlap). Returns None
    when no overlap — we'd rather skip than guess wrong coords.
    """
    cap_tokens = {t for t in caption_simple_desc.lower().split()
                  if t and t not in _LOGO_STOPWORDS}
    if not cap_tokens:
        return None
    best = None
    best_overlap = 0
    for ld in agent_logos:
        ld_tokens = {t for t in ld["desc"].split()
                     if t and t not in _LOGO_STOPWORDS}
        ov = len(cap_tokens & ld_tokens)
        if ov > best_overlap:
            best, best_overlap = ld, ov
    return best if best_overlap >= 1 else None


def _center(bb: list[float]) -> tuple[float, float]:
    return ((bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2)


def _dist_centres(bb1: list[float], bb2: list[float]) -> float:
    c1 = _center(bb1)
    c2 = _center(bb2)
    return ((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2) ** 0.5


def ground_on_extracted(
    sam3: SAM3Client, cleaned_path: str, caption: dict,
    placement_bbox: list[float] | None,
    min_score: float = 0.20,
    max_offset: float = MAX_PLACEMENT_OFFSET,
    extra_captions: list[str] | None = None,
    prompt_addendum: str = "",
) -> tuple[list[float] | None, float, str]:
    """Try simple_desc → detailed_desc → kind. Filter by placement anchor.

    SAM3 may return multiple candidates for a caption (e.g. several
    "color_strip" instances). The placement bbox tells us roughly
    where the icon should land in the original image — the extracted
    version is at the same location ± a small gpt-image-2 shift, so we
    keep only candidates whose centre is within `max_offset` px of the
    placement centre. From the filtered set, pick highest score.

    If the filter empties the candidate list, fall back to the next
    description level (which may produce different candidates).
    """
    best_unfiltered = None  # last-ditch fallback if all filters empty

    # Build the cascade: per-caption descs first (simple → detailed → kind),
    # then style-aware extra_captions as a last resort. Each caption gets
    # the prompt_addendum appended so SAM3 sees the style framing.
    desc_cascade = []
    for dk in ("simple_desc", "detailed_desc", "kind"):
        d = caption.get(dk, "")
        if d:
            desc_cascade.append((dk, d + prompt_addendum))
    for ec in (extra_captions or []):
        desc_cascade.append(("extra_caption", ec + prompt_addendum))

    for desc_kind, desc in desc_cascade:
        try:
            cands = sam3.segment_text(cleaned_path, [desc],
                                      min_score=min_score)
        except Exception as e:
            logger.warning("  SAM3 %s failed: %s", desc_kind, e)
            continue
        if not cands:
            continue
        # Track unfiltered fallback (highest score across all desc tries)
        local_best = max(cands, key=lambda c: c.get("score", 0.0))
        if best_unfiltered is None or \
                local_best.get("score", 0) > best_unfiltered[0].get("score", 0):
            best_unfiltered = (local_best, desc_kind)

        if placement_bbox:
            kept = [
                c for c in cands
                if _dist_centres([c["x1"], c["y1"], c["x2"], c["y2"]],
                                 placement_bbox) <= max_offset
            ]
            if not kept:
                continue  # try next desc_kind, this caption matched
                          # something but in the wrong place
        else:
            kept = cands

        best = max(kept, key=lambda c: c.get("score", 0.0))
        bb = [best["x1"], best["y1"], best["x2"], best["y2"]]
        return bb, float(best.get("score", 0.0)), desc_kind

    # All desc_kinds tried; no candidate landed inside the anchor.
    # Accept the best globally as a degraded fallback (mark it).
    if best_unfiltered is not None:
        b, dk = best_unfiltered
        bb = [b["x1"], b["y1"], b["x2"], b["y2"]]
        return bb, float(b.get("score", 0.0)), f"{dk}_unanchored"

    return None, 0.0, "missing"


def overlay_with_boxes(img: Image.Image, items: list[dict],
                       bbox_field: str, color=(50, 220, 50)) -> Image.Image:
    out = img.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    font = find_font(18)
    for it in items:
        bb = it.get(bbox_field)
        if not bb:
            continue
        x1, y1, x2, y2 = [int(round(v)) for v in bb]
        for w in range(4):
            draw.rectangle([x1 - w, y1 - w, x2 + w, y2 + w], outline=color)
        label = f"{it['id']} {it.get('kind','')[:14]}"
        tb = draw.textbbox((0, 0), label, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        draw.rectangle([x1, max(0, y1 - th - 4),
                        x1 + tw + 8, max(0, y1 - th - 4) + th + 4],
                       fill=color)
        draw.text((x1 + 4, max(0, y1 - th - 3)), label,
                  fill="black", font=font)
    return out


def build_inspection(items, original, cleaned, crops_dir, out_path):
    rows = [it for it in items if it.get("sam3_extracted_bbox")]
    row_h = 130
    text_w, col_w = 320, 220
    sheet_w = text_w + col_w * 3 + 30
    sheet = Image.new("RGB", (sheet_w, row_h * len(rows) + 20),
                      (245, 245, 245))
    draw = ImageDraw.Draw(sheet)
    big = find_font(13)
    small = find_font(11)
    y = 10
    for it in rows:
        # Text col
        eb = it["sam3_extracted_bbox"]
        ob = it.get("sam3_orig_bbox_for_placement") or [0, 0, 0, 0]
        offset = it.get("centre_offset_px", "n/a")
        is_unanch = (it.get("status") == "ok_unanchored")
        # Color the row background to flag suspect anchoring
        if is_unanch:
            draw.rectangle([0, y, sheet_w, y + row_h - 1],
                           fill=(255, 230, 220))
        elif isinstance(offset, (int, float)) and offset > 60:
            draw.rectangle([0, y, sheet_w, y + row_h - 1],
                           fill=(255, 250, 220))
        lines = [
            f"{it['id']}  [{it.get('kind','')}]  desc_used={it.get('which_desc','')}",
            f"  desc: {it.get('simple_desc','')[:50]}",
            f"  ext bbox: ({eb[0]:.0f},{eb[1]:.0f},{eb[2]:.0f},{eb[3]:.0f}) score={it['ext_score']:.2f}",
            f"  orig (pos): ({ob[0]:.0f},{ob[1]:.0f},{ob[2]:.0f},{ob[3]:.0f})",
            f"  centre offset: {offset} px"
            + ("  ⚠ UNANCHORED (likely wrong instance)" if is_unanch else ""),
        ]
        for i, ln in enumerate(lines):
            color = (180, 30, 30) if is_unanch and i == 4 else (40, 40, 40)
            draw.text((10, y + 4 + i * 14), ln, fill=color, font=small)
        # Original placement crop
        if ob and ob != [0, 0, 0, 0]:
            c = crop_rgb(original, ob).convert("RGB")
            c.thumbnail((col_w - 10, row_h - 10))
            x = text_w
            sheet.paste(c, (x + (col_w - 10 - c.width) // 2,
                            y + (row_h - 10 - c.height) // 2))
        # Extracted (cleaned RGB) crop
        c = crop_rgb(cleaned, eb).convert("RGB")
        c.thumbnail((col_w - 10, row_h - 10))
        x = text_w + col_w
        sheet.paste(c, (x + (col_w - 10 - c.width) // 2,
                        y + (row_h - 10 - c.height) // 2))
        # RGBA crop on checker
        rgba_path = crops_dir / f"{it['id']}.png"
        if rgba_path.exists():
            r = Image.open(rgba_path).convert("RGBA")
            r.thumbnail((col_w - 10, row_h - 10))
            bg = Image.new("RGB", r.size, (220, 220, 220))
            bd = ImageDraw.Draw(bg)
            s = 8
            for yy in range(0, r.size[1], s):
                for xx in range(0, r.size[0], s):
                    if ((xx // s) + (yy // s)) % 2 == 0:
                        bd.rectangle([xx, yy, xx + s - 1, yy + s - 1],
                                     fill=(245, 245, 245))
            bg.paste(r, (0, 0), r)
            x = text_w + col_w * 2
            sheet.paste(bg, (x + (col_w - 10 - bg.width) // 2,
                             y + (row_h - 10 - bg.height) // 2))
        draw.line([(0, y + row_h - 1), (sheet_w, y + row_h - 1)],
                  fill=(200, 200, 200))
        y += row_h
    # Header strip
    header = Image.new("RGB", (sheet_w, 24), (60, 60, 60))
    hd = ImageDraw.Draw(header)
    hd.text((10, 4), "id / kind / desc", fill="white", font=big)
    hd.text((text_w + 10, 4), "ORIGINAL crop @ placement bbox",
            fill="white", font=big)
    hd.text((text_w + col_w + 10, 4), "EXTRACTED crop (RGB)",
            fill="white", font=big)
    hd.text((text_w + col_w * 2 + 10, 4), "RGBA (final icon, alpha)",
            fill="white", font=big)
    final = Image.new("RGB", (sheet_w, 24 + sheet.height), (245, 245, 245))
    final.paste(header, (0, 0))
    final.paste(sheet, (0, 24))
    final.save(out_path)


def process_one(
    img_name: str,
    original_path: Path,
    cleaned_path: Path,
    captions_json: Path,
    placement_pairs_json: Path,
    vector_codeable_json: Path | None,
    out_dir: Path,
    sam3: SAM3Client,
    enable_style_aware: bool = True,
    enable_hallucination_guard: bool = True,
    use_gpt_image2: bool = True,
    rmbg_url: str | None = None,
    agent_design_json: Path | None = None,
) -> dict:
    """When use_gpt_image2=False, ground SAM3 on the *original* image
    (not the cleaned one) and convert each crop to RGBA via the RMBG-
    2.0 service at rmbg_url. cleaned_path is then ignored. This is the
    ablation path for the 'gpt-image-2 contribution' question.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    crops_dir = out_dir / "crops"
    crops_dir.mkdir(exist_ok=True)
    if not use_gpt_image2 and not rmbg_url:
        raise ValueError("--no-gpt-image2 requires --rmbg-url (RMBG-2.0 service)")

    # === Step 0 — style analyzer + prompt selector ===========================
    style_profile: dict | None = None
    extract_cfg: dict
    if enable_style_aware:
        try:
            style_profile = style_analyzer.analyze(
                original_path,
                cache_path=out_dir / "style_profile.json")
            extract_cfg = prompt_selector.select(style_profile)
            logger.info("  style: %s/%s — min_score=%.2f offset_mult=%.2f "
                        "extra_caps=%d drop_unanchored=%s",
                        style_profile["style"], style_profile["complexity"],
                        extract_cfg["sam3_min_score"],
                        extract_cfg["max_offset_mult"],
                        len(extract_cfg["extra_captions"]),
                        extract_cfg["drop_unanchored"])
        except Exception as e:
            logger.warning("  style/prompt setup failed: %s — defaults", e)
            extract_cfg = prompt_selector.DEFAULT_CONFIG.copy()
            extract_cfg["extra_captions"] = []
    else:
        extract_cfg = prompt_selector.DEFAULT_CONFIG.copy()
        extract_cfg["extra_captions"] = []
    sam3_min_score = float(extract_cfg.get("sam3_min_score", 0.20))
    max_offset_eff = MAX_PLACEMENT_OFFSET * float(
        extract_cfg.get("max_offset_mult", 1.0))
    extra_captions = extract_cfg.get("extra_captions", [])
    prompt_addendum = extract_cfg.get("prompt_addendum", "")
    drop_unanchored_cfg = bool(extract_cfg.get("drop_unanchored", False))

    captions_data = json.loads(captions_json.read_text())
    captions = json.loads(captions_data["content"])["icons"]

    # Load per-icon vector_codeable classification (VLM-based, replaces
    # the rule-based classify(kind) heuristic).
    vc_by_id: dict[str, dict] = {}
    if vector_codeable_json and vector_codeable_json.exists():
        vc_by_id = json.loads(vector_codeable_json.read_text()).get(
            "items", {})
    else:
        logger.warning("  vector_codeable.json not found at %s — defaulting "
                       "all icons to raster", vector_codeable_json)

    # Load existing SAM3-on-original bboxes for placement
    placement_by_id: dict[str, list[float]] = {}
    if placement_pairs_json.exists():
        pp = json.loads(placement_pairs_json.read_text())
        pairs_list = pp.get("pairs", []) if "pairs" in pp else pp
        for p in pairs_list:
            if p.get("sam3_bbox"):
                placement_by_id[p["id"]] = list(p["sam3_bbox"])

    # Load agent_design.json — gives us logo bboxes the agent
    # identified during prompt design. Used as a fallback when SAM3
    # fails to ground a logo caption (text-styled wordmarks tend to
    # confuse SAM3 grounding even though the cleaned image preserves
    # them visibly). Schema:
    #   logos_detected: [{"desc": "...", "bbox": [x1,y1,x2,y2]}, ...]
    agent_logos: list[dict] = []
    if agent_design_json is not None:
        agent_design_path = Path(agent_design_json)
    else:
        # Legacy fallback path (kept for standalone CLI use).
        agent_design_path = (REPO / "crafter" / "editor" / "_runs"
                             / "runs" / img_name / "extract"
                             / "agent_design.json")
    if agent_design_path.exists():
        try:
            ad = json.loads(agent_design_path.read_text())
            for ld in ad.get("logos_detected") or []:
                if isinstance(ld, dict) and ld.get("bbox") and len(
                        ld["bbox"]) == 4 and ld.get("desc"):
                    agent_logos.append({
                        "desc": str(ld["desc"]).lower(),
                        "bbox": [float(x) for x in ld["bbox"]],
                    })
            if agent_logos:
                logger.info("  agent_design: %d logos available for "
                            "SAM3-fallback bypass", len(agent_logos))
        except Exception as e:
            logger.warning("  failed to read agent_design.json: %s", e)

    original = Image.open(original_path).convert("RGB")
    if use_gpt_image2:
        cleaned = Image.open(cleaned_path).convert("RGB")
        if cleaned.size != original.size:
            # rare — but we keep extracted bbox in cleaned coords; resize for
            # consistency only for the inspection image.
            cleaned = cleaned.resize(original.size, Image.LANCZOS)
        ground_path = str(cleaned_path)
        crop_source = cleaned
    else:
        # Ablation path: ground SAM3 on the *original* image directly.
        # crops also come from the original; RMBG-2.0 strips background.
        cleaned = original
        ground_path = str(original_path)
        crop_source = original
        logger.info("  --no-gpt-image2: SAM3 grounding on ORIGINAL, "
                    "RMBG-2.0 @ %s for crop alpha", rmbg_url)

    items: list[dict] = []
    skipped_vector: list[dict] = []
    n_ok = 0
    n_missing = 0
    n_unanchored = 0
    for cap in captions:
        cid = cap["id"]
        placement = placement_by_id.get(cid)
        # Per-icon classifier decides raster vs vector. Default raster
        # if classifier didn't see this id (conservative).
        vc_info = vc_by_id.get(cid, {})
        vlm_vc = vc_info.get("vector_codeable") is True
        vlm_reason = vc_info.get("reason", "")
        # Minimal rule override (vectorable bars + circle-series).
        ov = classify_override(cap.get("kind", ""),
                               cap.get("simple_desc", ""),
                               cap.get("detailed_desc", ""))
        if ov is not None:
            is_vector = ov[0]
            reason_field = f"OVERRIDE — {ov[1]}; vlm_said={vlm_vc}"
        else:
            is_vector = vlm_vc
            reason_field = vlm_reason
        # Logo guard: brand wordmarks / institution marks are NEVER
        # vector_codeable in our pipeline — even if they look like "a
        # single text label with no decoration", LLM-written SVG cannot
        # reproduce brand-specific typography.
        if is_vector:
            agent_logo_match = _logo_match(
                cap.get("simple_desc", ""), agent_logos)
            desc_lower = (cap.get("simple_desc", "") + " "
                          + cap.get("kind", "")).lower()
            keyword_logo = any(w in desc_lower for w in
                               ("logo", "wordmark", "brand", "seal",
                                "crest", "badge", "trademark"))
            if agent_logo_match or keyword_logo:
                is_vector = False
                reason_field = (f"LOGO_OVERRIDE — was vector_codeable "
                                f"({reason_field}); kept as raster because "
                                f"caption is a logo/brand mark")
                logger.info("  %s [LOGO OVERRIDE] keeping as raster "
                            "(classifier said vector)", cid)
        if is_vector:
            skipped_vector.append({
                "id": cid,
                "kind": cap.get("kind", ""),
                "simple_desc": cap.get("simple_desc", ""),
                "detailed_desc": cap.get("detailed_desc", ""),
                "placement_bbox": placement,
                "complexity": vc_info.get("complexity"),
                "vector_codeable_reason": reason_field,
            })
            continue
        ext_bbox, score, used = ground_on_extracted(
            sam3, ground_path, cap, placement,
            min_score=sam3_min_score,
            max_offset=max_offset_eff,
            extra_captions=extra_captions,
            prompt_addendum=prompt_addendum)
        # Placement bbox: in normal mode, use the pre-existing SAM3-on-
        # original record (handles gpt-image-2 shift). In the no-
        # gpt-image2 ablation, the newly-extracted bbox IS already in
        # original-image coords, so it doubles as the placement bbox.
        if use_gpt_image2:
            placement_for_rec = placement_by_id.get(cid)
        else:
            placement_for_rec = ext_bbox
        rec = {
            "id": cid,
            "kind": cap.get("kind", ""),
            "simple_desc": cap.get("simple_desc", ""),
            "detailed_desc": cap.get("detailed_desc", ""),
            "sam3_extracted_bbox": ext_bbox,
            "ext_score": score,
            "which_desc": used,
            "sam3_orig_bbox_for_placement": placement_for_rec,
            "status": "ok" if ext_bbox else "missing",
        }
        # Fix A: SAM3-fallback for logos. If SAM3 missed or anchored
        # the wrong region for a logo caption, try the agent's bbox.
        sam3_failed_logo = (
            (not ext_bbox) or used.endswith("_unanchored")
        ) and agent_logos
        agent_bbox = None
        if sam3_failed_logo:
            agent_bbox_match = _logo_match(
                cap.get("simple_desc", ""), agent_logos)
            if agent_bbox_match:
                agent_bbox = agent_bbox_match["bbox"]

        if ext_bbox and not used.endswith("_unanchored"):
            n_ok += 1
            # Crop & save. In the no-gpt-image2 fallback, send the
            # original-image crop through RMBG-2.0 instead of the
            # corner-sampling cleaned_to_rgba path.
            rgb_crop = crop_rgb(crop_source, ext_bbox)
            if use_gpt_image2:
                rgba = cleaned_to_rgba(rgb_crop)
            else:
                try:
                    rgba = rmbg_crop(rgb_crop, rmbg_url)
                except Exception as e:
                    logger.warning("  rmbg_crop failed for %s: %s — "
                                   "falling back to corner-sampling RGBA",
                                   cid, e)
                    rgba = cleaned_to_rgba(rgb_crop)
            crop_path = crops_dir / f"{cid}.png"
            rgba.save(crop_path)
            rec["cleaned_crop_path"] = str(crop_path.relative_to(out_dir))
        elif agent_bbox is not None:
            # Fix A: agent-bbox bypass for logos. Crop directly from
            # the original image (not cleaned — gpt-image-2 may have
            # subtly altered the logo even if it didn't delete it).
            n_ok += 1
            rgb_crop = crop_rgb(original, agent_bbox)
            if use_gpt_image2:
                rgba = cleaned_to_rgba(rgb_crop)
            else:
                try:
                    rgba = rmbg_crop(rgb_crop, rmbg_url)
                except Exception:
                    rgba = cleaned_to_rgba(rgb_crop)
            crop_path = crops_dir / f"{cid}.png"
            rgba.save(crop_path)
            rec["status"] = "ok_agent_bbox"
            rec["sam3_extracted_bbox"] = list(agent_bbox)
            rec["sam3_orig_bbox_for_placement"] = list(agent_bbox)
            rec["cleaned_crop_path"] = str(crop_path.relative_to(out_dir))
            rec["agent_bbox_fallback"] = True
            logger.info("  %s [LOGO BYPASS] used agent bbox %s "
                        "(SAM3 was %s)", cid, agent_bbox,
                        "missing" if not ext_bbox else "unanchored")
        elif ext_bbox and used.endswith("_unanchored"):
            # User-directed: discard unanchored — the wrong instance is
            # worse than no icon. Mark as missing for downstream.
            n_unanchored += 1
            n_missing += 1
            rec["status"] = "missing_unanchored"
            rec["sam3_extracted_bbox"] = None
        else:
            n_missing += 1
        # Distance from extracted centre to placement centre (diagnostic)
        if ext_bbox and placement:
            rec["centre_offset_px"] = round(
                _dist_centres(ext_bbox, placement), 1)
        logger.info("  %s [%s] used=%s score=%.2f offset=%s",
                    cid, rec["kind"][:14], used, score,
                    rec.get("centre_offset_px", "n/a"))
        items.append(rec)

    # === NEW: Hallucination guard ============================================
    halluc_outcome = None
    if enable_hallucination_guard and items:
        # Build verification list — use placement bbox (original-image
        # coords) so the model sees real bboxes from the unmodified figure.
        guard_items = []
        for it in items:
            bb = (it.get("sam3_orig_bbox_for_placement")
                  or it.get("sam3_extracted_bbox"))
            if not bb:
                continue
            guard_items.append({
                "id": it["id"],
                "bbox": bb,
                "kind": it.get("kind", "?"),
                "desc": (it.get("simple_desc") or it.get("detailed_desc")
                         or it.get("kind", ""))[:120],
            })
        try:
            halluc_outcome = hallucination_guard.verify(
                original_path, guard_items,
                out_dir=out_dir,
                cache_path=out_dir / "hallucination_verdicts.json",
                overlay_path=out_dir / "halluc_overlay.png",
            )
            dropped = set(halluc_outcome.get("dropped", []))
            if dropped:
                logger.info("  hallucination_guard dropped %d/%d icons: %s",
                            len(dropped), len(guard_items),
                            sorted(dropped)[:8])
            for it in items:
                if it["id"] in dropped:
                    it["status"] = "dropped_hallucinated"
                    it["sam3_extracted_bbox"] = None
                    it["hallucination_reason"] = halluc_outcome[
                        "verdicts"][it["id"]]["reason"]
        except Exception as e:
            logger.warning("  hallucination_guard failed: %s — keep all", e)

    # Overlays
    overlay_extracted = overlay_with_boxes(
        cleaned, items, "sam3_extracted_bbox", color=(50, 220, 50))
    overlay_extracted.save(out_dir / "overlay_extracted.png")
    overlay_original = overlay_with_boxes(
        original, items, "sam3_orig_bbox_for_placement", color=(50, 200, 230))
    overlay_original.save(out_dir / "overlay_original.png")

    build_inspection(items, original, cleaned, crops_dir,
                     out_dir / "inspection.png")

    n_halluc_dropped = sum(1 for it in items
                           if it.get("status") == "dropped_hallucinated")
    summary = {
        "image": img_name,
        "n_total_captions": len(items) + len(skipped_vector),
        "n_raster_kinds_attempted": len(items),
        "n_raster_ok": n_ok,
        "n_raster_unanchored_dropped": n_unanchored,
        "n_raster_missing": n_missing,
        "n_raster_hallucinated_dropped": n_halluc_dropped,
        "n_vector_kinds_skipped": len(skipped_vector),
        "style_profile": style_profile,
        "extract_config": {
            "sam3_min_score": sam3_min_score,
            "max_offset_eff": max_offset_eff,
            "extra_captions": extra_captions,
            "prompt_addendum": prompt_addendum,
            "drop_unanchored_cfg": drop_unanchored_cfg,
        },
        "hallucination_outcome": halluc_outcome,
        "raster_items": items,
        "vector_descriptors": skipped_vector,
    }
    (out_dir / "pairs.json").write_text(
        json.dumps(summary, indent=2))
    logger.info(
        "  summary: %d captions → %d raster (%d ok, %d miss, %d dropped-unanchored) "
        "+ %d vector descriptors",
        len(items) + len(skipped_vector), len(items),
        n_ok, n_missing - n_unanchored, n_unanchored,
        len(skipped_vector),
    )
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imgs", nargs="+",
                    default=["img1", "img2", "img3", "img4"])
    ap.add_argument("--out", default=str(REPO / "runs" / "extract_icons"))
    ap.add_argument("--sam3-url",
                    default=os.environ.get("SAM3_SERVER_URL", ""))
    ap.add_argument("--no-gpt-image2", action="store_true",
                    help="ABLATION: skip gpt-image-2 cleaning. Ground "
                         "SAM3 on the original image and use RMBG-2.0 "
                         "to alpha-strip each crop. Requires --rmbg-url.")
    ap.add_argument("--rmbg-url",
                    default=os.environ.get("RMBG_SERVER_URL",
                                           "http://localhost:9101"),
                    help="RMBG-2.0 service URL (only used with "
                         "--no-gpt-image2).")
    # Per-image input overrides — callers (crafter.editor.processing)
    # pass these so the script does not need to know the project layout.
    ap.add_argument("--original", default="",
                    help="Absolute path to original raster PNG "
                         "(single-image runs only)")
    ap.add_argument("--cleaned", default="",
                    help="Absolute path to cleaned-icons PNG "
                         "(single-image runs only)")
    ap.add_argument("--captions", default="",
                    help="Absolute path to stage1_describe.json "
                         "(single-image runs only)")
    ap.add_argument("--placement-pairs", default="",
                    help="Absolute path to grounded pairs.json "
                         "(single-image runs only)")
    ap.add_argument("--vector-codeable", default="",
                    help="Absolute path to vector_codeable.json "
                         "(single-image runs only)")
    ap.add_argument("--agent-design", default="",
                    help="Absolute path to extraction-phase agent_design.json "
                         "(provides logo bbox hints; single-image runs only)")
    args = ap.parse_args()
    use_g2 = not args.no_gpt_image2

    sam3 = SAM3Client(args.sam3_url)
    if not sam3.wait_ready(120, 5):
        sys.exit(f"SAM3 not ready at {args.sam3_url}")

    out_root = Path(args.out)

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
                        / "text_referring_grounding" / img
                        / "stage1_describe.json")
        if args.placement_pairs:
            placement_pairs = Path(args.placement_pairs)
        else:
            placement = (REPO / "crafter" / "editor"
                         / "caption_sam3_referring" / "runs" / img)
            pp_corrected = placement / "pairs_corrected.json"
            pp_orig = placement / "pairs.json"
            placement_pairs = pp_corrected if pp_corrected.exists() else pp_orig
        if args.vector_codeable:
            vc_json = Path(args.vector_codeable)
        else:
            vc_json = (REPO / "crafter" / "editor" / "approach" / "runs"
                       / img / "vector_codeable.json")

        agent_design = Path(args.agent_design) if args.agent_design else None

        out_dir = out_root / img
        logger.info("=== %s — out=%s ===", img, out_dir.name)
        process_one(img, original, cleaned, captions,
                    placement_pairs, vc_json, out_dir, sam3,
                    use_gpt_image2=use_g2,
                    rmbg_url=(None if use_g2 else args.rmbg_url),
                    agent_design_json=agent_design)

        # Dump per-image usage to file — subprocess loses the
        # module-global usage log on exit.
        try:
            from crafter.editor.approach.iter_svg_fix import get_usage_log
            img_log = [r for r in get_usage_log()
                       if "stage5" in r.get("label", "")]
            (out_dir / "stage5_usage.json").write_text(
                json.dumps(img_log, indent=2))
        except Exception as e:
            logger.warning("  stage5 usage dump failed: %s", e)


if __name__ == "__main__":
    main()
