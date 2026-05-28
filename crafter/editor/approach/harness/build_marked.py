"""Step 1 of the Stage B harness: build the marked image.

The marked image is what the SVG-skeleton LLM call sees. It shows
where each element should land in the final SVG by drawing labelled
boxes on top of the original:

  RAS_NN   solid grey fill, black outline   (raster-icon target —
           later replaced by <image href=base64>)
  VEC_NN   no fill, dashed cyan outline     (vector-element target —
           later replaced by an LLM-generated <g> fragment)

Outputs:
  marked.png       PNG with all boxes drawn
  labels.json      {"raster": [{label, id, kind, simple_desc, bbox}],
                   "vector":  [{label, id, kind, simple_desc, bbox}]}
"""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def find_font(size: int) -> ImageFont.ImageFont:
    for p in [
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def build(
    original_png: Path,
    pairs: dict,
    out_dir: Path,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    img = Image.open(original_png).convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    font = find_font(14)

    raster_meta = []
    vector_meta = []

    for i, it in enumerate(pairs.get("raster_items", [])):
        if not it.get("sam3_orig_bbox_for_placement"):
            continue
        x1, y1, x2, y2 = [int(round(v)) for v in
                          it["sam3_orig_bbox_for_placement"]]
        if x2 <= x1 or y2 <= y1:
            continue
        label = f"RAS_{i + 1:02d}"
        draw.rectangle([x1, y1, x2, y2], fill="#808080", outline="#000000",
                       width=2)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        draw.text((cx, cy), label, fill="white", anchor="mm", font=font)
        raster_meta.append({
            "label": label,
            "id": it["id"],
            "kind": it.get("kind", ""),
            "simple_desc": it.get("simple_desc", ""),
            "bbox": [x1, y1, x2, y2],
            "cleaned_crop_path": it.get("cleaned_crop_path"),
        })

    for i, v in enumerate(pairs.get("vector_descriptors", [])):
        bb = v.get("placement_bbox")
        if not bb:
            continue
        x1, y1, x2, y2 = [int(round(c)) for c in bb]
        if x2 <= x1 or y2 <= y1:
            continue
        label = f"VEC_{i + 1:02d}"
        # dashed cyan outline (no fill so panel/text behind is visible)
        for w in range(2):
            draw.rectangle([x1 - w, y1 - w, x2 + w, y2 + w],
                           outline="#00bcd4")
        # cheap dash overlay
        d_step = 8
        for dx in range(x1, x2, d_step):
            draw.line([dx, y1, min(dx + 3, x2), y1], fill="white", width=1)
            draw.line([dx, y2, min(dx + 3, x2), y2], fill="white", width=1)
        for dy in range(y1, y2, d_step):
            draw.line([x1, dy, x1, min(dy + 3, y2)], fill="white", width=1)
            draw.line([x2, dy, x2, min(dy + 3, y2)], fill="white", width=1)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        draw.text((cx, cy), label, fill="#00bcd4", anchor="mm", font=font)
        vector_meta.append({
            "label": label,
            "id": v["id"],
            "kind": v.get("kind", ""),
            "simple_desc": v.get("simple_desc", ""),
            "detailed_desc": v.get("detailed_desc", ""),
            "bbox": [x1, y1, x2, y2],
        })

    img.save(out_dir / "marked.png", quality=92)
    labels = {"raster": raster_meta, "vector": vector_meta}
    (out_dir / "labels.json").write_text(json.dumps(labels, indent=2))
    return labels
