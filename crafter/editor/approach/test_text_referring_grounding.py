"""Two-stage text-referring grounding:

  STAGE 1 (description): VLM reads icons_only and produces, for each
    visible icon, BOTH a simple phrase AND a detailed text description.
  STAGE 2 (grounding):   VLM gets the original image + each description
    and returns the icon's location.

Test 4 variants of stage 2:
  (a) simple_desc → bbox      (red)
  (b) simple_desc → pinpoint  (green)
  (c) detailed_desc → bbox    (orange)
  (d) detailed_desc → pinpoint (blue)

5 calls total per image (1 description + 4 grounding); all parallel.

All images are resized to div-16 (preserving aspect ratio).
Visualisations drawn on the RESIZED ORIGINAL (where stage-2 ran).
"""
from __future__ import annotations
import base64
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import requests
from PIL import Image

REPO = Path(os.environ.get("CRAFTER_HOME", str(Path(__file__).resolve().parents[2])))
# Unified API_KEY with backward-compat fallback to alternative env names.
OPENROUTER_KEY = (os.environ.get("API_KEY")
            or os.environ.get("OPENROUTER_API_KEY")
            or os.environ.get("LLM_API_KEY")
            or "")
OPENROUTER_URL = (os.environ.get("API_ENDPOINT", "https://openrouter.ai/api/v1").rstrip("/")
            + "/chat/completions")

IMAGES = [("img1", "1cf6416a"), ("img2", "1efa244a"),
          ("img3", "3d030d50"), ("img4", "66ef00c9")]
OUT_BASE = REPO / "crafter" / "editor" / "_runs" / "text_referring_grounding"


# ---- helpers ---------------------------------------------------------

def round16(v): return ((v + 8) // 16) * 16


def resize_to_div16(img: Image.Image) -> Image.Image:
    W, H = img.size
    aspect = W / H
    cands = []
    for w16 in range(round16(W) - 16, round16(W) + 17, 16):
        if w16 < 16: continue
        h16 = round16(int(round(w16 / aspect)))
        if h16 < 16: continue
        cands.append((abs(w16 / h16 - aspect) / aspect, w16, h16))
    cands.sort()
    _, tw, th = cands[0]
    if (tw, th) == (W, H): return img
    return img.resize((tw, th), Image.LANCZOS)


def encode_b64(p: Path) -> str:
    return base64.b64encode(p.read_bytes()).decode("ascii")


def parse_json(text):
    text = (text or "").strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if m: text = m.group(1).strip()
    try: return json.loads(text)
    except: pass
    for i, c in enumerate(text):
        if c == "{":
            d = 0
            for j in range(i, len(text)):
                if text[j] == "{": d += 1
                elif text[j] == "}":
                    d -= 1
                    if d == 0:
                        try: return json.loads(text[i:j+1])
                        except: break
    return None


# Self-contained VLM usage log for cross-process instrumentation. Each
# subprocess maintains its own log and dumps it on completion; the
# coordinator collates the per-stage files.
_VLM_USAGE_LOG: list[dict] = []


def reset_vlm_usage_log() -> None:
    _VLM_USAGE_LOG.clear()


def get_vlm_usage_log() -> list[dict]:
    return list(_VLM_USAGE_LOG)


def call_vlm(image_path: Path, prompt: str,
             model: str = "openai/gpt-5.5",
             detail: str = "original",
             max_tokens: int = 16000,
             label: str | None = None) -> dict:
    """VLM call. Now delegates to editor.providers.llm via the LLMProvider
    interface (vision messages format). Returns a dict.

    `label` tags this call for token-usage attribution.
    """
    import sys as _sys
    from pathlib import Path as _Path
    _root = _Path(__file__).resolve().parents[2]
    if str(_root) not in _sys.path:
        _sys.path.insert(0, str(_root))
    from crafter.shared.providers.llm import get_default_llm  # noqa: E402
    b64 = encode_b64(image_path)
    messages = [{
        "role": "user",
        "content": [
            {"type": "image_url",
             "image_url": {"url": f"data:image/png;base64,{b64}",
                            "detail": detail}},
            {"type": "text", "text": prompt},
        ],
    }]
    t0 = time.time()
    try:
        resp = get_default_llm().chat(messages, model=model,
                                        max_tokens=max_tokens, label=label)
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}",
                "elapsed": round(time.time() - t0, 1)}
    info = {
        "status": 200,
        "elapsed": resp.elapsed_s,
        "usage": {
            "prompt_tokens": resp.prompt_tokens,
            "completion_tokens": resp.completion_tokens,
            "total_tokens": resp.total_tokens,
        },
    }
    # Mirror into local module-global log (preserves get_vlm_usage_log() API)
    _VLM_USAGE_LOG.append({
        "model": resp.model,
        "label": label or "unlabeled",
        "elapsed_s": resp.elapsed_s,
        "prompt_tokens": resp.prompt_tokens,
        "completion_tokens": resp.completion_tokens,
        "total_tokens": resp.total_tokens,
    })
    content = resp.text
    info["content"] = content
    info["data"] = parse_json(content)
    return info


# ---- prompts ---------------------------------------------------------

DESCRIBE_PROMPT = """The image below is a CLEANED variant of an academic figure \
where text/arrows/shapes have been removed (white-filled). Only \
non-vectorisable raster icons remain.

For EACH distinct icon, provide:
  • simple_desc: a short referring phrase (3-8 words) — e.g. "the dog \
    photograph", "the 4x4 attention heatmap", "the green color grid", \
    "the bar histogram with truncation".
  • detailed_desc: a sentence describing it in detail including its \
    visual content, position relative to other icons, distinctive \
    colors / patterns / sizes — enough info for someone to find this \
    icon among many similar ones.
  • vector_codeable (boolean): TRUE if this icon could be faithfully \
    reproduced using a small number of basic SVG primitives (<rect>, \
    <circle>, <ellipse>, <line>, <polygon>, <path> with a few control \
    points, <text>) — i.e. its visual content is a flat colour, a \
    single solid shape, a simple line/arrow, a single label, or a \
    geometric primitive a code-generation model could draw from a \
    short description. FALSE if it requires raster preservation: \
    photographs, video frames, dense heatmaps / colour grids / \
    histograms / scatter plots, 3D renderings (rubik's cube, \
    gaussian splats), molecular graphs, neural-network spaghetti, \
    intricate patterns, decorative iconography (snowflakes, clocks, \
    detailed logos, axes glyphs, plus / star / tag marks with \
    distinct visual style, badges, letter circles with embedded \
    glyphs, stacked tile composites, etc.). When in doubt, prefer \
    FALSE — losing icon content is worse than wasting one raster slot.
  • complexity (string): "low", "medium", or "high" — overall visual \
    intricacy (number of distinct colours, shapes, sub-elements).

Return STRICT JSON:
{{
  "icons": [
    {{
      "id": "ic01",
      "kind": "filmstrip",
      "simple_desc": "the long vertical filmstrip on the left",
      "detailed_desc": "a tall vertical strip of about six road-view photos \
stacked from top to bottom, located along the left edge, with film-perforation \
borders. The frames show a road with cars and trees.",
      "vector_codeable": false,
      "complexity": "high"
    }},
    {{
      "id": "ic02",
      "kind": "color_strip",
      "simple_desc": "the small green flat block",
      "detailed_desc": "a uniform green rectangle in the middle row.",
      "vector_codeable": true,
      "complexity": "low"
    }}
  ]
}}

Be exhaustive. Return ONLY the JSON.
"""


GROUND_BBOX_PROMPT = """The image is exactly {W}x{H} pixels. Below is a list of \
{N} icons. For EACH icon, find it in this image and return its TIGHT \
bounding box in PIXEL coordinates.

ICONS:
{ICON_LIST}

Return STRICT JSON:
{{
  "icons": [
    {{"id": "ic01", "bbox": [x1, y1, x2, y2]}}
  ]
}}

bbox = [x1, y1, x2, y2] in pixel coords of this {W}x{H} image. If you \
truly cannot find an icon, omit it from the list.

Return ONLY the JSON.
"""


GROUND_POINT_PROMPT = """The image is exactly {W}x{H} pixels. Below is a list of \
{N} icons. For EACH icon, find it in this image and return its CENTER \
PIN-POINT in PIXEL coordinates.

ICONS:
{ICON_LIST}

Return STRICT JSON:
{{
  "icons": [
    {{"id": "ic01", "point": [x, y]}}
  ]
}}

point = [x, y] in pixel coords of this {W}x{H} image, locating the \
visual center of the icon. If you truly cannot find an icon, omit it.

Return ONLY the JSON.
"""


def fmt_icon_list(icons: list, desc_field: str) -> str:
    out = []
    for ic in icons:
        out.append(f"  - id={ic['id']} kind={ic.get('kind','?')} "
                   f"desc=\"{ic.get(desc_field, '')}\"")
    return "\n".join(out)


# ---- visualisation ---------------------------------------------------

VARIANT_COLOR = {
    "simple_bbox": (50, 50, 220),     # red
    "detailed_bbox": (40, 165, 245),  # orange
    "simple_point": (50, 200, 50),    # green
    "detailed_point": (220, 80, 50),  # blue
}


def draw_results(img: np.ndarray, results: dict, W: int, H: int) -> np.ndarray:
    """Draw all 4 variants on one canvas with distinct colors."""
    out = img.copy()
    for variant, color in VARIANT_COLOR.items():
        info = results.get(variant, {})
        items = []
        if info.get("data") and "icons" in info["data"]:
            items = info["data"]["icons"]
        for it in items:
            if "bbox" in it and len(it["bbox"]) == 4:
                try:
                    x1, y1, x2, y2 = [int(v) for v in it["bbox"]]
                except: continue
                cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
                cv2.putText(out, it.get("id", ""), (x1 + 3, y1 + 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
            elif "point" in it and len(it["point"]) == 2:
                try:
                    x, y = [int(v) for v in it["point"]]
                except: continue
                cv2.circle(out, (x, y), 8, color, -1)
                cv2.circle(out, (x, y), 12, color, 2)
                cv2.putText(out, it.get("id", ""), (x + 14, y + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    # legend
    legend_y = 24
    for v, c in VARIANT_COLOR.items():
        cv2.putText(out, v, (10, legend_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2)
        legend_y += 22
    return out


def draw_single(img: np.ndarray, info: dict, color, title: str,
                 mode: str) -> np.ndarray:
    out = img.copy()
    items = []
    if info.get("data") and "icons" in info["data"]:
        items = info["data"]["icons"]
    for it in items:
        if mode == "bbox" and "bbox" in it and len(it["bbox"]) == 4:
            try:
                x1, y1, x2, y2 = [int(v) for v in it["bbox"]]
            except: continue
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            cv2.putText(out, it.get("id", ""), (x1 + 3, y1 + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        elif mode == "point" and "point" in it and len(it["point"]) == 2:
            try:
                x, y = [int(v) for v in it["point"]]
            except: continue
            cv2.circle(out, (x, y), 8, color, -1)
            cv2.circle(out, (x, y), 12, color, 2)
            cv2.putText(out, it.get("id", ""), (x + 14, y + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    cv2.putText(out, f"{title}  n={len(items)}", (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    return out


# ---- main ------------------------------------------------------------

def run_one(name: str, hash_: str) -> dict:
    """Stage 1 + Stage 2 for one image. All Stage-2 calls in parallel."""
    out_dir = OUT_BASE / name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resize ORIGINAL and ICONS_ONLY to div-16
    orig_in = REPO / "test_output" / "4img_inputs" / f"{hash_}.png"
    icons_only_in = (REPO / "crafter" / "editor" / "_runs" / name
                      / "extract" / "icons_only" / "cleaned.png")

    orig_resized = resize_to_div16(Image.open(orig_in).convert("RGB"))
    icons_resized = resize_to_div16(Image.open(icons_only_in).convert("RGB"))
    W, H = orig_resized.size
    Wi, Hi = icons_resized.size
    orig_path = out_dir / "original_resized.png"
    icons_path = out_dir / "icons_only_resized.png"
    orig_resized.save(orig_path)
    icons_resized.save(icons_path)
    print(f"\n=== {name} ===")
    print(f"  original (resized): {W}x{H}")
    print(f"  icons_only (resized): {Wi}x{Hi}")

    # ---- STAGE 1: describe icons from icons_only ----
    print("  STAGE 1: describing icons in icons_only ...")
    desc_info = call_vlm(icons_path, DESCRIBE_PROMPT)
    (out_dir / "stage1_describe.json").write_text(
        json.dumps(desc_info, indent=2, ensure_ascii=False))
    if not desc_info.get("data") or "icons" not in desc_info["data"]:
        print(f"  ✗ stage 1 failed: {desc_info.get('error', desc_info.get('content','')[:200])}")
        return {}
    icons = desc_info["data"]["icons"]
    print(f"  ✓ described {len(icons)} icons in {desc_info['elapsed']}s")

    # Save sample descriptions
    sample = "\n".join(
        f"  {ic['id']:5} {ic.get('kind','?'):20} simple=\"{ic.get('simple_desc','')[:40]}\"  "
        f"detailed=\"{ic.get('detailed_desc','')[:60]}\""
        for ic in icons[:6])
    print(f"  sample descriptions:\n{sample}")

    # ---- STAGE 2: 4 variants in parallel ----
    icon_list_simple = fmt_icon_list(icons, "simple_desc")
    icon_list_detailed = fmt_icon_list(icons, "detailed_desc")

    jobs = {
        "simple_bbox":   GROUND_BBOX_PROMPT.format(
            W=W, H=H, N=len(icons), ICON_LIST=icon_list_simple),
        "detailed_bbox": GROUND_BBOX_PROMPT.format(
            W=W, H=H, N=len(icons), ICON_LIST=icon_list_detailed),
        "simple_point":  GROUND_POINT_PROMPT.format(
            W=W, H=H, N=len(icons), ICON_LIST=icon_list_simple),
        "detailed_point": GROUND_POINT_PROMPT.format(
            W=W, H=H, N=len(icons), ICON_LIST=icon_list_detailed),
    }

    print(f"  STAGE 2: 4 grounding calls in parallel ...")
    results = {"_descriptions": icons}
    with ThreadPoolExecutor(max_workers=4) as ex:
        futures = {ex.submit(call_vlm, orig_path, prompt): variant
                    for variant, prompt in jobs.items()}
        for fut in as_completed(futures):
            v = futures[fut]
            info = fut.result()
            results[v] = info
            (out_dir / f"stage2_{v}.json").write_text(
                json.dumps(info, indent=2, ensure_ascii=False))
            n_found = (len(info.get("data", {}).get("icons", []) or [])
                        if info.get("data") else 0)
            print(f"    ✓ {v:18} status={info.get('status','-')} "
                  f"elapsed={info.get('elapsed','-')}s  found={n_found}/{len(icons)}")

    # ---- Visualisations ----
    cv_orig = cv2.cvtColor(np.array(orig_resized), cv2.COLOR_RGB2BGR)

    # Combined (all 4 colors)
    combined = draw_results(cv_orig, results, W, H)
    cv2.imwrite(str(out_dir / "combined_all_4_variants.png"), combined)

    # Single panels: 4 separate
    panels = {}
    for variant, mode in [("simple_bbox", "bbox"),
                          ("detailed_bbox", "bbox"),
                          ("simple_point", "point"),
                          ("detailed_point", "point")]:
        info = results.get(variant, {})
        ov = draw_single(cv_orig, info, VARIANT_COLOR[variant], variant, mode)
        cv2.imwrite(str(out_dir / f"single_{variant}.png"), ov)
        panels[variant] = ov

    # 2×2 grid
    gap = 10
    H_, W_ = panels["simple_bbox"].shape[:2]
    hgap = np.full((H_, gap, 3), 136, dtype=np.uint8)
    vgap = np.full((gap, 2 * W_ + gap, 3), 136, dtype=np.uint8)
    top = np.hstack([panels["simple_bbox"], hgap, panels["detailed_bbox"]])
    bot = np.hstack([panels["simple_point"], hgap, panels["detailed_point"]])
    grid = np.vstack([top, vgap, bot])
    cv2.imwrite(str(out_dir / "grid_2x2.png"), grid)
    print(f"  outputs: {out_dir}")
    return results


def main():
    for name, hash_ in IMAGES:
        try:
            run_one(name, hash_)
        except Exception as exc:
            import traceback; traceback.print_exc()
            print(f"  ERROR: {exc}")


if __name__ == "__main__":
    main()
