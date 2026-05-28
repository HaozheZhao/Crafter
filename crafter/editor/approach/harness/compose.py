"""Composition-phase building blocks:
   skeleton → inject raster icons → inject vector elements.

skeleton — single LLM call: model sees (original PNG, marked PNG,
   element list) and produces an SVG with <g id="RAS_NN"/> and
   <g id="VEC_NN"/> placeholders plus all panels / text / arrows /
   dashed borders / background colours. Multi-temperature best-of-N
   picked by quick_judge.

inject_raster — for every RAS_NN placeholder, splice in
   <image href="data:image/png;base64,..."> at the placement bbox.
   Position uses the SAM3-on-original placement bbox.

inject_vector — for every VEC_NN placeholder, single LLM call with
   (original_crop, kind, desc, bbox) → an SVG <g> fragment using only
   basic primitives (no <image>, no base64). Parallelised per vector.
"""
from __future__ import annotations
import os

import base64
import io
import logging
import re
import sys
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image

REPO = Path(os.environ.get("CRAFTER_HOME", str(Path(__file__).resolve().parents[2])))

from crafter.editor.approach.iter_svg_fix import call_model, extract_svg_from_response  # noqa: E402

logger = logging.getLogger("harness.compose")


POSTER_MODE_ADDENDUM = """

POSTER / INFOGRAPHIC MODE (style_analyzer detected this is a poster
or rich infographic, not a clean academic pipeline figure). These
layouts use arrows / leader lines / brackets / callout pointers
DECORATIVELY rather than to encode dataflow between modules. Apply
these rules strictly:

  • Do NOT recreate decorative arrows, leader lines, callout pointers,
    brackets, dotted/dashed connectors, or zoom-callout indicators
    unless they CLEARLY connect two named modules (input-arrow,
    output-arrow, A→B in a flow diagram). Clean omission scores
    HIGHER than a wrong arrow.
  • Section headers / banner titles are usually large stylized text
    blocks — render only if you can match font weight + colour
    plausibly; otherwise leave the placeholder area visually empty
    rather than guess.
  • Background colour bands (yellow/blue/grey strips behind sections)
    are part of the layout — preserve them as flat <rect> fills, but
    do NOT add per-cell shadows or gradients you cannot match.
  • Logos / institution marks come from the raster injector — do not
    try to redraw them as vector text.
"""


SKELETON_PROMPT = """=========================================================
PRIMARY OBJECTIVE — READ FIRST, INTERNALISE BEFORE READING ANY RULE BELOW

实现像素级别的复现 Image 1 (achieve pixel-level reproduction of Image 1).

This is the SINGLE most important goal. Every detailed rule that
follows is a tactic in service of this goal — the rules are NOT a
checklist to mechanically satisfy, they are guardrails that protect
against common failure modes. The rules are subordinate to the
PRIMARY OBJECTIVE: when in doubt, look at Image 1 and reproduce
what you actually see, even if a rule below would suggest a slight
deviation.

Concretely this means:
  • Every text content, font weight, font style, font color, font
    size, position must MATCH Image 1.
  • Every panel / rect fill colour must MATCH Image 1's panel colour.
  • Every arrow stroke colour, stroke width, dash pattern,
    arrowhead style must MATCH Image 1's arrow.
  • Every shape, line, border, separator that is visually distinct
    in Image 1 must appear in the SVG with matching style.

ICONS are the ONLY exception: replace each icon region with a
labelled placeholder per the rules below — they will be filled in
later by an automatic injector. NEVER try to redraw icon contents.
=========================================================

You are an expert SVG designer. Generate an SVG that EXACTLY
reproduces the academic figure in Image 1.

Image 1 = ORIGINAL figure (the ground truth you must replicate)
Image 2 = MARKED figure showing where pre-extracted elements live:
  - solid GREY boxes labelled RAS_NN  → raster-icon targets that will
    be filled in later by an automatic injector. EMIT each as a
    TRANSPARENT placeholder:
        <g id="RAS_NN"><rect x= y= width= height=
        fill="none" stroke="none"/></g>
    at the EXACT bbox shown. NEVER fill these with grey, never add a
    stroke, never draw any decoration inside the RAS_NN bbox.
  - dashed CYAN boxes labelled VEC_NN → vector-element targets that will
    be filled in later. EMIT each as <g id="VEC_NN"><rect x= y= width= height=
    fill="none" stroke="#00bcd4" stroke-width="1.5" stroke-dasharray="6 3"/>
    </g> at the EXACT bbox shown.

RAS_NN bbox — avoid REDUNDANT mid-layer drawing:
  RAS_NN slots will be filled with raster icons by the injector. Avoid
  the "灰底 + duplicate icon" bug, but keep legitimate overlays:

  DO NOT do these (they cause double-drawing):
    • A non-transparent <rect> covering the RAS_NN bbox — that becomes
      a grey/coloured halo under the raster icon.
    • A <path>/<circle>/<polygon> that REDRAWS the same icon shape
      that the raster icon already shows.

  OK to draw these (they are legitimate, on top of raster):
    • Small <text> labels next to or inside the RAS_NN bbox.
    • Small markers (e.g. histogram bars over a chart panel image).
    • Annotation arrows pointing into the RAS_NN bbox.
    • If Image 1 has a coloured panel CONTAINING the icon, you may
      draw the panel — just be aware the raster icon will render on
      top with its own (possibly white-ish) padding visible.

  Rule: never duplicate what the raster will render; do add what the
  raster doesn't show.

EVERYTHING ELSE you must draw yourself by looking at Image 1:
  - the figure background colour (full <rect> at viewBox 0,0) — but only
    in regions NOT covered by any RAS_NN slot
  - any panels / containers / coloured backgrounds — same constraint
  - dashed module borders
  - arrows / connectors

Pre-extracted element list (id → kind → short description → bbox in original coords):
{ELEMENT_LIST}

TEXT LIST (from PaddleOCR — use these as AUTHORITATIVE ground truth
for content and position; ALSO add any other text you observe in the
original that isn't in this list, especially:
  - tiny labels, math symbols, subscripts, single characters
  - structural section headers Paddle may have missed
Don't omit text just because Paddle didn't see it.):
{PADDLE_TEXTS}

CRITICAL REQUIREMENTS:
1. viewBox="0 0 {W} {H}" width="{W}" height="{H}"
2. Match EXACT colours from Image 1. Do not darken, desaturate, or fade.
3. Each RAS_NN and VEC_NN placeholder MUST appear EXACTLY once with the
   stated id; do not emit text containing "RAS" or "VEC" elsewhere.

TEXT RULES — STRICT (a wrong text size makes the output unusable):

** VISUAL TEXT INSPECTION (read this first) **
For EVERY <text> you emit, INSPECT Image 1 and copy these visual properties
from what you actually see — DO NOT default everything to neutral:
  - font-weight: "bold" or "700" if the strokes are clearly heavy in Image 1;
    otherwise omit (defaults to 400 normal).
  - font-style: "italic" if characters are slanted in Image 1; otherwise omit.
  - font-family CSS generic: pick ONE of {"sans-serif", "serif", "monospace"}
    based on the character shapes in Image 1. Default is "sans-serif". Use
    "serif" for figures with clearly serifed body fonts (Times-like). Use
    "monospace" for code blocks or terminal-style text only.
  - fill: when the text is clearly not near-black, copy the actual hex
    color from Image 1 (e.g., section-title text in a colored band, or
    callout text matching its panel hue). Use "#000" / dark default
    only when the original text is genuinely black/near-black.
A text whose font-weight/style/family/fill matches Image 1 scores
much higher on the text aspect than one rendered with neutral defaults,
even if the content is correct.

** POSITION + SIZE **
- For every entry in the TEXT LIST: the listed bbox gives an APPROXIMATE
  position (OCR is noisy on stylized text). Anchor the <text> at
  (bbox_x, bbox_y + 0.85 * bbox_h) (SVG y is baseline, not top), then
  micro-adjust if Image 1 visibly disagrees.
- ALSO add any text you see in the original that is NOT in the list,
  using your best position estimate. Coverage matters — never omit
  visible text.
- font-size: VISUALLY measure the character cap-height in Image 1 and
  set font-size accordingly. The OCR-derived font_size from the list
  is an UPPER BOUND, not a floor: shrink to fit when needed (down to
  ~9px min). Heading text is often visibly larger than OCR estimates.

** ENCODING **
- Subscripts/superscripts INSIDE a single <text> (Unicode), not separate
  <tspan>s. Use ₀₁₂₃₄₅₆₇₈₉ₜₙₘₖ and ⁰¹²³⁴⁵⁶⁷⁸⁹ for things like "F_{i,j}".
- Multi-line in the original → split into <tspan x="..." dy="1em"> rows.
- Two text bboxes that overlap (collision) must NOT both render at full
  size — shrink one or both.
- DO NOT use HTML-only named entities (`&nbsp;`, `&ndash;`, `&copy;`, etc.).
  SVG/XML only defines {&amp; &lt; &gt; &quot; &apos;}; everything else
  crashes the renderer. Use Unicode characters directly instead
  (regular space, ` `, `—`, `©`).

ARROW RULES — STRICT (oversized arrowheads ruin output):

** VISUAL ARROW INSPECTION (read this first) **
For EVERY arrow / connector you emit (`<line>` / `<path>` /
`<polyline>` with marker-end), INSPECT Image 1 and copy these visual
properties from what you actually see:
  - stroke colour: copy the actual line colour. Default #555555 only
    when the arrow is genuinely grey/black; when Image 1 shows a
    coloured connector (blue dataflow arrows, red emphasis, green
    return paths) emit `stroke="#xxxxxx"` matching what you see.
  - stroke-width: read the actual line thickness. Thin connectors
    are 1.0–1.2; main flow arrows are typically 1.5–2.0; emphasis
    arrows can be 2.5–3.0. Don't default everything to 1.5.
  - dash pattern: only mark dashed if Image 1 shows a clearly broken
    line; otherwise solid (no dasharray).
  - arrowhead shape: triangle (most common) vs filled-triangle vs
    open-V — the marker definition reflects what you see.
A coloured / correctly-thick arrow scores much higher on the arrow
aspect than a uniform grey 1.5px arrow even if the endpoints are
correct.

** STRICT GEOMETRY **
- markerWidth=8 markerHeight=6 maximum (NEVER bigger than this).
- Endpoints must land near (≤ 12px from) an actual element bbox on
  BOTH ends — never both endpoints in mid-air.
- Dashed: stroke-dasharray="8 5" when Image 1 shows broken lines.

VECTOR-STYLE RULES — STRICT (academic figures are FLAT):
- DO NOT use <feDropShadow>, <linearGradient>, <radialGradient>,
  <feGaussianBlur>, or any filter= attribute on structural shapes.
- Boxes are FLAT solid fills with a 1.5-2px stroke.
- No 3D effects, no gloss, no shadows on simple panels.

** VISUAL RECT / PANEL FIDELITY **
For EVERY <rect> that represents a panel, container, or coloured
background band, read its `fill` (and `stroke` if present) DIRECTLY
from Image 1 — do NOT approximate. Common targets:
  - Section bands (yellow / blue / grey strips behind groups of content)
  - Colored category boxes (light blue / pink / mint backgrounds in
    dataflow diagrams)
  - Title bars / header strips
  - Dashed module borders → match the dash pattern + stroke colour
For these rects:
  1. Pick the RGB you actually see at the panel center in Image 1.
  2. Emit `fill="#xxxxxx"` (use the hex you observed). NEVER reuse a
     stale/guessed colour from a similar-looking figure.
  3. If the panel has a thin border in Image 1, set `stroke="#yyyyyy"`
     `stroke-width="1.5"` (or 2 for emphasis). If no border is visible,
     omit stroke entirely — empty stroke is correct.
A rect whose fill matches Image 1 within ~5% RGB scores much higher on
both the color and style aspects than a default-coloured rect, even if
its position is correct.

DASHED CONTAINER RULES (dashed panels):
- stroke-width="2", stroke="#888888", stroke-dasharray="10 7"

BACKGROUND:
- If Image 1 has a coloured background, reproduce it as a full-size <rect>.

LAYOUT:
- Spread elements to fill the {W}x{H} canvas; do not compress to a corner.
- Right-most element should reach within ~50px of x={W}.
- Bottom-most element should reach within ~50px of y={H}.

OUTPUT: ONLY a complete SVG. Start with <svg, end with </svg>, no markdown.
"""


def _b64(path: Path, max_dim: int = 1500) -> str:
    img = Image.open(path).convert("RGB")
    if max(img.size) > max_dim:
        scale = max_dim / max(img.size)
        img = img.resize(
            (int(img.width * scale), int(img.height * scale)),
            Image.LANCZOS,
        )
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88)
    return base64.b64encode(buf.getvalue()).decode()


def _format_elements(labels: dict) -> str:
    lines = []
    for r in labels["raster"]:
        bb = r["bbox"]
        lines.append(
            f"  {r['label']}  RASTER  [{r['kind']}]  {r['simple_desc'][:50]}"
            f"   bbox=({bb[0]},{bb[1]},{bb[2]},{bb[3]})"
        )
    for v in labels["vector"]:
        bb = v["bbox"]
        lines.append(
            f"  {v['label']}  VECTOR  [{v['kind']}]  {v['simple_desc'][:50]}"
            f"   bbox=({bb[0]},{bb[1]},{bb[2]},{bb[3]})"
        )
    return "\n".join(lines)


def generate_skeleton(
    original_png: Path,
    marked_png: Path,
    labels: dict,
    w: int,
    h: int,
    paddle_texts: list[dict] | None = None,
    style_profile: dict | None = None,
    model: str = "openai/gpt-5.5",
    # Best-of-N at temps (0.20, 0.45). Paper §F-Editor implementation.
    temperatures: tuple[float, ...] = (0.20, 0.45),
) -> list[str]:
    """Return list of SVG candidates (one per temperature)."""
    from .checkers.paddle_text import format_for_prompt as _ptext_fmt
    paddle_block = (_ptext_fmt(paddle_texts)
                    if paddle_texts else "  (no Paddle text available)")
    # Font / style hints from style_analyzer (optional)
    fh_block = ""
    if style_profile and isinstance(style_profile.get("font_hints"), dict):
        fh = style_profile["font_hints"]
        parts = []
        if fh.get("title_style"):
            parts.append(f"  title_style: {fh['title_style']}")
        if fh.get("body_style"):
            parts.append(f"  body_style: {fh['body_style']}")
        if fh.get("body_font_family_guess"):
            parts.append(f"  body_font_family_guess: "
                         f"{fh['body_font_family_guess']}")
        if fh.get("math_present"):
            parts.append("  math_present: true — render math symbols as "
                         "unicode (italic for variables/Greek)")
        if fh.get("cursive_present"):
            parts.append("  cursive_present: true — match cursive style "
                         "for any handwritten labels")
        if fh.get("emphasis_uses"):
            parts.append("  emphasis_uses: " + ", ".join(fh["emphasis_uses"]))
        if parts:
            fh_block = "\nFONT HINTS (from style analyzer):\n" + \
                "\n".join(parts) + "\n"
    # str.format would treat curly braces inside Paddle text (e.g. "F_{i,j}")
    # as placeholders → KeyError. Use plain replace.
    prompt = (SKELETON_PROMPT
              .replace("{ELEMENT_LIST}", _format_elements(labels))
              .replace("{PADDLE_TEXTS}", paddle_block + fh_block)
              .replace("{W}", str(w))
              .replace("{H}", str(h)))

    orig_b64 = _b64(original_png)
    mark_b64 = _b64(marked_png)
    msgs = [{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url":
                {"url": f"data:image/jpeg;base64,{orig_b64}"}},
            {"type": "image_url", "image_url":
                {"url": f"data:image/jpeg;base64,{mark_b64}"}},
            {"type": "text", "text": prompt},
        ],
    }]

    def _run(t):
        try:
            resp = call_model(msgs, model=model, max_tokens=50000,
                              label="stage6_skeleton",
                              temperature=t)
            return extract_svg_from_response(resp)
        except Exception as e:
            logger.warning("skeleton temp=%.2f failed: %s", t, e)
            return None

    out = []
    with ThreadPoolExecutor(max_workers=len(temperatures)) as ex:
        futs = {ex.submit(_run, t): t for t in temperatures}
        for f in as_completed(futs):
            svg = f.result()
            if svg:
                out.append(svg)
    return out


def inject_raster(svg: str, raster_meta: list[dict],
                  crops_dir: Path,
                  canvas_w: float | None = None,
                  canvas_h: float | None = None) -> str:
    """Replace each <g id="RAS_NN"> placeholder with <image href=...>.

    After injection, scrub vector primitives that overlap any <image>
    bbox (defensive cleanup against double-drawing — see
    raster_cleanup.py)."""
    n_done, n_skipped = 0, 0
    for r in raster_meta:
        crop_path = r.get("cleaned_crop_path")
        if not crop_path:
            n_skipped += 1
            continue
        full = crops_dir.parent / crop_path if not Path(crop_path).is_absolute() \
            else Path(crop_path)
        # crop path is relative to runs/imgN/ directory
        if not full.exists():
            full = crops_dir / Path(crop_path).name
        if not full.exists():
            n_skipped += 1
            continue
        b64 = base64.b64encode(full.read_bytes()).decode()
        x1, y1, x2, y2 = r["bbox"]
        w, h = x2 - x1, y2 - y1
        repl = (
            f'<image id="icon_{r["label"]}" x="{x1}" y="{y1}" '
            f'width="{w}" height="{h}" '
            f'href="data:image/png;base64,{b64}" '
            f'preserveAspectRatio="xMidYMid meet"/>'
        )
        pat = re.compile(
            rf'<g[^>]*\bid=["\']?{re.escape(r["label"])}["\']?[^>]*>'
            rf'[\s\S]*?</g>',
            re.IGNORECASE,
        )
        new_svg, n = pat.subn(repl, svg, count=1)
        if n == 0:
            # placeholder missing — append before </svg>
            new_svg = svg.replace("</svg>", f"  {repl}\n</svg>")
        svg = new_svg
        n_done += 1
    logger.info("  raster injected: %d done, %d skipped", n_done, n_skipped)
    # Defensive cleanup is disabled by default — empirically (the
    # post-hoc test on 8 imgs) it dropped mean score 7.86 → 6.97
    # because it removed vector primitives that were covering up
    # poor-quality raster icons. Re-enable only after Step A raster
    # quality is fixed (gpt-image-2 prompt + alpha-background fix).
    # The SKELETON_PROMPT + REFINE_PROMPT changes still prevent NEW
    # double-drawing in fresh runs — this auto-cleanup was the
    # belt-and-suspenders that turned out to be over-aggressive.
    return svg


VECTOR_FRAG_PROMPT = """You see one cropped region of an academic figure
showing an icon. Your job: produce ONE SVG `<g id="{LABEL}">…</g>`
fragment that reproduces this icon using basic SVG primitives.

Reference info:
  kind:        {KIND}
  description: {DESC}
  bbox in original coords: ({X1},{Y1}) to ({X2},{Y2})  → width={W}, height={H}

CONSTRAINTS:
- All coordinates ABSOLUTE in the original 1376x768 canvas (use the
  bbox above; place every shape inside it).
- Use only <rect>, <circle>, <ellipse>, <line>, <polygon>, <path>,
  <text>; optionally <defs>+<linearGradient>/<radialGradient>.
- NO <image>. NO base64. NO external href.
- Be visually rich: ≥ 5 sub-elements for shapes wider than 20px;
  match colours observed in the crop.
- For text labels, use font-family="DejaVu Sans".

Output ONLY the `<g id="{LABEL}">…</g>` block. No markdown, no prose.
"""

_G_OPEN = re.compile(r'<g\s+id="VEC_\d+"[^>]*>')


def _extract_g_block(text: str, label: str) -> str | None:
    """Depth-balanced extractor for a `<g id="LABEL">…</g>` block."""
    text = text.strip()
    m = re.search(r"```(?:xml|svg)?\s*\n?(.*?)```", text, re.DOTALL)
    if m:
        text = m.group(1).strip()
    pat = re.compile(rf'<g\s+id="{re.escape(label)}"[^>]*>')
    om = pat.search(text)
    if not om:
        return None
    body = text[om.start():]
    depth, i = 0, 0
    while i < len(body):
        if body[i:i + 2] == "<g" and (i + 2 < len(body)) and \
                body[i + 2] in (" ", "\t", "\n", ">"):
            depth += 1
            j = body.find(">", i)
            if j < 0:
                break
            i = j + 1
            continue
        if body[i:i + 4] == "</g>":
            depth -= 1
            i += 4
            if depth == 0:
                return body[:i]
            continue
        i += 1
    # Truncated — best effort: trim to last self-close + close depth
    last = max(body.rfind("/>"), body.rfind("</text>"),
               body.rfind("</rect>"), body.rfind("</path>"))
    if last < 0:
        return None
    salv = body[:last + 2]
    d2 = 0
    j = 0
    while j < len(salv):
        if salv[j:j + 2] == "<g" and (j + 2 < len(salv)) and \
                salv[j + 2] in (" ", "\t", "\n", ">"):
            d2 += 1
            k = salv.find(">", j)
            if k < 0:
                break
            j = k + 1
            continue
        if salv[j:j + 4] == "</g>":
            d2 -= 1
            j += 4
            continue
        j += 1
    if d2 < 0:
        return None
    return salv + ("</g>" * d2)


def _crop_b64(original_png: Path, bbox, pad=6) -> str:
    img = Image.open(original_png).convert("RGB")
    W, H = img.size
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
    x2, y2 = min(W, x2 + pad), min(H, y2 + pad)
    crop = img.crop((x1, y1, x2, y2))
    buf = io.BytesIO()
    crop.save(buf, format="JPEG", quality=88)
    return base64.b64encode(buf.getvalue()).decode()


def _gen_one_vector_fragment(
    label: str, vec_meta: dict, original_png: Path, model: str,
) -> tuple[str, str | None]:
    bb = vec_meta["bbox"]
    desc = vec_meta.get("simple_desc", "") + ". " + \
        vec_meta.get("detailed_desc", "")
    crop_b64 = _crop_b64(original_png, bb)
    prompt = VECTOR_FRAG_PROMPT.format(
        LABEL=label, KIND=vec_meta.get("kind", ""), DESC=desc[:200],
        X1=bb[0], Y1=bb[1], X2=bb[2], Y2=bb[3],
        W=bb[2] - bb[0], H=bb[3] - bb[1],
    )
    msgs = [{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url":
                {"url": f"data:image/jpeg;base64,{crop_b64}"}},
            {"type": "text", "text": prompt},
        ],
    }]
    try:
        resp = call_model(msgs, model=model, max_tokens=12000,
                          label="stage6_inject_vector")
    except Exception as e:
        return label, None
    return label, _extract_g_block(resp, label)


def inject_vector(
    svg: str, vector_meta: list[dict], original_png: Path,
    model: str = "openai/gpt-5.5", parallel: int = 4,
) -> tuple[str, dict]:
    """Replace each <g id="VEC_NN"> placeholder with an LLM-generated <g>."""
    if not vector_meta:
        return svg, {"n_total": 0, "n_done": 0, "n_failed": 0}

    fragments: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=parallel) as ex:
        futs = {
            ex.submit(_gen_one_vector_fragment,
                      v["label"], v, original_png, model): v["label"]
            for v in vector_meta
        }
        for f in as_completed(futs):
            label, frag = f.result()
            if frag:
                fragments[label] = frag

    n_done = 0
    for v in vector_meta:
        label = v["label"]
        frag = fragments.get(label)
        pat = re.compile(
            rf'<g[^>]*\bid=["\']?{re.escape(label)}["\']?[^>]*>'
            rf'[\s\S]*?</g>',
            re.IGNORECASE,
        )
        if frag:
            new_svg, n = pat.subn(frag, svg, count=1)
            if n == 0:
                new_svg = svg.replace("</svg>", f"  {frag}\n</svg>")
            svg = new_svg
            n_done += 1
        # If frag is None, leave the placeholder rect (cyan dashed
        # outline visible — at least preserves layout signal).

    info = {
        "n_total": len(vector_meta),
        "n_done": n_done,
        "n_failed": len(vector_meta) - n_done,
    }
    logger.info("  vector injected: %d/%d (%d failed)",
                info["n_done"], info["n_total"], info["n_failed"])
    return svg, info


def render_to_png(svg_path: Path, png_path: Path, w: int, h: int) -> bool:
    """Render via cairosvg / firefox helper from iter_svg_fix."""
    from crafter.editor.approach.iter_svg_fix import render_svg
    return render_svg(str(svg_path), str(png_path), w, h)
