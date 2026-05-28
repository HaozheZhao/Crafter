"""PaddleOCR-based text extractor.

Replaces an LLM-driven text reader. PaddleOCR PP-StructureV3 (English)
returns per-text-region polygon, recognized text, confidence, and
orientation angle directly from pixels — no model hallucination.
Deterministic style measurement on top:

  * font_color  = mode/k-means dominant of the dark pixels in the bbox
  * font_size   = bbox height (in px = SVG units; both spaces use the
                  same pixel coordinate system)
  * font_weight = stroke-density heuristic on the binarised crop

Concretely this gives us:
  - text fidelity (no LLM re-typing drift) — OCR reads pixels
  - consistent sizing across logical text classes — measured, not guessed
  - no duplicate texts — OCR returns one region per text

PaddleOCR loads lazily and is held as a module-level singleton because
the first .predict() call warms up the model (~50s). enable_mkldnn=False
avoids an oneDNN runtime error seen on some setups.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

from crafter.editor.raster_to_svg.schema import StyledText, TextResult

if TYPE_CHECKING:
    from paddleocr import PaddleOCR  # type: ignore

logger = logging.getLogger(__name__)

_OCR_SINGLETON: "PaddleOCR | None" = None


def _get_ocr() -> "PaddleOCR":
    """Lazy singleton — first call ~50s, subsequent ~3-5s/image."""
    global _OCR_SINGLETON
    if _OCR_SINGLETON is None:
        from paddleocr import PaddleOCR
        logger.info("Initialising PaddleOCR (first call ~50s)...")
        _OCR_SINGLETON = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=True,
            lang="en",
            enable_mkldnn=False,  # required: oneDNN crashes on some setups
        )
        logger.info("PaddleOCR ready.")
    return _OCR_SINGLETON


class PaddleTextExtractor:
    """Drop-in replacement for TextStyleAnalyzer.analyze().

    Same call surface so the pipeline can swap implementations behind an env
    var without touching other stages.
    """

    def __init__(self, *args, **kwargs):
        # Match TextStyleAnalyzer __init__ signature; we don't need the router
        # but accept and ignore the args so the pipeline can construct us
        # with the same call.
        self._args = args
        self._kwargs = kwargs

    def analyze(
        self,
        image_path: str,
        image_size: tuple[int, int],
        output_dir: str,
    ) -> TextResult:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        ocr = _get_ocr()
        try:
            results = ocr.predict(image_path)
        except Exception as e:
            logger.warning("PaddleOCR predict failed on %s: %s", image_path, e)
            return TextResult()

        if not results:
            logger.warning("PaddleOCR returned no results for %s", image_path)
            return TextResult()

        page = results[0]
        # PaddleOCR returns either a wrapper with .json or a dict directly
        data = page.json if hasattr(page, "json") else page
        if isinstance(data, dict) and "res" in data:
            data = data["res"]

        rec_texts = data.get("rec_texts") or []
        rec_scores = data.get("rec_scores") or []
        rec_polys = data.get("rec_polys") or []
        rec_boxes = data.get("rec_boxes") or []
        angles = data.get("textline_orientation_angles") or []

        if not rec_texts:
            return TextResult()

        try:
            img = np.array(Image.open(image_path).convert("RGB"))
        except Exception as e:
            logger.warning("Cannot read %s for color sampling: %s", image_path, e)
            img = None

        styled: list[StyledText] = []
        for i, text in enumerate(rec_texts):
            text = (text or "").strip()
            if not text:
                continue
            score = float(rec_scores[i]) if i < len(rec_scores) else 0.0
            if score < 0.5:
                continue  # discard low-confidence noise

            box = _coerce_bbox(rec_boxes, rec_polys, i)
            if box is None:
                continue
            x1, y1, x2, y2 = box
            w = x2 - x1
            h = y2 - y1
            if w < 4 or h < 4:
                continue

            # Pixel measurements
            font_color = "#000000"
            font_weight = "normal"
            if img is not None:
                crop = img[max(0, y1):min(img.shape[0], y2),
                           max(0, x1):min(img.shape[1], x2)]
                if crop.size > 0:
                    font_color = _dominant_text_color(crop)
                    font_weight = _estimate_weight(crop)

            # Vertical text detector: PaddleOCR's axis-aligned rec_box
            # for a rotated label is tall-and-narrow (h >> w). Using the
            # bbox height as the glyph height then produces a monstrous
            # font-size (for "Concatenation" h=182 → font-size=64) and
            # catastrophically overflows the figure. When the bbox is
            # clearly rotated, use the SHORT edge for font size.
            if h > w * 1.5:
                font_size = _bbox_height_to_pt(w)
            else:
                font_size = _bbox_height_to_pt(h)

            styled.append(StyledText(
                id=f"text_{i:03d}",
                content=text,
                bbox=[int(x1), int(y1), int(x2), int(y2)],
                font_size=font_size,
                font_weight=font_weight,
                font_color=font_color,
                font_family="DejaVu Sans",
                alignment="left",
                is_equation=_looks_like_equation(text),
                parent_box_id="",
            ))

        logger.info("PaddleOCR extracted %d text blocks (filtered to >=0.5 conf).",
                    len(styled))

        # Persist for debugging / downstream consumers
        try:
            import json
            out_json = out / "paddle_ocr.json"
            out_json.write_text(json.dumps(
                [{"id": t.id, "content": t.content, "bbox": t.bbox,
                  "font_size": t.font_size, "font_weight": t.font_weight,
                  "font_color": t.font_color} for t in styled],
                indent=2,
            ), encoding="utf-8")
        except Exception:
            pass

        return TextResult(texts=styled)


# ---------------------------------------------------------------------------
# Pixel measurements
# ---------------------------------------------------------------------------

def _coerce_bbox(rec_boxes, rec_polys, i: int) -> tuple[int, int, int, int] | None:
    """Return (x1, y1, x2, y2) for the i-th text region.

    PaddleOCR sometimes returns rec_boxes as a numpy array; fall back to
    deriving the axis-aligned bbox from rec_polys (the rotated quadrilateral).
    """
    if i < len(rec_boxes):
        b = rec_boxes[i]
        try:
            x1, y1, x2, y2 = (int(b[0]), int(b[1]), int(b[2]), int(b[3]))
            return x1, y1, x2, y2
        except Exception:
            pass
    if i < len(rec_polys):
        poly = rec_polys[i]
        try:
            xs = [int(p[0]) for p in poly]
            ys = [int(p[1]) for p in poly]
            return min(xs), min(ys), max(xs), max(ys)
        except Exception:
            return None
    return None


def _dominant_text_color(crop: np.ndarray) -> str:
    """Pick the darkest cluster colour in the crop — that's the ink colour for
    typical dark-on-light text. For light-on-dark text, this picks white-ish."""
    pixels = crop.reshape(-1, 3).astype(np.int32)
    if pixels.shape[0] == 0:
        return "#000000"
    brightness = pixels.sum(axis=1)
    # Take the darkest 25% (or lightest if mean is dark — handle inverse)
    median_brightness = np.median(brightness)
    if median_brightness > 192 * 3:  # mostly light bg
        ink = pixels[brightness <= np.percentile(brightness, 25)]
    else:
        ink = pixels[brightness >= np.percentile(brightness, 75)]
    if ink.shape[0] == 0:
        ink = pixels
    r, g, b = np.median(ink, axis=0).astype(int)
    return f"#{int(r):02x}{int(g):02x}{int(b):02x}"


def _estimate_weight(crop: np.ndarray) -> str:
    """Stroke-density heuristic: the fraction of ink pixels in the text crop.

    Bold text saturates around 0.32–0.45; regular text 0.18–0.28. The
    threshold sits at 0.32 so it clears roman-weight body text while
    catching genuine bold.
    """
    gray = crop.mean(axis=2)
    median = float(np.median(gray))
    # Dark text on light bg vs inverse
    if median > 128:
        ink_mask = gray < median - 30
    else:
        ink_mask = gray > median + 30
    ratio = float(ink_mask.mean())
    return "bold" if ratio > 0.32 else "normal"


def _bbox_height_to_pt(h: int) -> int:
    """PaddleOCR bbox height is the full tight bounding box of the glyphs
    (ascender-to-descender), not the cap height. Empirically the matching
    SVG font-size is ~0.85×h for DejaVu Sans at our render resolution.
    Clamp to [8, 64]."""
    px = int(round(h * 0.85))
    return max(8, min(64, px))


def _looks_like_equation(text: str) -> bool:
    """Quick heuristic — skipping a real classifier here. Detect math-y chars."""
    math_chars = set("∑∏√∫∞±×÷≤≥≠≈→←↔αβγδεζηθλμπρστφψω")
    if any(c in math_chars for c in text):
        return True
    if text.count("=") >= 1 and any(c.isalpha() for c in text):
        if any(c.isdigit() for c in text):
            return True
    return False
