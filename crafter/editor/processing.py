"""Processing phase — caption, ground, classify per-element assets.

Paper §3.3 \\Editor: a processing phase captions, grounds, and
classifies each element. This phase is pure perception scaffolding —
no harness loop, no validate-revise iteration.

Three sub-stages:
  caption    — VLM produces referring expressions for each element on
               the cleaned canvas
  ground     — SAM3 grounds each referring expression on the original
               raster, then a broad-class SAM3 pass supplements
               same-class repeats the caption-driven grounding cannot
               disambiguate (paper §F.7 "AF-style supplemental SAM3")
  classify   — per-element vector-vs-raster classifier
  extract    — crop each grounded element from the cleaned canvas;
               a hallucination filter discards blank, mismatched, or
               text-only crops
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from .config import Config, api_endpoint, api_key, sam3_url

logger = logging.getLogger("editor_v2.processing")

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "crafter" / "editor" / "approach"))


@dataclass
class ProcessingResult:
    img_id: str
    pairs_json: Path     # element inventory consumed by composition phase
    crops_dir: Path         # extracted per-element rasters (transparent PNG)
    n_total_captions: int
    n_raster_ok: int
    n_vector_descriptors: int
    n_af_supplemental: int


# -------------------------------------------------------------------
# Sub-stage: captioning
# -------------------------------------------------------------------
def _caption_elements(img_id: str, cleaned_png: Path,
                       config: Config) -> Path:
    """VLM-driven element captioning. Returns path to
    stage1_describe.json which lists icons with referring expressions.
    """
    from crafter.editor.approach.test_text_referring_grounding import (
        call_vlm, DESCRIBE_PROMPT, resize_to_div16,
    )
    from PIL import Image as PILImage

    # Write captions next to the run's per-sample outputs (out_dir of the
    # Editor pipeline), not inside the repo.
    case_root = Path(cleaned_png).resolve().parent.parent.parent
    out_dir = case_root / "processing" / "caption"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "stage1_describe.json"
    if out.exists():
        try:
            data = json.loads(out.read_text())
            if data.get("data", {}).get("icons"):
                logger.info("[processing] caption: cached")
                return out
        except Exception:
            pass

    logger.info("[processing] caption ▸ %s", config.processing.caption_model)
    resized = resize_to_div16(PILImage.open(cleaned_png).convert("RGB"))
    resized_path = out_dir / "icons_only_resized.png"
    resized.save(resized_path)
    info = call_vlm(resized_path, DESCRIBE_PROMPT,
                    label="processing_caption")
    out.write_text(json.dumps(info, indent=2, ensure_ascii=False))
    return out


# -------------------------------------------------------------------
# Sub-stage: grounding (caption-driven + AF-supplemental)
# -------------------------------------------------------------------
def _ground_elements(img_id: str, original_png: Path, cleaned_png: Path,
                      captions_json: Path, config: Config) -> tuple[Path, int]:
    """Run caption-driven SAM3 grounding, then AF-style supplemental
    broad-class grounding. Returns (pairs.json, n_supplemental_added).
    """
    from crafter.editor.raster_to_svg.tools.sam3_client import SAM3Client
    from crafter.editor.approach.af_supplemental_detect import af_detect_and_merge

    case_root = Path(cleaned_png).resolve().parent.parent.parent
    out_dir = case_root / "processing" / "grounding"
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs = out_dir / "pairs.json"

    if not pairs.exists():
        logger.info("[processing] ground ▸ caption-driven SAM3")
        # Pass a literal sub-name as --imgs so the script's nested
        # `out_root/<img>/pairs.json` lands at our `out_dir/pairs.json`.
        cmd = [
            sys.executable,
            str(_REPO / "crafter/editor/caption_sam3_referring/"
                "build_referring_pairs.py"),
            "--imgs", out_dir.name,
            "--out", str(out_dir.parent),
            "--sam3-url", sam3_url(),
            "--original", str(Path(original_png).resolve()),
            "--cleaned", str(Path(cleaned_png).resolve()),
            "--captions", str(Path(captions_json).resolve()),
        ]
        env = os.environ.copy()
        env.setdefault("SAM3_SERVER_URL", sam3_url())
        env.setdefault("API_KEY", api_key())
        proc = subprocess.run(cmd, env=env, cwd=str(_REPO))
        if proc.returncode != 0 or not pairs.exists():
            raise RuntimeError(
                f"[processing] caption-driven SAM3 grounding failed "
                f"for {img_id}: see {out_dir}/"
            )

    logger.info("[processing] ground ▸ AF-style supplemental (broad-class)")
    client = SAM3Client(server_url=sam3_url())
    log = af_detect_and_merge(
        cleaned_png, pairs, client,
        out_log_json=out_dir / "af_supplemental_log.json",
        captions_json=captions_json,
        prompt_list=list(config.processing.af_supplemental_prompts),
        min_score=config.processing.af_supplemental_min_score,
        iou_dedup=config.processing.af_supplemental_iou_dedup,
        max_supplemental=config.processing.af_supplemental_max_total,
    )
    n_added = log.get("n_added", 0)
    if n_added:
        logger.info("[processing]   +%d supplemental boxes", n_added)
    return pairs, n_added


# -------------------------------------------------------------------
# Sub-stage: classify + extract (raster crop with alpha + hallucination filter)
# -------------------------------------------------------------------
def _classify_and_extract(img_id: str, original_png: Path,
                            cleaned_png: Path,
                            captions_json: Path, pairs_json: Path,
                            config: Config) -> Path:
    """Run per-element vector/raster classifier (one VLM call per
    image), then crop+alpha-strip each raster element from the
    cleaned canvas via gpt-image-2 with a hallucination guard.

    Writes pairs.json + crops/ under <out_dir>/<img_id>/processing/.
    """
    # Derive run dir from cleaned_png: <out_dir>/<img_id>/extract/icons_only/cleaned.png
    case_root = Path(cleaned_png).resolve().parent.parent.parent
    approach_runs = case_root / "processing"
    approach_runs.mkdir(parents=True, exist_ok=True)

    # ---- classify ----
    vc_out = approach_runs / "vector_codeable.json"
    if not vc_out.exists():
        logger.info("[processing] classify ▸ vector vs raster (VLM)")
        # --imgs is a label for the subprocess output dir name; pass
        # approach_runs.name so out_root/<label>/ matches approach_runs.
        cmd = [sys.executable,
               str(_REPO / "crafter/editor/approach/classify_vector_codeable.py"),
               "--imgs", approach_runs.name,
               "--captions", str(Path(captions_json).resolve()),
               "--cleaned", str(Path(cleaned_png).resolve()),
               "--out-root", str(approach_runs.parent)]
        env = os.environ.copy()
        env.setdefault("API_KEY", api_key())
        proc = subprocess.run(cmd, env=env, cwd=str(_REPO))
        if proc.returncode != 0 or not vc_out.exists():
            raise RuntimeError(
                f"[processing] classify failed for {img_id}"
            )

    # ---- extract (crop + alpha + hallucination filter) ----
    pv3 = approach_runs / "pairs.json"
    if not pv3.exists():
        logger.info("[processing] extract ▸ gpt-image-2 + hallucination guard")
        # agent_design.json sits beside cleaned.png inside the extraction
        # phase output dir; pass it so logo-bypass logic in extract works.
        agent_design = Path(cleaned_png).resolve().parent.parent / "agent_design.json"
        cmd = [sys.executable,
               str(_REPO / "crafter/editor/approach/extract_icons.py"),
               "--imgs", approach_runs.name,
               "--sam3-url", sam3_url(),
               "--out", str(approach_runs.parent),
               "--original", str(Path(original_png).resolve()),
               "--cleaned", str(Path(cleaned_png).resolve()),
               "--captions", str(Path(captions_json).resolve()),
               "--placement-pairs", str(Path(pairs_json).resolve()),
               "--vector-codeable", str(vc_out.resolve()),
               "--agent-design", str(agent_design)]
        env = os.environ.copy()
        env.setdefault("SAM3_SERVER_URL", sam3_url())
        env.setdefault("API_KEY", api_key())
        proc = subprocess.run(cmd, env=env, cwd=str(_REPO))
        if proc.returncode != 0 or not pv3.exists():
            raise RuntimeError(
                f"[processing] extract failed for {img_id}"
            )
    return pv3


# -------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------
def run(img_id: str, original_png: Path, cleaned_png: Path,
        out_root: Path, config: Config) -> ProcessingResult:
    """Run the processing phase end-to-end on one image."""
    logger.info("[processing] %s", img_id)
    captions = _caption_elements(img_id, cleaned_png, config)
    pairs, n_supp = _ground_elements(img_id, original_png, cleaned_png,
                                       captions, config)
    pv3 = _classify_and_extract(img_id, original_png, cleaned_png,
                                  captions, pairs, config)

    # Stats for reporting
    pv3_data = json.loads(pv3.read_text())
    return ProcessingResult(
        img_id=img_id,
        pairs_json=pv3,
        crops_dir=pv3.parent / "crops",
        n_total_captions=pv3_data.get("n_total_captions", 0),
        n_raster_ok=pv3_data.get("n_raster_ok", 0),
        n_vector_descriptors=len(pv3_data.get("vector_descriptors", [])),
        n_af_supplemental=n_supp,
    )
