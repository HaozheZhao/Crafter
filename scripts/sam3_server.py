"""SAM3 HTTP server using the official ``sam3`` package.

Endpoints
---------

``/segment``       — text-prompted detection (returns boxes).
``/segment_bbox``  — bbox-prompted segmentation (returns per-box masks).
``/health``        — health probe.

Setup
-----

1. Install the official package somewhere importable, e.g.

    git clone https://github.com/facebookresearch/sam3
    pip install -e ./sam3
    pip install timm ftfy iopath portalocker flask torch numpy Pillow

2. Run this script on a machine with a CUDA-capable GPU:

    python scripts/sam3_server.py --port 8765 --host 0.0.0.0

3. Point Crafter at the server:

    export SAM3_SERVER_URL="http://<host>:8765"

If the official ``sam3`` package lives in a non-standard location, set
``SAM3_PACKAGE_PATH`` to its directory before launching this script.
"""

import argparse
import base64
import io
import json
import logging
import os
import sys

import numpy as np
import torch
from flask import Flask, jsonify, request
from PIL import Image

if "SAM3_PACKAGE_PATH" in os.environ:
    sys.path.insert(0, os.environ["SAM3_PACKAGE_PATH"])

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
sam3_model = None
sam3_processor = None
device = None


def _force_fp32_recursive(obj, depth=0):
    """Recursively convert all fp16 / bf16 tensors to fp32."""
    if depth > 5:
        return
    if isinstance(obj, torch.Tensor) and obj.is_floating_point() and obj.dtype != torch.float32:
        obj.data = obj.data.float()
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, torch.Tensor) and v.is_floating_point() and v.dtype != torch.float32:
                obj[k] = v.float()
            elif isinstance(v, (dict, list)):
                _force_fp32_recursive(v, depth + 1)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            if isinstance(v, torch.Tensor) and v.is_floating_point() and v.dtype != torch.float32:
                obj[i] = v.float()
            elif isinstance(v, (dict, list)):
                _force_fp32_recursive(v, depth + 1)


def load_model():
    """Build SAM3, force float32 everywhere, and patch the processor to disable autocast."""
    global sam3_model, sam3_processor, device
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading SAM3 on {device}")

    model = build_sam3_image_model(
        device=device,
        eval_mode=True,
        load_from_HF=True,
        enable_segmentation=True,
    )

    # The published checkpoint mixes fp16 / bf16 weights, which trips dtype
    # mismatches on some non-NVIDIA GPUs. Force everything to fp32.
    for param in model.parameters():
        param.data = param.data.float()
    for _, buf in model.named_buffers():
        if buf.is_floating_point():
            buf.data = buf.data.float()
    sam3_model = model
    if hasattr(torch, "set_autocast_enabled"):
        torch.set_autocast_enabled(False)

    sam3_processor = Sam3Processor(
        model=model,
        resolution=1008,
        device=device,
        confidence_threshold=0.3,
    )

    # Disable autocast on every processor entry point.
    def _wrap(fn):
        @torch.inference_mode()
        def inner(*args, **kwargs):
            with torch.amp.autocast("cuda", enabled=False):
                state = fn(*args, **kwargs)
            _force_fp32_recursive(state)
            return state
        return inner

    sam3_processor.set_image          = _wrap(sam3_processor.set_image)
    sam3_processor._forward_grounding = _wrap(sam3_processor._forward_grounding)
    sam3_processor.add_geometric_prompt = _wrap(sam3_processor.add_geometric_prompt)
    sam3_processor.set_text_prompt    = _wrap(sam3_processor.set_text_prompt)

    if device == "cuda":
        logger.info(f"SAM3 loaded; VRAM: {torch.cuda.memory_allocated() / 1e9:.1f} GB")
    else:
        logger.info("SAM3 loaded on CPU")


def _load_image() -> Image.Image:
    if "image" in request.files:
        return Image.open(request.files["image"]).convert("RGB")
    if "image_b64" in request.form:
        raw = base64.b64decode(request.form["image_b64"])
        return Image.open(io.BytesIO(raw)).convert("RGB")
    return None


@app.route("/segment", methods=["POST"])
def segment():
    """Text-prompted object detection. Returns a list of bounding boxes."""
    if sam3_processor is None:
        return jsonify({"error": "Model not loaded"}), 500
    img = _load_image()
    if img is None:
        return jsonify({"error": "No image"}), 400

    prompts = [p.strip() for p in request.form.get("prompts", "icon").split(",") if p.strip()]
    sam3_processor.confidence_threshold = float(request.form.get("min_score", "0.3"))
    w, h = img.size

    all_boxes = []
    for prompt in prompts:
        try:
            state = sam3_processor.set_image(img)
            state = sam3_processor.set_text_prompt(prompt, state)
            if state.get("boxes") is not None and len(state["boxes"]) > 0:
                boxes  = state["boxes"].cpu().numpy()
                scores = state["scores"].cpu().numpy()
                for idx in range(len(boxes)):
                    x1, y1, x2, y2 = boxes[idx]
                    all_boxes.append({
                        "x1": int(x1), "y1": int(y1),
                        "x2": int(x2), "y2": int(y2),
                        "score": round(float(scores[idx]), 3),
                        "prompt": prompt,
                    })
                logger.info(f"  '{prompt}': {len(boxes)} detections")
            else:
                logger.info(f"  '{prompt}': 0 detections")
        except Exception as e:
            logger.warning(f"Prompt {prompt!r} failed: {e}")

    return jsonify({"boxes": all_boxes, "image_size": [w, h]})


def _iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    a1 = (a[2] - a[0]) * (a[3] - a[1])
    a2 = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (a1 + a2 - inter) if (a1 + a2 - inter) > 0 else 0.0


@app.route("/segment_bbox", methods=["POST"])
def segment_bbox():
    """Bbox-prompted segmentation.

    Body
    ----
    ``image``     — image file (or ``image_b64`` form field).
    ``boxes``     — JSON array of ``[x1, y1, x2, y2]`` pixel boxes (positive).
    ``neg_boxes`` — JSON array of negative boxes (optional).

    Returns per-box ``{tight_box, det_box, mask_b64, iou_score, ...}``.
    """
    if sam3_processor is None:
        return jsonify({"error": "Model not loaded"}), 500
    img = _load_image()
    if img is None:
        return jsonify({"error": "No image"}), 400

    w, h = img.size
    try:
        input_boxes = json.loads(request.form.get("boxes", "[]")) or []
        neg_boxes   = json.loads(request.form.get("neg_boxes", "[]")) or []
    except json.JSONDecodeError as e:
        return jsonify({"error": f"Invalid JSON: {e}"}), 400
    if not input_boxes:
        return jsonify({"error": "No boxes provided"}), 400

    state = sam3_processor.set_image(img)
    sam3_processor.confidence_threshold = 0.1
    results = []

    for i, box in enumerate(input_boxes):
        try:
            x1, y1, x2, y2 = (int(c) for c in box[:4])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            cx, cy = (x1 + x2) / 2.0 / w, (y1 + y2) / 2.0 / h
            bw, bh = (x2 - x1) / float(w), (y2 - y1) / float(h)

            sam3_processor.reset_all_prompts(state)
            state = sam3_processor.add_geometric_prompt(
                box=[cx, cy, bw, bh], label=True, state=state,
            )
            for n in neg_boxes:
                nx1, ny1, nx2, ny2 = (int(c) for c in n[:4])
                state = sam3_processor.add_geometric_prompt(
                    box=[(nx1 + nx2) / 2.0 / w, (ny1 + ny2) / 2.0 / h,
                         (nx2 - nx1) / float(w), (ny2 - ny1) / float(h)],
                    label=False, state=state,
                )

            if state.get("masks") is None or len(state["masks"]) == 0:
                continue
            masks  = state["masks"].cpu().numpy()
            scores = state["scores"].cpu().numpy()
            dboxes = state["boxes"].cpu().numpy()

            best, best_iou = 0, 0.0
            for di in range(len(dboxes)):
                iou_ = _iou(
                    (x1, y1, x2, y2),
                    (int(dboxes[di][0]), int(dboxes[di][1]),
                     int(dboxes[di][2]), int(dboxes[di][3])),
                )
                if iou_ > best_iou:
                    best, best_iou = di, iou_

            mask = masks[best, 0]
            ys, xs = np.where(mask)
            tight = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())] \
                if len(xs) else [x1, y1, x2, y2]

            buf = io.BytesIO()
            Image.fromarray(mask.astype(np.uint8) * 255).save(buf, format="PNG")
            results.append({
                "index": i,
                "input_box": [x1, y1, x2, y2],
                "tight_box": tight,
                "det_box": [int(dboxes[best][0]), int(dboxes[best][1]),
                             int(dboxes[best][2]), int(dboxes[best][3])],
                "mask_b64": base64.b64encode(buf.getvalue()).decode(),
                "iou_score": round(best_iou, 3),
                "confidence": round(float(scores[best]), 3),
                "mask_size": [w, h],
                "mask_pixels": int(mask.sum()),
            })
        except Exception as e:
            logger.warning(f"Box {i} segmentation failed: {e}")

    return jsonify({"results": results, "image_size": [w, h]})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": sam3_processor is not None,
        "device": str(device),
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()
    load_model()
    app.run(host=args.host, port=args.port, debug=False)
