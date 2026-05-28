"""HTTP client for the SAM3 segmentation server."""
from __future__ import annotations

import base64
import io
import json
import logging
import time

import numpy as np
import requests
from PIL import Image

logger = logging.getLogger(__name__)


class SAM3Client:
    """Client for the SAM3 HTTP server."""

    def __init__(self, server_url: str, timeout: int = 300):
        self.url = server_url.rstrip("/")
        self.timeout = timeout

    def wait_ready(self, max_wait: int = 300, interval: int = 10) -> bool:
        """Wait for SAM3 server to be ready."""
        for i in range(max_wait // interval):
            try:
                r = requests.get(f"{self.url}/health", timeout=5)
                if r.status_code == 200:
                    info = r.json()
                    logger.info(f"SAM3 ready: {info}")
                    return True
            except Exception:
                pass
            logger.info(f"Waiting for SAM3... ({i + 1})")
            time.sleep(interval)
        return False

    def segment_text(
        self,
        image_path: str,
        prompts: list[str],
        min_score: float = 0.3,
        return_masks: bool = False,
    ) -> list[dict]:
        """Text-prompted detection with optional masks.

        Returns list of {x1,y1,x2,y2,score,prompt[,mask_b64,mask_pixels]}.
        When return_masks=True, each result includes the pixel-perfect mask
        directly from SAM3's text-prompted inference (no second bbox pass needed).
        """
        with open(image_path, "rb") as f:
            resp = requests.post(
                f"{self.url}/segment",
                files={"image": f},
                data={
                    "prompts": ",".join(prompts),
                    "min_score": str(min_score),
                    "return_masks": "true" if return_masks else "false",
                },
                timeout=self.timeout,
            )
        resp.raise_for_status()
        data = resp.json()
        return data.get("results", data.get("boxes", []))

    def segment_bbox(
        self,
        image_path: str,
        boxes: list[list[int]],
    ) -> list[dict]:
        """Bbox-prompted segmentation. Returns masks + refined bboxes.

        Each result: {index, input_box, tight_box, mask_b64, confidence, mask_pixels}
        """
        with open(image_path, "rb") as f:
            resp = requests.post(
                f"{self.url}/segment_bbox",
                files={"image": f},
                data={"boxes": json.dumps(boxes)},
                timeout=self.timeout,
            )
        resp.raise_for_status()
        data = resp.json()
        return data.get("results", [])

    def segment_single_bbox(
        self,
        image_path: str,
        bbox: list[int],
    ) -> tuple[np.ndarray | None, dict]:
        """Segment a single bbox, return (mask_array, result_dict)."""
        results = self.segment_bbox(image_path, [bbox])
        if not results:
            return None, {}
        result = results[0]
        mask_b64 = result.get("mask_b64", "")
        if not mask_b64:
            return None, result
        mask_img = Image.open(io.BytesIO(base64.b64decode(mask_b64))).convert("L")
        mask_arr = np.array(mask_img)
        return mask_arr, result

    def crop_with_mask(
        self,
        image_path: str,
        bbox: list[int],
        mask: np.ndarray,
    ) -> Image.Image:
        """Crop image region using SAM3 mask for transparency."""
        img = Image.open(image_path).convert("RGBA")
        x1, y1, x2, y2 = bbox

        # Apply mask as alpha channel
        full_alpha = Image.fromarray(mask)
        img.putalpha(full_alpha)

        # Crop to bbox
        cropped = img.crop((x1, y1, x2, y2))
        return cropped
