"""SAM3 grounded-segmentation provider — wraps the existing HTTP client."""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

from ._config import SAM3Config
from .base import SAM3Provider, SAM3Result

logger = logging.getLogger(__name__)

# Import the existing low-level HTTP client (in r2e/tools/sam3_client.py).
# This stays as-is to preserve behavior; we wrap it to expose the official
# editor-package interface.
_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
from crafter.editor.raster_to_svg.tools.sam3_client import SAM3Client  # noqa: E402


class SAM3Server(SAM3Provider):
    """HTTP client for the self-hosted SAM3 server.

    Wraps `r2e.tools.sam3_client.SAM3Client` to expose the editor-package
    `SAM3Provider` interface.
    """

    def __init__(self, config: Optional[SAM3Config] = None):
        self.cfg = config or SAM3Config()
        if not self.cfg.server_url:
            raise RuntimeError(
                "SAM3 server URL missing (set SAM3_SERVER_URL env)")
        self._client = SAM3Client(self.cfg.server_url, timeout=self.cfg.timeout_s)

    def wait_ready(self, max_wait_s: int = 300, interval_s: int = 10) -> bool:
        return self._client.wait_ready(max_wait=max_wait_s, interval=interval_s)

    def segment_text(self, image_path: Path, prompts: list[str],
                      min_score: float = 0.3,
                      return_masks: bool = False,
                      ) -> list[list[SAM3Result]]:
        # Underlying client returns flat list of {x1, y1, x2, y2, score, prompt[, mask_b64]}
        raw = self._client.segment_text(
            str(image_path), prompts, min_score=min_score,
            return_masks=return_masks)
        # Group by prompt (preserve input order)
        by_prompt: dict[str, list[SAM3Result]] = {p: [] for p in prompts}
        for r in raw:
            res = SAM3Result(
                x1=r["x1"], y1=r["y1"], x2=r["x2"], y2=r["y2"],
                score=r["score"], prompt=r["prompt"],
                mask_b64=r.get("mask_b64"),
            )
            by_prompt.setdefault(r["prompt"], []).append(res)
        return [by_prompt[p] for p in prompts]

    def segment_bbox(self, image_path: Path,
                      bboxes: list[tuple[float, float, float, float]],
                      ) -> list[Optional[str]]:
        # Underlying client returns list of {mask_b64} or None
        raw = self._client.segment_bbox(str(image_path), bboxes)
        return [r.get("mask_b64") if r else None for r in raw]

    @property
    def raw_client(self) -> SAM3Client:
        """Return the underlying client for direct use."""
        return self._client


# Default singleton
_DEFAULT: Optional[SAM3Provider] = None


def get_default_sam3() -> SAM3Provider:
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = SAM3Server()
    return _DEFAULT


def set_default_sam3(provider: SAM3Provider) -> None:
    global _DEFAULT
    _DEFAULT = provider
