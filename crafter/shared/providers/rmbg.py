"""RMBG-2.0 background-removal provider — HTTP client for self-hosted service."""
from __future__ import annotations

import logging
from typing import Optional

import requests

from ._config import RMBGConfig
from .base import RMBGProvider

logger = logging.getLogger(__name__)


class RMBGService(RMBGProvider):
    """HTTP client for self-hosted RMBG-2.0 background-removal service."""

    def __init__(self, config: Optional[RMBGConfig] = None):
        self.cfg = config or RMBGConfig()

    def remove_background(self, image_bytes: bytes) -> bytes:
        """POST RGB PNG bytes, return RGBA PNG bytes with bg removed."""
        url = f"{self.cfg.server_url.rstrip('/')}/remove"
        r = requests.post(url, files={"image": ("img.png", image_bytes, "image/png")},
                           timeout=self.cfg.timeout_s)
        r.raise_for_status()
        return r.content


# Default singleton
_DEFAULT: Optional[RMBGProvider] = None


def get_default_rmbg() -> RMBGProvider:
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = RMBGService()
    return _DEFAULT


def set_default_rmbg(provider: RMBGProvider) -> None:
    global _DEFAULT
    _DEFAULT = provider
