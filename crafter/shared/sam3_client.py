"""SAM3 client — thin re-export from the raster_to_svg submodule.

Underlying implementation: ``r2e.tools.sam3_client``. We expose it under
``crafter.shared.sam3_client`` for the unified import path; existing
editor modules import via this path
unchanged.
"""
from __future__ import annotations

from crafter.editor.raster_to_svg.tools.sam3_client import SAM3Client  # noqa: F401

__all__ = ["SAM3Client"]
