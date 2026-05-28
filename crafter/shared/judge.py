"""Multi-VLM ensemble judge — thin re-export from raster_to_svg submodule.

Underlying implementation: ``r2e.agents.judge.JudgeAgent``. Used by
craftEditor (§4.3) round-5 polish-acceptance gate and by paper-time
benchmark evaluation pipelines.
"""
from __future__ import annotations

from crafter.editor.raster_to_svg.agents.judge import JudgeAgent  # noqa: F401

__all__ = ["JudgeAgent"]
