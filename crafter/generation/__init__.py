"""Crafter — text + paper context → AI raster scientific figure.

Public API:
    from crafter.generation import generate
    bytes_ = generate(paper_text, caption, task="t2i", role="academic",
                       refer_image_path=None)

    # Or, for full control over the harness:
    from crafter.generation import CraftSession, CraftInput, CraftConfig
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from crafter.generation.core.config import CraftConfig
from crafter.generation.craft.session import (
    CraftSession,
    CraftInput,
    CraftResult,
    CraftIteration,
)

__all__ = [
    "generate",
    "CraftConfig",
    "CraftSession",
    "CraftInput",
    "CraftResult",
    "CraftIteration",
]


def generate(
    paper_text: str,
    caption: str,
    *,
    task: str = "t2i",
    role: str = "academic",
    refer_image_path: Optional[str] = None,
    refer_image_role: str = "",
    venue: str = "neurips",
    num_variants: int = 3,
    config: Optional[CraftConfig] = None,
    output_path: str = "",
) -> Optional[bytes]:
    """Thin wrapper: build CraftInput + CraftConfig, run CraftSession, return bytes.

    For batch / bench use, callers should construct a per-sample
    `CraftConfig` with `output_dir=str(out_dir / sid)` and apply
    truncation upstream (`paper_text[:15000]`, `caption[:500]`).
    """
    cfg = config or CraftConfig(api_key=_load_api_key())
    cfg.ensure_dirs()

    if refer_image_path and not refer_image_role:
        refer_image_role = {
            "inpaint":  "preserve_partial",
            "keyelems": "use_elements",
            "sketch":   "refine_sketch",
        }.get(task, "generic_edit")

    if refer_image_path:
        num_variants = 1

    ci = CraftInput(
        paper_text=paper_text,
        description=caption,
        figure_type=task,
        venue=venue,
        visual_style="",
        reference_paths=[refer_image_path] if refer_image_path else [],
        max_iterations=cfg.max_iterations,
        num_variants=num_variants,
        output_path=output_path,
        skill_round=0,
        role=role,
        refer_image_role=refer_image_role,
    )
    session = CraftSession(cfg)
    result = session.craft(ci)
    path = result.final_image_path or result.best_image_path
    if not path or not Path(path).exists():
        return None
    return Path(path).read_bytes()


def _load_api_key() -> str:
    import os
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY not set in environment")
    return key
