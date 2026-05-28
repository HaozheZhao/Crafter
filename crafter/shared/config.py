"""Loads the active Crafter YAML.

    from crafter.shared.config import load_config
    cfg = load_config()                  # reads $CRAFTER_CONFIG or configs/default.yaml
    cfg.models["llm"]                    # → "anthropic/claude-opus-4.6"

API credentials are read from environment variables (``OPENROUTER_API_KEY`` /
``OPENROUTER_API_BASE``); the YAML never holds secrets.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)

OPENROUTER_BASE = "https://openrouter.ai/api/v1"


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


@dataclass
class ResolvedConfig:
    """Parsed Crafter YAML."""
    raw: dict
    config_path: Path

    @property
    def models(self) -> dict:
        return self.raw.get("models", {})

    def api_key(self) -> str:
        key = os.environ.get("OPENROUTER_API_KEY", "")
        if not key:
            logger.warning("OPENROUTER_API_KEY is empty; calls will fail.")
        return key

    def api_base_url(self) -> str:
        return os.environ.get("OPENROUTER_API_BASE", OPENROUTER_BASE)


def load_config(path: Optional[str] = None) -> ResolvedConfig:
    """Resolve the active config: ``path`` arg, ``$CRAFTER_CONFIG``, or default."""
    if path is None:
        path = os.environ.get("CRAFTER_CONFIG", "")
    if not path:
        path = str(_repo_root() / "configs" / "default.yaml")

    cfg_path = Path(path).expanduser()
    if not cfg_path.is_absolute():
        cfg_path = _repo_root() / cfg_path
    if not cfg_path.exists():
        raise FileNotFoundError(f"Crafter config not found: {cfg_path}")

    raw = yaml.safe_load(cfg_path.read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Top-level YAML in {cfg_path} must be a mapping")
    return ResolvedConfig(raw=raw, config_path=cfg_path)
