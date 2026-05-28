"""CraftEditor configuration — production settings only.

External-service credentials and endpoints are resolved via
``crafter.editor._env``. Set these env vars at deployment time:

    API_KEY              — chat-LLM API key
    API_ENDPOINT         — chat-LLM base URL (OpenAI-compatible /v1)
    IMAGE_EDIT_API_KEY   — image-edit API key
    IMAGE_EDIT_ENDPOINT  — image-edit full endpoint URL
    SAM3_SERVER_URL      — SAM3 grounding server (or discovery file
                           at $CRAFTER_HOME/.sam3_server_url)
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Re-export env accessors so callers can write
# `from crafter.editor.config import api_key` if they prefer.
from ._env import (
    api_endpoint, api_key, image_edit_api_key, image_edit_endpoint,
    rmbg_url, sam3_url,
)


# ===================================================================
# LLM (text + vision)
# ===================================================================
LLM_DESIGNER_MODEL = "openai/gpt-5.5"           # D role (both phases)
LLM_REVISER_MODEL = "openai/gpt-5.5"            # R role (both phases)
LLM_VERIFIER_MODEL = "openai/gpt-5.5"           # V role in extraction phase

# Vision-language model used by quick_judge inside the composition loop
# (lighter than the 3-VLM headline ensemble).
LLM_QUICK_JUDGE_MODEL = "openai/gpt-5.5"

# 3-VLM ensemble — used both by the composition-phase polish acceptance
# gate and by the external benchmark judge. Order is stable.
JUDGE_ENSEMBLE = [
    "gemini-3.1-flash-lite-preview",
    "openai/gpt-5.4",
    "doubao-seed-2.0-pro",
]


# Backward-compat alias for callers that still import `llm_api_key`.
llm_api_key = api_key


# ===================================================================
# Image editor (gpt-image-2) — extraction-phase E role
# ===================================================================
IMAGE_EDIT_MODEL = "gpt-image-2"


# ===================================================================
# Extraction phase
# ===================================================================
@dataclass(frozen=True)
class ExtractionConfig:
    # Verify-then-refine loop bound. T=2 captures most of the gain
    # T=3 would give while keeping wall-time bounded.
    max_iters: int = 2
    # Designer role uses the strong model so the keep/delete plan is
    # specific. Verifier role uses a cheaper, decoupled model.
    designer_model: str = LLM_DESIGNER_MODEL
    executor_model: str = IMAGE_EDIT_MODEL
    verifier_model: str = LLM_VERIFIER_MODEL
    # When False, the extraction phase is skipped and the original
    # raster is passed straight to the processing (grounding) phase.
    # Default True matches the published pipeline; flip for SAM-only
    # runs or for image-edit-free deployments.
    use_gpt_image2: bool = True


# ===================================================================
# Processing phase (captioning + grounding + classifying)
# ===================================================================
@dataclass(frozen=True)
class ProcessingConfig:
    # VLM produces per-element referring expressions on the cleaned canvas.
    caption_model: str = LLM_DESIGNER_MODEL
    # SAM3 grounds each referring expression (caption-driven).
    sam3_min_score: float = 0.20
    # Broad-class supplemental SAM3 prompts. Captures same-class repeats
    # that caption-driven referring grounding cannot disambiguate (typical
    # for dense posters with many portraits).
    af_supplemental_prompts: tuple = (
        "person", "robot", "animal", "chart", "thumbnail", "logo",
    )
    af_supplemental_min_score: float = 0.5
    af_supplemental_iou_dedup: float = 0.4
    af_supplemental_per_prompt_cap: int = 15
    af_supplemental_area_cap_px2: int = 80_000
    af_supplemental_max_total: int = 50
    # Vector vs raster classifier model.
    classify_model: str = LLM_DESIGNER_MODEL


# ===================================================================
# Composition phase
# ===================================================================
@dataclass(frozen=True)
class CompositionConfig:
    # Designer role (skeleton author) and reviser role (refine LLM).
    designer_model: str = LLM_DESIGNER_MODEL
    reviser_model: str = LLM_REVISER_MODEL
    # Skeleton best-of-N temperatures. Sequential best-of-N is the only
    # mode; the first candidate to clear `skeleton_early_adopt_threshold`
    # is accepted and remaining temperatures are skipped.
    skeleton_temperatures: tuple = (0.20, 0.45)
    skeleton_early_adopt_threshold: float = 7.0
    # Refinement-loop bound (paper §3.3.2 T=4).
    refine_max_iters: int = 4
    # Hybrid critic: VLM quick-judge + programmatic checkers.
    quick_judge_model: str = LLM_QUICK_JUDGE_MODEL
    enable_structural_critic: bool = True


@dataclass(frozen=True)
class Config:
    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    composition: CompositionConfig = field(default_factory=CompositionConfig)

    @classmethod
    def from_yaml(cls, config_path: str | Path | None = None) -> "Config":
        """Build a CraftEditor Config from the unified Crafter YAML.

        Reads model names and per-phase hyperparameters from
        ``crafter.shared.config`` (default: ``configs/default.yaml``)
        so a single YAML drives both sub-packages.
        """
        from crafter.shared.config import load_config

        cfg = load_config(config_path)
        m = cfg.models
        llm = m.get("llm") or LLM_DESIGNER_MODEL
        vlm = m.get("vlm") or LLM_VERIFIER_MODEL
        gen = m.get("generator") or IMAGE_EDIT_MODEL

        return cls(
            extraction=ExtractionConfig(
                max_iters=ExtractionConfig.max_iters,
                designer_model=llm,
                executor_model=gen,
                verifier_model=vlm,
                use_gpt_image2=True,
            ),
            processing=ProcessingConfig(
                caption_model=vlm,
                classify_model=llm,
            ),
            composition=CompositionConfig(
                designer_model=llm,
                reviser_model=llm,
                refine_max_iters=CompositionConfig.refine_max_iters,
                skeleton_temperatures=CompositionConfig.skeleton_temperatures,
                skeleton_early_adopt_threshold=CompositionConfig.skeleton_early_adopt_threshold,
                quick_judge_model=vlm,
            ),
        )


DEFAULT_CONFIG = Config()
