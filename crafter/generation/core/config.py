"""Configuration for the Crafter agentic loop."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class CraftConfig:
    """Configuration for the agentic craft loop."""

    # API settings (OpenAI-compatible endpoint).
    api_key: str = ""
    api_base_url: str = "https://openrouter.ai/api/v1"
    api_timeout: int = 300

    # Model per agent role. Three user-facing slots — llm, vlm, generator —
    # drive the five internal roles below. ``quick`` defaults to a small,
    # cheap router model and is rarely worth surfacing.
    planner_model: str = "anthropic/claude-opus-4.6"
    refiner_model: str = "anthropic/claude-opus-4.6"
    critic_model: str = "google/gemini-3.1-pro-preview"
    quick_model: str = "google/gemini-3.1-flash-lite-preview"
    generator_model: str = "google/gemini-3-pro-image-preview"

    # When True, every chat-completion call uses temperature 0, collapsing
    # plan exploration to a single deterministic candidate.
    temperature_zero_mode: bool = False

    # Loop settings
    max_iterations: int = 3
    quality_threshold: float = 7.5
    max_retries_per_generation: int = 3
    agent_judge_max_iter: int = 3

    # Output
    output_dir: str = "./craft_output"
    save_all_iterations: bool = True

    # Reference images
    max_reference_images: int = 3
    reference_cache_dir: str = ".craft_cache/references"
    serper_api_key: str = ""  # optional, enables Serper image search

    # Plan-exploration cap. The session infers per-sample K from the
    # input (visual_style, refer image, aesthetic intent, role) and uses
    # this only as a fallback / upper bound.
    num_variants: int = 3

    # Internal-only flags. Kept here so the harness components stay
    # togglable for research; the YAML does not surface them.
    use_visual_grounder: bool = True
    use_skill_evolver: bool = True
    use_figure_spec: bool = True
    use_visual_metaphor_translator: bool = True

    def ensure_dirs(self) -> None:
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.reference_cache_dir).mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_yaml(cls, config_path: str | Path | None = None, **overrides) -> "CraftConfig":
        """Build a CraftConfig from the active YAML.

        The YAML exposes three model slots — ``llm``, ``vlm``,
        ``generator`` — which are mapped to the internal agent roles.
        Direct keyword overrides win over YAML values.
        """
        from crafter.shared.config import load_config

        cfg = load_config(config_path)
        m = cfg.models
        llm = m.get("llm") or cls.planner_model
        vlm = m.get("vlm") or cls.critic_model
        gen = m.get("generator") or cls.generator_model
        import os
        kwargs = dict(
            api_key=cfg.api_key(),
            api_base_url=cfg.api_base_url(),
            planner_model=llm,
            refiner_model=llm,
            critic_model=vlm,
            quick_model=cls.quick_model,
            generator_model=gen,
            serper_api_key=os.environ.get("SERPER_API_KEY", ""),
        )
        kwargs.update(overrides)
        return cls(**kwargs)
