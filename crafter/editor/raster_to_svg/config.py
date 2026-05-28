"""Pipeline configuration with YAML loading."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class APIConfig:
    base_url: str = "https://openrouter.ai/api/v1"
    api_key: str = ""
    timeout: int = 300


@dataclass
class ModelConfig:
    grounding: str = "openai/gpt-5.4"
    text_ocr: str = "openai/gpt-5.4"
    judge_ensemble: list[str] = field(default_factory=lambda: [
        "openai/gpt-5.4",
        "gemini-3.1-flash-lite-preview",
        "doubao-seed-2.0-pro",
    ])


@dataclass
class SAM3Config:
    server_url: str = ""
    server_url_file: str = ""  # fallback: read URL from file
    confidence_threshold: float = 0.3
    bbox_confidence: float = 0.1


@dataclass
class MaskQualityConfig:
    max_retries: int = 3
    min_boundary_smoothness: float = 0.4
    max_fragment_count: int = 3
    min_largest_fragment_ratio: float = 0.85
    jagged_threshold: float = 0.6
    use_rmbg_fallback: bool = True


@dataclass
class VectorizationConfig:
    min_area: int = 500
    contour_epsilon: float = 0.02
    corner_roundness_threshold: float = 0.1
    circularity_threshold: float = 0.85


@dataclass
class ColorConfig:
    kmeans_clusters_fill: int = 3
    kmeans_clusters_stroke: int = 2
    erode_px: int = 5
    border_width: int = 10


@dataclass
class PipelineConfig:
    max_judge_rounds: int = 4
    quality_threshold: float = 6.5
    save_intermediates: bool = True
    output_formats: list[str] = field(default_factory=lambda: ["svg", "drawio"])
    # Sub-configs
    api: APIConfig = field(default_factory=APIConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    sam3: SAM3Config = field(default_factory=SAM3Config)
    mask_quality: MaskQualityConfig = field(default_factory=MaskQualityConfig)
    vectorization: VectorizationConfig = field(default_factory=VectorizationConfig)
    color: ColorConfig = field(default_factory=ColorConfig)

    @classmethod
    def from_yaml(cls, path: str) -> PipelineConfig:
        with open(path) as f:
            raw = yaml.safe_load(f) or {}

        cfg = cls()
        # Resolve env vars
        if "api" in raw:
            api = raw["api"]
            cfg.api.base_url = api.get("base_url", cfg.api.base_url)
            key = api.get("api_key", "")
            if key.startswith("${") and key.endswith("}"):
                key = os.environ.get(key[2:-1], "")
            cfg.api.api_key = key or os.environ.get("OPENROUTER_API_KEY", "")
            cfg.api.timeout = api.get("timeout", cfg.api.timeout)

        if "models" in raw:
            m = raw["models"]
            cfg.models.grounding = m.get("grounding", cfg.models.grounding)
            cfg.models.text_ocr = m.get("text_ocr", cfg.models.text_ocr)
            if "judge_ensemble" in m:
                cfg.models.judge_ensemble = m["judge_ensemble"]

        if "sam3" in raw:
            s = raw["sam3"]
            url = s.get("server_url", "")
            if url.startswith("${") and url.endswith("}"):
                url = os.environ.get(url[2:-1], "")
            cfg.sam3.server_url = url
            cfg.sam3.server_url_file = s.get("server_url_file", "")
            cfg.sam3.confidence_threshold = s.get("confidence_threshold", cfg.sam3.confidence_threshold)
            cfg.sam3.bbox_confidence = s.get("bbox_confidence", cfg.sam3.bbox_confidence)

        if "pipeline" in raw:
            p = raw["pipeline"]
            cfg.max_judge_rounds = p.get("max_judge_rounds", cfg.max_judge_rounds)
            cfg.quality_threshold = p.get("quality_threshold", cfg.quality_threshold)
            cfg.save_intermediates = p.get("save_intermediates", cfg.save_intermediates)
            cfg.output_formats = p.get("output_formats", cfg.output_formats)

        if "mask_quality" in raw:
            mq = raw["mask_quality"]
            for k, v in mq.items():
                if hasattr(cfg.mask_quality, k):
                    setattr(cfg.mask_quality, k, v)

        if "vectorization" in raw:
            for k, v in raw["vectorization"].items():
                if hasattr(cfg.vectorization, k):
                    setattr(cfg.vectorization, k, v)

        if "color" in raw:
            for k, v in raw["color"].items():
                if hasattr(cfg.color, k):
                    setattr(cfg.color, k, v)

        return cfg

    def resolve_sam3_url(self) -> str:
        """Get SAM3 server URL, trying env/file fallbacks."""
        if self.sam3.server_url:
            return self.sam3.server_url
        if self.sam3.server_url_file:
            p = Path(self.sam3.server_url_file)
            if p.exists():
                return p.read_text().strip()
        # Try default locations
        import os
        candidates = []
        if os.environ.get("CRAFTER_HOME"):
            candidates.append(Path(os.environ["CRAFTER_HOME"]) / ".sam3_server_url")
        candidates += [
            Path.cwd() / ".sam3_server_url",
            Path.home() / ".sam3_server_url",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate.read_text().strip()
        return ""
