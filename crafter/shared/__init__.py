"""crafter.shared — cross-harness infrastructure.

Modules here are imported by BOTH crafter.generation (§4.2) and
crafter.editor (§4.3). Don't put harness-specific logic here — only
truly common building blocks:

    providers/        Provider abstraction (OpenAI-compatible / OpenRouter)
                      → configs/*.yaml selects which provider serves
                        which logical role at runtime
    model_router      Unified call-routing layer over providers
    sam3_client       SAM3 grounding HTTP client (craftEditor processing)
    judge             Multi-VLM ensemble judge (eval pipelines)
    config            YAML config loader + ResolvedConfig dataclass
"""
from __future__ import annotations
