"""Provider abstraction layer — swap external services without touching pipeline code."""
from .base import (
    LLMProvider, LLMResponse,
    ImageEditProvider, ImageEditResponse,
    SAM3Provider, SAM3Result,
    RMBGProvider,
)
from .llm import OpenRouterLLM, OpenAILLM, get_default_llm, set_default_llm
from .image_edit import GptImageEditor, get_default_image_edit, set_default_image_edit
from .sam3 import SAM3Server, get_default_sam3, set_default_sam3
from .rmbg import RMBGService, get_default_rmbg, set_default_rmbg

__all__ = [
    # Base interfaces
    "LLMProvider", "LLMResponse",
    "ImageEditProvider", "ImageEditResponse",
    "SAM3Provider", "SAM3Result",
    "RMBGProvider",
    # Concrete impls
    "OpenRouterLLM", "OpenAILLM", "GptImageEditor", "SAM3Server", "RMBGService",
    # Default singletons (used by facade)
    "get_default_llm", "set_default_llm",
    "get_default_image_edit", "set_default_image_edit",
    "get_default_sam3", "set_default_sam3",
    "get_default_rmbg", "set_default_rmbg",
]
