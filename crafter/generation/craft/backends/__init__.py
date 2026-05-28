"""Image-generation backends.

The ``ImageGenBackend`` interface lets the image generator swap between
vendors without touching the rest of the pipeline. The default backend
delegates the actual call to ``ModelRouter``.
"""
from .base import ImageGenBackend, GenerateResult
from .chat_image import ChatImageBackend
from .gpt_image import GptImageBackend  # stub; raises NotImplementedError

__all__ = [
    "ImageGenBackend", "GenerateResult",
    "ChatImageBackend", "GptImageBackend",
]
