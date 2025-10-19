"""Ollama caption contract utilities."""

from .contract import build_caption_payload

__all__ = ["build_caption_payload"]

try:  # pragma: no cover - optional dependency guard
    from .workflow import OllamaCaptionEngine
except ModuleNotFoundError as exc:  # pragma: no cover - requests missing
    class OllamaCaptionEngine:  # type: ignore[misc]
        def __init__(self, *_, **__) -> None:
            raise ModuleNotFoundError("The 'requests' package is required for OllamaCaptionEngine") from exc
else:  # pragma: no cover - exercised indirectly
    __all__.append("OllamaCaptionEngine")

try:  # pragma: no cover - optional dependency guard
    from .adapter import OllamaClient, OllamaResponseError, OllamaTimeoutError
except ModuleNotFoundError as exc:  # pragma: no cover - requests missing
    class OllamaClient:  # type: ignore[misc]
        def __init__(self, *_, **__) -> None:
            raise ModuleNotFoundError("The 'requests' package is required for OllamaClient") from exc

    OllamaResponseError = ModuleNotFoundError
    OllamaTimeoutError = ModuleNotFoundError
else:  # pragma: no cover - exercised indirectly
    __all__.extend(["OllamaClient", "OllamaResponseError", "OllamaTimeoutError"])
