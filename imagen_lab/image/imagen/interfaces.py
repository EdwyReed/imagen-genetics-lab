from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence


@dataclass(frozen=True)
class ImagenRequest:
    prompt: str
    aspect_ratio: str
    variants: int
    person_mode: str | None
    guidance_scale: float | None
    seed: int | None


@dataclass(frozen=True)
class ImagenVariant:
    index: int
    image_bytes: bytes
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class ImagenResult:
    response: Any
    variants: Sequence[ImagenVariant]
    metadata: Mapping[str, Any]


class ImagenEngineProtocol(Protocol):
    def generate(self, request: ImagenRequest) -> ImagenResult:
        """Invoke Imagen with the given request."""
