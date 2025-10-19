from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol


@dataclass(frozen=True)
class ImagenRequest:
    prompt: str
    aspect_ratio: str
    variants: int
    person_mode: str | None
    guidance_scale: float | None
    seed: int | None


@dataclass(frozen=True)
class ImagenResult:
    response: Any


class ImagenEngineProtocol(Protocol):
    def generate(self, request: ImagenRequest) -> ImagenResult:
        """Invoke Imagen with the given request."""
