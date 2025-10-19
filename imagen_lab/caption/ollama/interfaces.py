from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Protocol

from .types import CaptionResult
from imagen_lab.scene.builder.interfaces import SceneDescription


@dataclass(frozen=True)
class CaptionRequest:
    scene: SceneDescription
    sfw_level: float
    temperature: float
    top_p: float
    seed: Optional[int]


class CaptionEngineProtocol(Protocol):
    def generate(self, request: CaptionRequest) -> CaptionResult:
        """Create a caption and final prompt for Imagen."""
