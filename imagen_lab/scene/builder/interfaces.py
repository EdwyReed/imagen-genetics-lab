from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Protocol

from imagen_lab.learning import StyleFeedback


@dataclass(frozen=True)
class SceneRequest:
    sfw_level: float
    temperature: float
    feedback: Optional[StyleFeedback] = None
    template_id: Optional[str] = None
    profile_id: Optional[str] = None
    macro_snapshot: Mapping[str, Any] | None = None
    meso_snapshot: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class SceneDescription:
    template_id: str
    caption_bounds: Mapping[str, Any]
    aspect_ratio: str
    gene_ids: Mapping[str, Optional[str]]
    payload: Mapping[str, Any]
    summary: str
    raw: Any

    def ollama_payload(self) -> Mapping[str, Any]:
        return self.payload


class SceneBuilderProtocol(Protocol):
    def build_scene(self, request: SceneRequest) -> SceneDescription:
        """Generate a scene description for downstream layers."""

    def rebuild_from_genes(
        self,
        genes: Mapping[str, Optional[str]],
        request: SceneRequest,
    ) -> SceneDescription:
        """Rebuild a scene from a fixed set of gene choices."""
