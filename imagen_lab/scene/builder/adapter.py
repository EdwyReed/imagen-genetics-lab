from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

from imagen_lab.scene_builder import SceneBuilder, short_readable

from .interfaces import SceneBuilderProtocol, SceneDescription, SceneRequest


@dataclass
class SceneBuilderAdapter(SceneBuilderProtocol):
    """Adapter converting the legacy :class:`SceneBuilder` output into contracts."""

    builder: SceneBuilder

    def build_scene(self, request: SceneRequest) -> SceneDescription:
        scene = self.builder.build_scene(
            sfw_level=request.sfw_level,
            temperature=request.temperature,
            feedback=request.feedback,
            template_id=request.template_id,
        )
        return SceneDescription(
            template_id=scene.template_id,
            caption_bounds=dict(scene.caption_bounds),
            aspect_ratio=scene.aspect_ratio,
            gene_ids=dict(scene.gene_ids),
            payload=scene.ollama_payload(),
            summary=short_readable(scene),
            raw=scene,
        )

    def rebuild_from_genes(
        self,
        genes: Mapping[str, Optional[str]],
        request: SceneRequest,
    ) -> SceneDescription:
        scene = self.builder.rebuild_from_genes(
            genes,
            sfw_level=request.sfw_level,
            temperature=request.temperature,
            feedback=request.feedback,
        )
        return SceneDescription(
            template_id=scene.template_id,
            caption_bounds=dict(scene.caption_bounds),
            aspect_ratio=scene.aspect_ratio,
            gene_ids=dict(scene.gene_ids),
            payload=scene.ollama_payload(),
            summary=short_readable(scene),
            raw=scene,
        )
