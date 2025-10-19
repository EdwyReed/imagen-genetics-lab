from __future__ import annotations

from dataclasses import dataclass

from imagen_lab.prompting import imagen_call

from .interfaces import ImagenEngineProtocol, ImagenRequest, ImagenResult


@dataclass
class ImagenClientEngine(ImagenEngineProtocol):
    client: object
    model: str

    def generate(self, request: ImagenRequest) -> ImagenResult:
        response = imagen_call(
            self.client,
            self.model,
            request.prompt,
            request.aspect_ratio,
            request.variants,
            request.person_mode,
            guidance_scale=request.guidance_scale,
        )
        return ImagenResult(response=response)
