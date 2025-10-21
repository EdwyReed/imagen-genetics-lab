from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import List, Tuple

from .adapters import smart_adapt
from .schema import ImagenConfig

_PLACEHOLDER_JPEG = base64.b64decode(
    (
        "/9j/4AAQSkZJRgABAQEAYABgAAD/2wBDAAMCAgICAgMCAgIDAwMDBAYEBAQEBAgGBgUGCQgKCgkICQkK"
        "CgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCv/2wBDAQMDAwQDBAgEBAgKCgoKCgoKCgoKCgoKCgoKCgoK"
        "CgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCv/wAARCAAaABoDASIAAhEBAxEB/8QAFQABAQAAAAAA"
        "AAAAAAAAAAABf/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAgT/xAAUE"
        "QEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAxEAPwCfAAH/2Q=="
    )
)


@dataclass(slots=True)
class ImagenVariant:
    index: int
    image_bytes: bytes


@dataclass(slots=True)
class ImagenMeta:
    model: str
    aspect_ratio: str
    person_mode: str
    resolution: str
    safety_filter: str


class ImagenGenerator:
    def __init__(self, config: ImagenConfig, dry_run: bool = False) -> None:
        self.config = config
        self.dry_run = dry_run

    def generate(self, caption: str) -> Tuple[List[ImagenVariant], ImagenMeta]:
        meta = ImagenMeta(
            model=self.config.model,
            aspect_ratio=self.config.resolution,
            person_mode=self.config.person_mode,
            resolution=self.config.resolution,
            safety_filter=self.config.safety_filter,
        )
        if self.dry_run:
            variants = [ImagenVariant(index=i, image_bytes=_PLACEHOLDER_JPEG) for i in range(1, self.config.variants + 1)]
            return variants, meta

        response = smart_adapt.imagen_call(
            model=self.config.model,
            caption=caption,
            variants=self.config.variants,
            aspect_ratio=self.config.resolution,
            person_mode=self.config.person_mode,
            safety_filter=self.config.safety_filter,
        )

        variants: List[ImagenVariant] = []
        for idx, generated in enumerate(getattr(response, "generated_images", []), start=1):
            image = getattr(generated, "image", None)
            data = getattr(image, "image_bytes", None) or getattr(image, "imageBytes", None)
            if not data:
                continue
            variants.append(ImagenVariant(index=idx, image_bytes=data))
        return variants, meta


__all__ = ["ImagenGenerator", "ImagenVariant", "ImagenMeta"]
