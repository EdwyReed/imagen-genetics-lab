"""Simplified Imagen client that materialises generated assets on disk."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

__all__ = ["ImagenService", "ImageResult"]


@dataclass(frozen=True)
class ImageResult:
    caption: str
    image_path: Path
    mime_type: str = "image/jpeg"


class ImagenService:
    """Writes deterministic placeholder JPEG data for testing flows."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(self, caption: str, *, seed: Optional[int] = None) -> ImageResult:
        digest = base64.urlsafe_b64encode(caption.encode("utf-8"))[:32].decode("ascii")
        filename = f"imagen_{digest}.jpg"
        path = self.output_dir / filename
        payload = (
            "This is a placeholder JPEG containing the caption:\n" + caption
        ).encode("utf-8")
        path.write_bytes(payload)
        return ImageResult(caption=caption, image_path=path)
