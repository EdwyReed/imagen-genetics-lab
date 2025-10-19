from __future__ import annotations

import io
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, MutableMapping

import numpy as np

from imagen_lab.image.imagen.interfaces import ImagenResult, ImagenVariant

try:  # pragma: no cover - optional dependency
    from PIL import Image  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - pillow optional
    Image = None  # type: ignore


@dataclass(frozen=True)
class SavedVariant:
    """Persisted Imagen variant with decoded pixels for scoring."""

    prompt_path: str
    metadata: Mapping[str, Any]
    image: np.ndarray
    scene_payload: Mapping[str, Any]
    gene_choices: Mapping[str, Any] | None
    option_probabilities: Mapping[str, Any] | None

    def as_dict(self) -> Mapping[str, Any]:
        return {
            "prompt_path": self.prompt_path,
            "metadata": dict(self.metadata),
            "scene": dict(self.scene_payload),
        }


class ImageArtifactStore:
    """Persist Imagen variants to disk and expose decoded arrays."""

    def __init__(self, output_dir: Path) -> None:
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def persist_variants(
        self,
        *,
        session_id: str,
        imagen: ImagenResult,
        prompt: str,
        caption: str,
        scene_payload: Mapping[str, Any],
        meta_base: Mapping[str, Any],
    ) -> list[SavedVariant]:
        """Write Imagen variants to disk and return decoded data."""

        variants = list(imagen.variants or ())
        if not variants:
            return []

        timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        saved: list[SavedVariant] = []
        gene_choices = scene_payload.get("choices") if isinstance(scene_payload, Mapping) else None
        option_probabilities = (
            scene_payload.get("option_probabilities") if isinstance(scene_payload, Mapping) else None
        )

        for variant in variants:
            base_name = self._build_base_name(session_id, timestamp, variant)
            image_path = self._output_dir / f"{base_name}.jpg"
            image_path.write_bytes(variant.image_bytes)

            metadata = self._compose_metadata(
                base_meta=meta_base,
                caption=caption,
                scene_payload=scene_payload,
                variant=variant,
                session_id=session_id,
                prompt=prompt,
            )
            image_path.with_suffix(".json").write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            image_path.with_suffix(".txt").write_text(prompt + "\n", encoding="utf-8")

            image_array = self._decode_pixels(variant.image_bytes, variant.index, len(variants))
            saved.append(
                SavedVariant(
                    prompt_path=image_path.name,
                    metadata=metadata,
                    image=image_array,
                    scene_payload=scene_payload,
                    gene_choices=gene_choices if isinstance(gene_choices, Mapping) else None,
                    option_probabilities=option_probabilities
                    if isinstance(option_probabilities, Mapping)
                    else None,
                )
            )
        return saved

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _build_base_name(self, session_id: str, timestamp: str, variant: ImagenVariant) -> str:
        safe_session = session_id.replace("/", "-").replace("\\", "-")
        return f"{timestamp}_{safe_session}_{variant.index:02d}"

    def _compose_metadata(
        self,
        *,
        base_meta: Mapping[str, Any],
        caption: str,
        scene_payload: Mapping[str, Any],
        variant: ImagenVariant,
        session_id: str,
        prompt: str,
    ) -> Mapping[str, Any]:
        metadata: MutableMapping[str, Any] = {}
        if isinstance(base_meta, Mapping):
            metadata.update({str(k): v for k, v in base_meta.items()})
        metadata["session_id"] = session_id
        metadata["variant_index"] = variant.index
        metadata["caption"] = caption
        metadata["final_prompt"] = prompt
        metadata["scene_model"] = dict(scene_payload)
        variant_meta = variant.metadata if isinstance(variant.metadata, Mapping) else {}
        metadata["variant_metadata"] = dict(variant_meta)
        return metadata

    def _decode_pixels(self, blob: bytes, index: int, total: int) -> np.ndarray:
        if not blob:
            return self._synthetic_pixels(index, total)
        if Image is None:
            return self._synthetic_pixels(index, total)
        try:
            with Image.open(io.BytesIO(blob)) as img:  # type: ignore[name-defined]
                array = np.asarray(img.convert("RGB"), dtype=np.float32)
        except Exception:  # pragma: no cover - fallback when decoding fails
            return self._synthetic_pixels(index, total)
        if array.size == 0:
            return self._synthetic_pixels(index, total)
        return np.clip(array / 255.0, 0.0, 1.0)

    def _synthetic_pixels(self, index: int, total: int) -> np.ndarray:
        level = (index % max(total, 1) + 1) / float(max(total, 1) + 1)
        array = np.full((64, 64, 3), level, dtype=np.float32)
        return array


__all__ = ["ImageArtifactStore", "SavedVariant"]
