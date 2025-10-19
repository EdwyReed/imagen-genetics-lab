from __future__ import annotations

from dataclasses import dataclass

from imagen_lab.prompting import imagen_call

from .interfaces import ImagenEngineProtocol, ImagenRequest, ImagenResult, ImagenVariant


def _coerce_bytes(blob) -> bytes | None:
    if isinstance(blob, (bytes, bytearray)):
        return bytes(blob)
    if isinstance(blob, str):
        import base64
        import binascii

        try:
            return base64.b64decode(blob)
        except (ValueError, binascii.Error):  # pragma: no cover - defensive
            return None
    return None


def _extract_metadata(response, request: ImagenRequest) -> tuple[dict[str, object], dict[str, object]]:
    base: dict[str, object] = {}
    version = getattr(response, "model_version", None) or getattr(response, "modelVersion", None) or getattr(response, "model", None)
    if version is not None:
        base["imagen_version"] = version
    root_meta = getattr(response, "metadata", None)
    if isinstance(root_meta, dict):
        base.update(root_meta)
    elif hasattr(root_meta, "to_dict"):
        try:
            base.update(root_meta.to_dict())  # type: ignore[attr-defined]
        except Exception:  # pragma: no cover - defensive
            pass

    steps = base.get("steps") or base.get("inference_steps")
    if steps is not None:
        try:
            base["steps"] = int(steps)
        except (TypeError, ValueError):  # pragma: no cover - defensive
            base.pop("steps", None)

    sampler = base.get("sampler") or base.get("sampler_name")
    if sampler is not None:
        base["sampler"] = sampler

    if request.seed is not None:
        base.setdefault("seed", request.seed)

    return base, {k: v for k, v in base.items()}


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
        base_meta, shared_meta = _extract_metadata(response, request)
        variants: list[ImagenVariant] = []
        for idx, generated in enumerate(getattr(response, "generated_images", []), start=1):
            image = getattr(generated, "image", None)
            blob = None
            if image is not None:
                blob = _coerce_bytes(getattr(image, "image_bytes", None) or getattr(image, "imageBytes", None))
            if blob is None:
                blob = _coerce_bytes(getattr(generated, "image", None))
            if not blob:
                continue
            meta = {}
            meta.update(shared_meta)
            generated_meta = getattr(generated, "metadata", None)
            if isinstance(generated_meta, dict):
                meta.update(generated_meta)
            elif hasattr(generated_meta, "to_dict"):
                try:
                    meta.update(generated_meta.to_dict())  # type: ignore[attr-defined]
                except Exception:  # pragma: no cover - defensive
                    pass
            meta.setdefault("imagen_version", base_meta.get("imagen_version"))
            meta.setdefault("sampler", base_meta.get("sampler"))
            if request.seed is not None:
                meta.setdefault("seed", request.seed)
            variants.append(ImagenVariant(index=idx, image_bytes=blob, metadata=meta))

        return ImagenResult(response=response, variants=tuple(variants), metadata=base_meta)
