"""Caption and image generation workflow helpers."""
from __future__ import annotations

import json
import logging
import subprocess
import time
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Mapping, MutableMapping, Sequence

import requests

from .build import BuildResult
from .system_prompt import SystemPromptBundle

LOGGER = logging.getLogger("imagen.generation")


class GenerationError(RuntimeError):
    """Raised when caption or image generation fails."""


def _to_serializable(value: Any) -> Any:
    """Convert arbitrary values into JSON-serialisable structures."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _to_serializable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_serializable(item) for item in value]
    return str(value)


def build_ollama_payload(result: BuildResult, *, weights: Mapping[str, float], stage_temperature: float) -> Dict[str, Any]:
    """Prepare the JSON payload sent to Ollama."""

    payload = {
        "scene": _to_serializable(result.scene),
        "meta": _to_serializable(result.meta),
        "weights": _to_serializable(weights),
        "system": {
            "required_terms": list(result.system_prompt.required_terms),
            "style_tokens": list(result.system_prompt.style_tokens),
            "rule_injections": list(result.system_prompt.rule_injections),
        },
        "stage_temperature": float(stage_temperature),
    }
    return payload


def _trim_words(text: str, max_words: int | None) -> str:
    if not text:
        return ""
    if max_words is None or max_words <= 0:
        return " ".join(text.split())
    words = text.split()
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words])


def _missing_terms(text: str, terms: Iterable[str]) -> list[str]:
    lowered = text.lower()
    missing: list[str] = []
    for term in terms:
        cleaned = (term or "").strip()
        if not cleaned:
            continue
        if cleaned.lower() not in lowered:
            missing.append(cleaned)
    return missing


def final_prompt(
    caption: str,
    bundle: SystemPromptBundle,
    *,
    extra_terms: Sequence[str] | None = None,
    min_words: int = 0,
    max_words: int | None = None,
) -> str:
    """Compose the final Imagen prompt with required terms and rule hints."""

    base = caption.strip()
    if not base:
        raise GenerationError("Caption is empty; cannot construct final prompt")

    combined_terms = list(bundle.required_terms)
    if extra_terms:
        combined_terms.extend(extra_terms)
    combined_terms = [term for term in combined_terms if term]
    missing = _missing_terms(base, combined_terms)

    final_lines: list[str] = []

    words = base.split()
    if min_words > 0 and len(words) < min_words:
        # Pad softly by repeating the caption so Imagen receives enough context
        while len(words) < min_words:
            words.extend(base.split())
        base = " ".join(words[:max(min_words, len(words))])
    final_lines.append(_trim_words(base, max_words))

    if missing:
        addition = "Include the following concepts: " + ", ".join(missing) + "."
        final_lines.append(addition)

    if bundle.style_tokens:
        styles = ", ".join(bundle.style_tokens)
        final_lines.append(f"Rendered with emphasis on {styles}.")

    if bundle.rule_injections:
        for rule in bundle.rule_injections:
            cleaned = (rule or "").strip()
            if cleaned:
                final_lines.append(cleaned)

    prompt = " ".join(line.strip() for line in final_lines if line.strip())
    return " ".join(prompt.split())


@dataclass
class OllamaOptions:
    base_url: str
    model: str
    temperature: float
    top_p: float
    timeout: float
    seed: int | None = None
    unload_model: bool = True
    startup_timeout: float = 30.0


class OllamaModelManager:
    """Ensure an Ollama model is available locally."""

    def __init__(self, model: str, *, unload: bool = True) -> None:
        self.model = model
        self.unload = unload
        self._pulled = False

    def __enter__(self) -> "OllamaModelManager":
        if not self._has_model():
            LOGGER.info("pulling Ollama model '%s'", self.model)
            self._run(["ollama", "pull", self.model])
            self._pulled = True
        else:
            LOGGER.debug("Ollama model '%s' already available", self.model)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # pragma: no cover - subprocess cleanup
        if self.unload and (self._pulled or self._should_force_remove()):
            try:
                LOGGER.info("unloading Ollama model '%s'", self.model)
                self._run(["ollama", "rm", self.model])
            except Exception as error:
                LOGGER.warning("failed to unload Ollama model '%s': %s", self.model, error)

    def _should_force_remove(self) -> bool:
        return False

    def _has_model(self) -> bool:
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                check=True,
                text=True,
            )
        except (FileNotFoundError, subprocess.CalledProcessError):
            return False
        return any(self.model in line for line in result.stdout.splitlines())

    @staticmethod
    def _run(cmd: Sequence[str]) -> None:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


@contextmanager
def ensure_ollama_service(base_url: str, *, timeout: float = 30.0) -> Iterator[None]:
    """Ensure the Ollama HTTP service is reachable."""

    start = time.time()
    while True:
        try:
            requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=2)
            yield
            break
        except requests.RequestException:
            if time.time() - start > timeout:
                raise GenerationError(
                    f"Ollama service at {base_url} is unreachable after {timeout} seconds"
                )
            time.sleep(1.0)


def generate_caption(
    *,
    system_prompt: str,
    payload: Mapping[str, Any],
    options: OllamaOptions,
) -> str:
    """Send a caption request to Ollama."""

    prompt = system_prompt.strip() + "\n\n" + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    request_payload = {
        "model": options.model,
        "prompt": prompt,
        "stream": False,
        "raw": False,
        "options": {
            "temperature": float(options.temperature),
            "top_p": float(options.top_p),
            "repeat_penalty": 1.05,
        },
    }
    if options.seed is not None:
        request_payload["options"]["seed"] = int(options.seed)
    response = requests.post(
        f"{options.base_url.rstrip('/')}/api/generate",
        json=request_payload,
        timeout=options.timeout,
    )
    response.raise_for_status()
    data = response.json()
    text = str(data.get("response", "")).strip()
    if not text:
        raise GenerationError("Ollama returned an empty caption")
    return " ".join(text.split())


def _coerce_bytes(blob: Any) -> bytes | None:
    if isinstance(blob, (bytes, bytearray)):
        return bytes(blob)
    if isinstance(blob, str):
        import base64
        import binascii

        try:
            return base64.b64decode(blob)
        except (ValueError, binascii.Error):
            return None
    return None


@dataclass
class ImagenOptions:
    model: str
    aspect_ratio: str
    variants: int
    person_mode: str
    guidance_scale: float


@dataclass
class ImagenVariant:
    index: int
    image_bytes: bytes
    metadata: Dict[str, Any]


@dataclass
class ImagenResult:
    variants: Sequence[ImagenVariant]
    metadata: Dict[str, Any]


class ImagenRuntime:
    """Wrapper around the Google Imagen client."""

    def __init__(self, options: ImagenOptions) -> None:
        from google import genai

        self.options = options
        self._client = genai.Client()

    @property
    def request_payload(self) -> Dict[str, Any]:
        return {
            "model": self.options.model,
            "aspect_ratio": self.options.aspect_ratio,
            "variants": int(self.options.variants),
            "person_mode": self.options.person_mode,
            "guidance_scale": float(self.options.guidance_scale),
        }

    def generate(self, prompt: str) -> ImagenResult:
        from google.genai import types

        cfg = types.GenerateImagesConfig(
            number_of_images=int(self.options.variants),
            aspect_ratio=self.options.aspect_ratio,
            person_generation=self.options.person_mode,
            safety_filter_level="block_low_and_above",
            output_mime_type="image/jpeg",
            guidance_scale=self.options.guidance_scale,
        )
        response = self._client.models.generate_images(
            model=self.options.model,
            prompt=prompt,
            config=cfg,
        )
        metadata: Dict[str, Any] = {}
        version = getattr(response, "model_version", None) or getattr(response, "modelVersion", None)
        if version:
            metadata["imagen_version"] = version
        root_meta = getattr(response, "metadata", None)
        if isinstance(root_meta, Mapping):
            metadata.update(_to_serializable(root_meta))
        elif hasattr(root_meta, "to_dict"):
            try:
                metadata.update(root_meta.to_dict())  # type: ignore[attr-defined]
            except Exception:  # pragma: no cover - defensive
                pass
        variants: list[ImagenVariant] = []
        for idx, generated in enumerate(getattr(response, "generated_images", []), start=1):
            image_obj = getattr(generated, "image", None)
            blob = None
            if image_obj is not None:
                blob = _coerce_bytes(getattr(image_obj, "image_bytes", None) or getattr(image_obj, "imageBytes", None))
            if blob is None:
                blob = _coerce_bytes(getattr(generated, "image", None))
            if not blob:
                continue
            variant_meta: Dict[str, Any] = {}
            raw_meta = getattr(generated, "metadata", None)
            if isinstance(raw_meta, Mapping):
                variant_meta.update(_to_serializable(raw_meta))
            elif hasattr(raw_meta, "to_dict"):
                try:
                    variant_meta.update(raw_meta.to_dict())  # type: ignore[attr-defined]
                except Exception:  # pragma: no cover - defensive
                    pass
            variants.append(ImagenVariant(index=idx, image_bytes=blob, metadata=variant_meta))
        return ImagenResult(variants=variants, metadata=metadata)


@dataclass
class GenerationArtifacts:
    caption: str
    final_prompt: str
    ollama_request: Dict[str, Any]
    imagen_request: Dict[str, Any]
    imagen_metadata: Dict[str, Any]
    variants: Sequence[Dict[str, Any]]


class ImageGenerationWorkflow:
    """High-level helper that coordinates caption and image generation."""

    def __init__(
        self,
        *,
        run_id: str,
        output_dir: Path,
        ollama: OllamaOptions,
        imagen: ImagenOptions,
        weights: Mapping[str, float],
        min_words: int,
        max_words: int,
    ) -> None:
        self.run_id = run_id
        self.output_dir = output_dir
        self.ollama = ollama
        self.imagen_options = imagen
        self.weights = dict(weights)
        self.min_words = min_words
        self.max_words = max_words
        self._stack = ExitStack()
        self._ollama_manager: OllamaModelManager | None = None
        self._imagen_runtime: ImagenRuntime | None = None
        self._images_dir = (self.output_dir / "images" / run_id).resolve()

    def __enter__(self) -> "ImageGenerationWorkflow":
        self._images_dir.mkdir(parents=True, exist_ok=True)
        self._stack.enter_context(ensure_ollama_service(self.ollama.base_url, timeout=self.ollama.startup_timeout))
        self._ollama_manager = self._stack.enter_context(
            OllamaModelManager(self.ollama.model, unload=self.ollama.unload_model)
        )
        self._imagen_runtime = ImagenRuntime(self.imagen_options)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # pragma: no cover - ExitStack handles cleanup
        self._stack.close()

    def _ensure_runtime(self) -> ImagenRuntime:
        if not self._imagen_runtime:
            raise RuntimeError("Imagen runtime not initialised")
        return self._imagen_runtime

    def _temperature_for_stage(self, stage_temperature: float | None) -> float:
        if stage_temperature is None or stage_temperature <= 0:
            return self.ollama.temperature
        return stage_temperature

    def process(
        self,
        *,
        stage_id: str,
        cycle: int,
        record_id: str,
        result: BuildResult,
        stage_temperature: float,
    ) -> GenerationArtifacts:
        payload = build_ollama_payload(result, weights=self.weights, stage_temperature=stage_temperature)
        temperature = self._temperature_for_stage(stage_temperature)
        LOGGER.info(
            "requesting caption from Ollama model=%s temperature=%.3f top_p=%.3f",
            self.ollama.model,
            temperature,
            self.ollama.top_p,
        )
        caption = generate_caption(
            system_prompt=result.system_prompt.text,
            payload=payload,
            options=OllamaOptions(
                base_url=self.ollama.base_url,
                model=self.ollama.model,
                temperature=temperature,
                top_p=self.ollama.top_p,
                timeout=self.ollama.timeout,
                seed=self.ollama.seed,
                unload_model=self.ollama.unload_model,
                startup_timeout=self.ollama.startup_timeout,
            ),
        )
        LOGGER.debug("caption received: %s", caption)

        final = final_prompt(
            caption,
            result.system_prompt,
            min_words=self.min_words,
            max_words=self.max_words,
        )
        LOGGER.info("final prompt prepared (length=%d words)", len(final.split()))

        imagen_runtime = self._ensure_runtime()
        imagen_result = imagen_runtime.generate(final)
        if not imagen_result.variants:
            raise GenerationError("Imagen returned no variants")

        variants_summary = []
        for variant in imagen_result.variants:
            variant_meta = self._store_variant(
                stage_id=stage_id,
                cycle=cycle,
                record_id=record_id,
                caption=caption,
                final_prompt_text=final,
                result=result,
                payload=payload,
                imagen_metadata=imagen_result.metadata,
                variant=variant,
            )
            variants_summary.append(variant_meta)

        ollama_request = {
            "url": self.ollama.base_url,
            "model": self.ollama.model,
            "temperature": temperature,
            "top_p": self.ollama.top_p,
            "seed": self.ollama.seed,
            "payload": payload,
            "system_prompt": result.system_prompt.text,
        }

        imagen_request = imagen_runtime.request_payload | {"prompt_length": len(final.split())}

        return GenerationArtifacts(
            caption=caption,
            final_prompt=final,
            ollama_request=ollama_request,
            imagen_request=imagen_request,
            imagen_metadata=imagen_result.metadata,
            variants=variants_summary,
        )

    def _store_variant(
        self,
        *,
        stage_id: str,
        cycle: int,
        record_id: str,
        caption: str,
        final_prompt_text: str,
        result: BuildResult,
        payload: Mapping[str, Any],
        imagen_metadata: Mapping[str, Any],
        variant: ImagenVariant,
    ) -> Dict[str, Any]:
        image_name = f"{self.run_id}_{stage_id}_{record_id}_v{variant.index:02d}.jpg"
        image_path = self._images_dir / image_name
        txt_path = image_path.with_suffix(".txt")
        json_path = image_path.with_suffix(".json")

        json_payload = {
            "run_id": self.run_id,
            "stage_id": stage_id,
            "cycle": cycle,
            "record_id": record_id,
            "variant_index": variant.index,
            "caption": caption,
            "final_prompt": final_prompt_text,
            "ollama": {
                "model": self.ollama.model,
                "url": self.ollama.base_url,
                "temperature": self.ollama.temperature,
                "top_p": self.ollama.top_p,
                "seed": self.ollama.seed,
                "weights": self.weights,
                "system_prompt": result.system_prompt.text,
            },
            "imagen": {
                "model": self.imagen_options.model,
                "aspect_ratio": self.imagen_options.aspect_ratio,
                "guidance_scale": self.imagen_options.guidance_scale,
                "person_mode": self.imagen_options.person_mode,
                "metadata": _to_serializable(imagen_metadata),
                "variant_metadata": _to_serializable(variant.metadata),
            },
            "scene": payload,
        }

        self._save_image_with_metadata(image_path, variant.image_bytes, json_payload)
        txt_path.write_text(final_prompt_text, encoding="utf-8")
        json_path.write_text(json.dumps(json_payload, ensure_ascii=False, indent=2), encoding="utf-8")

        return {
            "image": str(image_path.relative_to(self.output_dir)),
            "metadata": str(json_path.relative_to(self.output_dir)),
            "prompt": str(txt_path.relative_to(self.output_dir)),
            "variant_index": variant.index,
        }

    def _save_image_with_metadata(self, path: Path, image_bytes: bytes, metadata: Mapping[str, Any]) -> None:
        try:
            from PIL import Image
        except ImportError as exc:  # pragma: no cover - optional dependency
            LOGGER.warning("Pillow not installed; saving image without EXIF metadata: %s", exc)
            path.write_bytes(image_bytes)
            return

        image = Image.open(BytesIO(image_bytes))
        exif = image.getexif()
        description = json.dumps(metadata, ensure_ascii=False)
        exif[270] = description
        image.save(path, format="JPEG", exif=exif)


__all__ = [
    "GenerationArtifacts",
    "GenerationError",
    "ImageGenerationWorkflow",
    "ImagenOptions",
    "OllamaOptions",
    "build_ollama_payload",
    "final_prompt",
]

