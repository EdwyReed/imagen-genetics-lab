from __future__ import annotations

import os
from typing import Any, Dict, Optional

from ..ollama_client import enforce_once as _enforce_once_impl
from ..ollama_client import ollama_generate as _ollama_generate_impl

try:  # pragma: no cover - runtime dependency probe
    from google import genai  # type: ignore
    from google.genai import types  # type: ignore
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency handling
    _GENAI_IMPORT_ERROR = exc
else:
    _GENAI_IMPORT_ERROR = None


def ollama_generate(
    host: str,
    model: str,
    system_prompt: str,
    payload: Dict[str, Any],
    temperature: float,
    top_p: float,
    timeout: int,
    seed: Optional[int],
) -> str:
    return _ollama_generate_impl(
        url=host,
        model=model,
        system_prompt=system_prompt,
        payload=payload,
        temperature=temperature,
        top_p=top_p,
        timeout=timeout,
        seed=seed,
    )


def enforce_once(
    host: str,
    model: str,
    system_prompt: str,
    payload: Dict[str, Any],
    base_caption: str,
    temperature: float,
    seed: Optional[int],
    timeout: Optional[int] = None,
) -> str:
    return _enforce_once_impl(
        url=host,
        model=model,
        system_prompt=system_prompt,
        payload=payload,
        base_caption=base_caption,
        temperature=temperature,
        seed=seed,
        timeout=timeout,
    )


def imagen_call(
    model: str,
    caption: str,
    variants: int,
    aspect_ratio: str,
    person_mode: str,
    safety_filter: str,
):
    if _GENAI_IMPORT_ERROR is not None:
        raise RuntimeError("google-genai package is required for Imagen generation") from _GENAI_IMPORT_ERROR
    api_key = os.getenv("IMAGEN_API_KEY")
    if not api_key:
        raise RuntimeError("IMAGEN_API_KEY environment variable is required for Imagen generation")
    endpoint = os.getenv("IMAGEN_ENDPOINT")
    client_kwargs: Dict[str, Any] = {"api_key": api_key}
    if endpoint:
        client_kwargs["base_url"] = endpoint
    client = genai.Client(**client_kwargs)
    config = types.GenerateImagesConfig(
        number_of_images=variants,
        aspect_ratio=aspect_ratio,
        person_generation=person_mode,
        safety_filter_level=safety_filter or "block_low_and_above",
        output_mime_type="image/jpeg",
        guidance_scale=0.5,
    )
    return client.models.generate_images(model=model, prompt=caption, config=config)


__all__ = ["ollama_generate", "enforce_once", "imagen_call"]
