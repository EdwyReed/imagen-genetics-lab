from __future__ import annotations

import os
from typing import Any, Dict, Optional

try:  # pragma: no cover - runtime dependency probe
    from smart import enforce_once as _smart_enforce_once
    from smart import imagen_call as _smart_imagen_call
    from smart import ollama_generate as _smart_ollama_generate
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency handling
    _SMART_IMPORT_ERROR = exc
else:
    _SMART_IMPORT_ERROR = None


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
    if _SMART_IMPORT_ERROR is not None:
        raise RuntimeError("smart.py dependencies are missing") from _SMART_IMPORT_ERROR
    return _smart_ollama_generate(
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
) -> str:
    if _SMART_IMPORT_ERROR is not None:
        raise RuntimeError("smart.py dependencies are missing") from _SMART_IMPORT_ERROR
    return _smart_enforce_once(
        url=host,
        model=model,
        system_prompt=system_prompt,
        payload=payload,
        base_caption=base_caption,
        temperature=temperature,
        seed=seed,
    )


def imagen_call(
    model: str,
    caption: str,
    variants: int,
    aspect_ratio: str,
    person_mode: str,
    safety_filter: str,
):
    if _SMART_IMPORT_ERROR is not None:
        raise RuntimeError("smart.py dependencies are missing") from _SMART_IMPORT_ERROR
    try:
        from google import genai  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("google-genai package is required for Imagen generation") from exc
    api_key = os.getenv("IMAGEN_API_KEY")
    if not api_key:
        raise RuntimeError("IMAGEN_API_KEY environment variable is required for Imagen generation")
    endpoint = os.getenv("IMAGEN_ENDPOINT")
    client_kwargs: Dict[str, Any] = {"api_key": api_key}
    if endpoint:
        client_kwargs["base_url"] = endpoint
    client = genai.Client(**client_kwargs)
    return _smart_imagen_call(
        client,
        model_name=model,
        prompt=caption,
        aspect_ratio=aspect_ratio,
        variants=variants,
        person_mode=person_mode,
    )


__all__ = ["ollama_generate", "enforce_once", "imagen_call"]
