from __future__ import annotations

import hashlib
import json
from typing import Iterable, List, Sequence

import requests
from google.genai import types

from .randomization import clamp
from .style_guide import StyleGuide


DEFAULT_REQUIRED_TERMS = ["illustration", "watercolor", "glossy", "paper", "pastel"]


def system_prompt_for(style: StyleGuide, sfw_level: float) -> str:
    sfw_level = clamp(sfw_level, 0.0, 1.0)
    tone_desc = (
        "wholesome and innocent" if sfw_level < 0.25 else "flirty but tasteful" if sfw_level < 0.7 else "bold adult tone"
    )
    context_lines = style.context_lines()
    context_block = "\n".join(context_lines)
    if context_block:
        context_block += "\n\n"

    required_line = ""
    if style.required_terms:
        required_line = (
            "Mandatory words (use naturally): "
            + ", ".join(style.required_terms)
            + ".\n"
        )

    return (
        f"You are a professional caption writer for the {style.brand} art catalog.\n\n"
        "Write one natural English caption (18–60 words) describing an illustration that embodies "
        f"{style.aesthetic}.\n"
        "Include: model, pose, wardrobe, accessories; camera angle and framing ratio; lighting; background; mood. "
        f"Keep it SFW-level proportional to {sfw_level:.2f} ({tone_desc}).\n\n"
        + context_block
        + required_line
        + "No bullet lists. One or two sentences. Cinematic, coherent, grounded in the provided JSON payload.\n"
        "The JSON payload includes style_profile (weights, boost/cooldown lists), scene_summary, and feedback_notes. "
        "Read them carefully, emphasize boost components, ease off cooldown components, and align the caption with scene_summary and feedback notes."
    )


def enforce_bounds(text: str, mn: int, mx: int) -> str:
    words = text.split()
    if len(words) > mx:
        words = words[:mx]
    return " ".join(words)


def ollama_generate(
    url: str,
    model: str,
    system_prompt: str,
    payload: dict,
    temperature: float = 0.55,
    top_p: float = 0.9,
    seed: int | None = None,
    timeout: int = 30,
) -> str:
    prompt = system_prompt.strip() + "\n\n" + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    options = {"temperature": float(temperature), "top_p": float(top_p), "repeat_penalty": 1.05}
    if seed is not None:
        options["seed"] = int(seed)
    response = requests.post(
        f"{url.rstrip('/')}/api/generate",
        json={"model": model, "prompt": prompt, "options": options, "stream": False, "raw": False},
        timeout=timeout,
    )
    response.raise_for_status()
    text = response.json().get("response", "").strip()
    text = " ".join(text.split()).replace("..", ".")
    return text


def needs_enforcement(text: str, required_terms: Sequence[str] = DEFAULT_REQUIRED_TERMS) -> List[str]:
    lower = text.lower()
    missing = [term for term in required_terms if term.lower() not in lower]
    return missing


def _format_terms_sentence(terms: Sequence[str]) -> str:
    unique: List[str] = []
    seen: set[str] = set()
    for term in terms:
        lowered = term.lower()
        if lowered in seen:
            continue
        unique.append(term)
        seen.add(lowered)
    if not unique:
        return ""
    if len(unique) == 1:
        return unique[0]
    if len(unique) == 2:
        return f"{unique[0]} and {unique[1]}"
    return f"{', '.join(unique[:-1])}, and {unique[-1]}"


def append_required_terms(
    text: str,
    required_terms: Iterable[str] = DEFAULT_REQUIRED_TERMS,
    *,
    max_words: int | None = None,
) -> str:
    """Ensure ``text`` contains each of ``required_terms``.

    If Ollama fails to include one or more mandatory style descriptors,
    fall back to appending a short sentence that mentions every missing
    term.  When ``max_words`` is provided, the result is trimmed from the
    main caption (never from the fallback sentence) to honor the word
    budget while still guaranteeing that the style hints appear in the
    final Imagen prompt.
    """

    required_list = [term for term in required_terms if term]
    if not required_list:
        return text.strip()

    missing = needs_enforcement(text, required_list)
    if not missing:
        return text.strip()

    base = text.strip().rstrip(" .")
    if base:
        base += "."

    sentence_core = _format_terms_sentence(missing)
    addition = "This artwork highlights " + sentence_core
    if not addition.endswith("."):
        addition += "."

    base_words = base.split()
    addition_words = addition.split()

    if max_words is not None and max_words > 0:
        total_words = len(base_words) + len(addition_words)
        if total_words > max_words:
            remove = total_words - max_words
            if remove >= len(base_words):
                base_words = []
            else:
                base_words = base_words[: len(base_words) - remove]

    combined = base_words + addition_words
    return " ".join(combined).strip()


def enforce_once(
    url: str,
    model: str,
    system_prompt: str,
    payload: dict,
    base_caption: str,
    required_terms: Sequence[str] = DEFAULT_REQUIRED_TERMS,
    temperature: float = 0.5,
    seed: int | None = None,
) -> str:
    missing = needs_enforcement(base_caption, required_terms)
    if not missing:
        return base_caption
    enforce_sys = (
        system_prompt
        + "\n\nRewrite the caption naturally (18–60 words) and include the missing words: "
        + ", ".join(missing)
        + ". Keep it one or two sentences."
    )
    return ollama_generate(url, model, enforce_sys, payload, temperature=temperature, seed=seed)


def imagen_call(
    client,
    model_name: str,
    prompt: str,
    aspect_ratio: str,
    variants: int,
    person_mode: str,
    guidance_scale: float = 0.5,
):
    cfg = types.GenerateImagesConfig(
        number_of_images=int(variants),
        aspect_ratio=aspect_ratio,
        person_generation=person_mode,
        safety_filter_level="block_low_and_above",
        output_mime_type="image/jpeg",
        guidance_scale=guidance_scale,
    )
    return client.models.generate_images(model=model_name, prompt=prompt, config=cfg)


def system_prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:12]
