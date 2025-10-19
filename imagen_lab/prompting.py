from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Iterable, List, Mapping, MutableMapping, Sequence

import requests
from google.genai import types

from .randomization import clamp
from .style_guide import StyleGuide

DEFAULT_REQUIRED_TERMS = ["illustration", "watercolor", "glossy", "paper", "pastel"]


@dataclass
class PromptComposer:
    """Build and post-process prompts for Ollama and Imagen."""

    style: StyleGuide
    required_terms: Sequence[str] | None = None

    def __post_init__(self) -> None:
        terms = list(self.required_terms or self.style.required_terms or DEFAULT_REQUIRED_TERMS)
        cleaned: list[str] = []
        seen: set[str] = set()
        for term in terms:
            if not isinstance(term, str):
                continue
            stripped = term.strip()
            if not stripped:
                continue
            lowered = stripped.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            cleaned.append(stripped)
        self.required_terms = tuple(cleaned)

    def _combined_terms(self, extra_terms: Iterable[str] | None = None) -> tuple[str, ...]:
        terms = list(self.required_terms)
        if extra_terms:
            for term in extra_terms:
                if not isinstance(term, str):
                    continue
                stripped = term.strip()
                if not stripped:
                    continue
                terms.append(stripped)
        cleaned: list[str] = []
        seen: set[str] = set()
        for term in terms:
            lowered = term.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            cleaned.append(term)
        return tuple(cleaned)

    # ------------------------------------------------------------------
    # Prompt building
    # ------------------------------------------------------------------
    def system_prompt(self, sfw_level: float) -> str:
        sfw_level = clamp(sfw_level, 0.0, 1.0)
        tone_desc = (
            "wholesome and innocent"
            if sfw_level < 0.25
            else "flirty but tasteful"
            if sfw_level < 0.7
            else "bold adult tone"
        )
        return (
            "Write one natural English caption (18–50 words) describing the illustration using the provided style, character, and scene facts. "
            f"Respect the SFW level {sfw_level:.2f} ({tone_desc}) and emphasise the most relevant top signals without inventing new details."
        )

    def missing_terms(self, text: str, extra_terms: Iterable[str] | None = None) -> List[str]:
        combined = self._combined_terms(extra_terms)
        return needs_enforcement(text, combined)

    def trim_to_limit(self, text: str, max_words: int) -> str:
        words = text.split()
        if max_words > 0 and len(words) > max_words:
            words = words[:max_words]
        return " ".join(words)

    def append_required_terms(
        self,
        text: str,
        *,
        max_words: int | None = None,
        extra_terms: Iterable[str] | None = None,
    ) -> str:
        combined = self._combined_terms(extra_terms)
        return append_required_terms(text, combined, max_words=max_words)

    def enforce_once(
        self,
        url: str,
        model: str,
        system_prompt: str,
        payload: Mapping[str, object] | MutableMapping[str, object],
        base_caption: str,
        *,
        temperature: float = 0.5,
        seed: int | None = None,
        extra_terms: Iterable[str] | None = None,
    ) -> str:
        missing = self.missing_terms(base_caption, extra_terms)
        if not missing:
            return base_caption
        enforce_sys = (
            system_prompt
            + "\n\nRewrite the caption naturally (18–60 words) and include the missing words: "
            + ", ".join(missing)
            + ". Keep it one or two sentences."
        )
        return ollama_generate(
            url,
            model,
            enforce_sys,
            dict(payload),
            temperature=temperature,
            seed=seed,
        )

    def final_prompt(
        self,
        caption: str,
        bounds: Mapping[str, object],
        *,
        extra_terms: Iterable[str] | None = None,
    ) -> str:
        max_words = int(bounds.get("max_words", 60) or 60)
        trimmed = self.trim_to_limit(caption, max_words)
        return self.append_required_terms(trimmed, max_words=max_words, extra_terms=extra_terms)


# ----------------------------------------------------------------------
# Compatibility helpers (module-level functions maintained for callers)
# ----------------------------------------------------------------------

def system_prompt_for(style: StyleGuide, sfw_level: float) -> str:
    return PromptComposer(style).system_prompt(sfw_level)


def enforce_bounds(text: str, mn: int, mx: int) -> str:  # pragma: no cover - deprecated
    _ = mn  # retained for backward compatibility
    words = text.split()
    if len(words) > mx:
        words = words[:mx]
    return " ".join(words)


def ollama_generate(
    url: str,
    model: str,
    system_prompt: str,
    payload: Mapping[str, object] | MutableMapping[str, object],
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
    terms = [term for term in required_terms if isinstance(term, str) and term]
    lower = text.lower()
    return [term for term in terms if term.lower() not in lower]


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
    """Ensure ``text`` contains each of ``required_terms``."""

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
    payload: Mapping[str, object] | MutableMapping[str, object],
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
    return ollama_generate(
        url,
        model,
        enforce_sys,
        payload,
        temperature=temperature,
        seed=seed,
    )


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
