from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

from imagen_lab.prompting import PromptComposer, system_prompt_hash

from .adapter import OllamaClient
from .contract import build_caption_payload
from .interfaces import CaptionEngineProtocol, CaptionRequest
from .types import CaptionResult

from imagen_lab.characters import character_keywords


def _character_terms(scene) -> tuple[str, ...]:
    payload = scene.ollama_payload() if hasattr(scene, "ollama_payload") else getattr(scene, "payload", {})
    character: Mapping[str, object] | None = None
    if isinstance(payload, Mapping):
        maybe = payload.get("character")
        if isinstance(maybe, Mapping):
            character = maybe
    if character is None:
        direct = getattr(scene, "character", None)
        if isinstance(direct, Mapping):
            character = direct
    if character is None:
        raw = getattr(scene, "raw", None)
        maybe_raw = getattr(raw, "character", None)
        if isinstance(maybe_raw, Mapping):
            character = maybe_raw
    if character is None:
        return ()
    return tuple(character_keywords(character))


@dataclass
class OllamaCaptionEngine(CaptionEngineProtocol):
    composer: PromptComposer
    client: OllamaClient

    def __post_init__(self) -> None:
        self._system_prompts: dict[float, str] = {}
        self._system_hashes: dict[float, str] = {}

    def _system_prompt(self, sfw_level: float) -> str:
        prompt = self._system_prompts.get(sfw_level)
        if prompt is None:
            prompt = self.composer.system_prompt(sfw_level)
            self._system_prompts[sfw_level] = prompt
        return prompt

    def _system_hash(self, sfw_level: float) -> str:
        if sfw_level not in self._system_hashes:
            self._system_hashes[sfw_level] = system_prompt_hash(self._system_prompt(sfw_level))
        return self._system_hashes[sfw_level]

    def _enforcement_temperature(self, temperature: float) -> float:
        return max(0.45, temperature - 0.05)

    def _enforcement_prompt(self, system_prompt: str, missing: list[str]) -> str:
        if not missing:
            return system_prompt
        return (
            system_prompt
            + "\n\nRewrite the caption naturally (18–60 words) and include the missing words: "
            + ", ".join(missing)
            + ". Keep it one or two sentences."
        )

    def generate(self, request: CaptionRequest) -> CaptionResult:
        payload = build_caption_payload(request.scene)
        extra_terms = _character_terms(request.scene)
        system_prompt = self._system_prompt(request.sfw_level)
        caption = self.client.generate(
            system_prompt,
            payload,
            temperature=request.temperature,
            top_p=request.top_p,
            seed=request.seed,
        )
        enforced = False
        missing = self.composer.missing_terms(caption, extra_terms)
        if missing:
            enforced = True
            caption = self.client.generate(
                self._enforcement_prompt(system_prompt, missing),
                payload,
                temperature=self._enforcement_temperature(request.temperature),
                top_p=request.top_p,
                seed=request.seed,
            )
            missing = self.composer.missing_terms(caption, extra_terms)
            if missing:
                caption = self.composer.append_required_terms(
                    caption,
                    max_words=int(request.scene.caption_bounds.get("max_words", 60) or 60),
                    extra_terms=missing,
                )
        final_prompt = self.composer.final_prompt(
            caption,
            request.scene.caption_bounds,
            extra_terms=extra_terms,
        )
        return CaptionResult(
            caption=caption,
            final_prompt=final_prompt,
            enforced=enforced,
            bounds=request.scene.caption_bounds,
            system_hash=self._system_hash(request.sfw_level),
        )
