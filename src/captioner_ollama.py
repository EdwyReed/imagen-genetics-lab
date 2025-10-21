from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from .adapters import smart_adapt
from .schema import EnforceConfig, OllamaConfig


def build_payload(options: Dict[str, Dict[str, object]], required_terms: List[str]) -> Dict[str, object]:
    character = options["character"]
    pose = options["pose"]
    action = options["action"]
    style = options["style"]
    clothes = options["clothes"]

    character_desc = character.get("summary") or character.get("name")
    pose_desc = pose.get("summary") or pose.get("description")
    action_desc = action.get("summary") or action.get("description")
    style_desc = style.get("summary") or style.get("lighting")
    extras = clothes.get("extras") or []
    clothes_desc = clothes.get("summary") or clothes.get("primary")
    if extras:
        clothes_desc = f"{clothes_desc} with {', '.join(extras)}"

    camera_angle = pose.get("camera_angle", "eye-level")
    framing = pose.get("framing", "mid")
    aspect_ratio = pose.get("aspect_ratio", "3:4")
    camera = f"{framing} framing, {camera_angle} angle, {aspect_ratio} aspect"

    lens = options.get("camera", {}).get("lens", "50mm lens")
    depth = options.get("camera", {}).get("depth", "shallow depth of field")

    mood_terms = action.get("mood") or []
    mood = ", ".join(mood_terms)

    payload = {
        "character": character_desc,
        "pose": pose_desc,
        "action": action_desc,
        "style": style_desc,
        "clothes": clothes_desc,
        "camera": f"{camera}, {lens}, {depth}",
        "mood": mood,
    }
    if required_terms:
        payload["required_terms"] = required_terms
    return payload


def _normalize_caption(text: str) -> str:
    words = text.strip().split()
    if len(words) > 60:
        words = words[:60]
    return " ".join(words)


def _missing_terms(text: str, required_terms: List[str]) -> List[str]:
    lower = text.lower()
    return [term for term in required_terms if term.lower() not in lower]


@dataclass(slots=True)
class CaptionResult:
    caption: str
    rewrites: int
    missing_terms: List[str]


class Captioner:
    def __init__(self, config: OllamaConfig, enforce: EnforceConfig, system_prompt: str, seed: int | None) -> None:
        self.config = config
        self.enforce = enforce
        self.system_prompt = system_prompt
        self.seed = seed

    def generate(self, payload: Dict[str, object]) -> CaptionResult:
        caption = smart_adapt.ollama_generate(
            host=self.config.host,
            model=self.config.model,
            system_prompt=self.system_prompt,
            payload=payload,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            timeout=self.config.timeout,
            seed=self.seed,
        )
        caption = _normalize_caption(caption)
        missing = _missing_terms(caption, self.enforce.required_terms)
        rewrites = 0
        while self.enforce.enabled and missing and rewrites < self.enforce.max_rewrite:
            caption = smart_adapt.enforce_once(
                host=self.config.host,
                model=self.config.model,
                system_prompt=self.system_prompt,
                payload=payload,
                base_caption=caption,
                temperature=max(0.1, self.config.temperature - 0.05),
                seed=self.seed,
            )
            rewrites += 1
            caption = _normalize_caption(caption)
            missing = _missing_terms(caption, self.enforce.required_terms)
        return CaptionResult(caption=caption, rewrites=rewrites, missing_terms=missing)


__all__ = ["Captioner", "CaptionResult", "build_payload"]
