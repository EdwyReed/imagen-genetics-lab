"""Caption service that consumes pre-prompts and produces succinct captions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from ..scene.builder import PrePrompt

__all__ = ["CaptionService", "CaptionRequest"]


@dataclass(frozen=True)
class CaptionRequest:
    """High-level summary sent to the caption model."""

    scene_summary: str
    macro_controls: Mapping[str, float | str]
    meso_signals: Mapping[str, float]
    visual_axes: Mapping[str, str]


class CaptionService:
    """A deterministic caption generator used for local orchestration tests."""

    def __init__(self, model_name: str = "ollama-lite") -> None:
        self.model_name = model_name

    def build_request(self, pre_prompt: PrePrompt) -> CaptionRequest:
        macro_subset = {
            key: pre_prompt.macro_controls[key]
            for key in sorted(pre_prompt.macro_controls.keys())
            if key in {"sfw_level", "focus_mode", "gloss_priority", "coverage_target", "retro_authenticity"}
        }
        return CaptionRequest(
            scene_summary=pre_prompt.scene_summary,
            macro_controls=macro_subset,
            meso_signals=pre_prompt.meso_signals,
            visual_axes=pre_prompt.visual_axes,
        )

    def generate(self, pre_prompt: PrePrompt) -> str:
        """Generate a caption of at most two sentences."""

        request = self.build_request(pre_prompt)
        visual_components = ", ".join(
            f"{category} {value}" for category, value in request.visual_axes.items() if value
        )
        sentence_one = f"{request.scene_summary}.".replace("..", ".")
        sentence_two = (
            f"Focus on {request.macro_controls.get('focus_mode', 'style_first')} with gloss {request.macro_controls.get('gloss_priority', 0.5)}; {visual_components}."
        )
        combined = f"{sentence_one} {sentence_two}".strip()
        words = combined.split()
        if len(words) > 40:
            combined = " ".join(words[:40])
        return combined
