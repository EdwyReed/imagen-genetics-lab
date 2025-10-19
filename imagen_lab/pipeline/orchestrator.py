"""High-level orchestration of the prompt → caption → image → scoring pipeline."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Dict, Optional

from ..caption.ollama import CaptionService
from ..db.repository import Repository
from ..image.imagen import ImagenService
from ..io.json_loader import JsonLoader
from ..scene.builder import PrePrompt, SceneBuilder
from ..scoring.core import ScoringEngine
from ..weights.manager import ProfileVector, WeightManager
from .configuration import PipelineConfig

__all__ = ["Pipeline", "PipelineResult"]


@dataclass(frozen=True)
class PipelineResult:
    run_id: str
    prompt_id: str
    pre_prompt: PrePrompt
    caption: str
    image_path: str
    conflicts: tuple


class Pipeline:
    """Coordinates services for a single run."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.loader = JsonLoader(config.styles_path, config.characters_path)
        self.repository = Repository(config.storage.database_path)
        self.caption_service = CaptionService(model_name=config.runtime.ollama_model)
        self.imagen_service = ImagenService(config.storage.artifacts_path)
        self.scoring_engine = ScoringEngine(config.aggregators)
        self.weight_manager = WeightManager(
            config.macro_weights,
            config.meso_overrides,
            profile=config.profile_defaults,
        )
        self._apply_stored_profile()

    def _apply_stored_profile(self) -> None:
        stored = self.repository.load_profile(self.config.profile_id)
        if not stored:
            return
        self.weight_manager.merge_profile(ProfileVector(
            macro=stored.get("macro", {}),
            meso={key: float(value) for key, value in stored.get("meso", {}).items()},
        ))

    def run(self, *, session_id: Optional[str] = None) -> PipelineResult:
        run_id = session_id or f"run_{uuid.uuid4().hex[:8]}"
        styles = self.loader.load_styles()
        characters = self.loader.load_characters()
        style = styles[self.config.style_preset]
        character = characters.get(self.config.character_preset or "") if self.config.character_preset else None

        macro, meso, conflicts = self.weight_manager.blend(style, character)
        gene_biases = self.repository.load_gene_biases()
        builder = SceneBuilder(style, character=character, gene_biases=gene_biases)
        pre_prompt = builder.build(macro, meso)
        caption = self.caption_service.generate(pre_prompt)
        image = self.imagen_service.generate(caption)

        score = None
        if not self.config.runtime.dry_run_no_scoring:
            score = self.scoring_engine.score(pre_prompt, caption)
            self.weight_manager.update_from_scores(score)
            profile_snapshot = ProfileVector(
                macro=self.weight_manager.profile.macro,
                meso=self.weight_manager.profile.meso,
            )
            self.repository.save_profile(self.config.profile_id, {
                "macro": profile_snapshot.macro,
                "meso": profile_snapshot.meso,
            })
            self.repository.record_gene_statistics(
                pre_prompt.selected_genes,
                score.fitness.get("fitness_visual", 0.0),
            )
        self.repository.record_run(run_id, {
            "style": pre_prompt.style_id,
            "character": pre_prompt.character_id,
            "macro": macro,
            "meso": meso,
        }, self.config.profile_id)
        prompt_id = f"prompt_{uuid.uuid4().hex[:8]}"
        self.repository.record_prompt(prompt_id, run_id, pre_prompt, caption, image.image_path, score)
        return PipelineResult(
            run_id=run_id,
            prompt_id=prompt_id,
            pre_prompt=pre_prompt,
            caption=caption,
            image_path=str(image.image_path),
            conflicts=conflicts,
        )

    def close(self) -> None:
        self.repository.close()
