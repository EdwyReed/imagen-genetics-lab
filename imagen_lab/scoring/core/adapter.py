from __future__ import annotations

from dataclasses import dataclass

from imagen_lab.storage import ArtifactWriter, PromptLogger, save_and_score
from scorer import DualScorer

from .interfaces import ScoringEngineProtocol, ScoringRequest, ScoringResult


@dataclass
class ScoringEngineAdapter(ScoringEngineProtocol):
    writer: ArtifactWriter
    logger: PromptLogger
    scorer: DualScorer | None

    def score(self, request: ScoringRequest) -> ScoringResult:
        batch = save_and_score(
            request.response,
            self.writer,
            self.logger,
            self.scorer,
            dict(request.meta),
            request.prompt,
            request.scene,
            request.session_id,
            gen=request.ga_context.get("gen"),
            indiv=request.ga_context.get("indiv"),
            w_style=request.weights.get("style", 0.0),
            w_nsfw=request.weights.get("nsfw", 0.0),
        )
        return ScoringResult(batch=batch)
