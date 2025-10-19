from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from google import genai

from imagen_lab.bias.engine.simple import SimpleBiasEngine
from imagen_lab.caption.ollama import OllamaCaptionEngine, OllamaClient
from imagen_lab.caption.ollama.interfaces import CaptionEngineProtocol
from imagen_lab.catalog import Catalog
from imagen_lab.config import PipelineConfig
from imagen_lab.db.repo.interfaces import RepositoryProtocol
from imagen_lab.db.repo.sqlite import SQLiteRepository
from imagen_lab.ga.engine.adapter import DefaultGAEngine
from imagen_lab.ga.engine.interfaces import GAEngineProtocol
from imagen_lab.image.imagen.adapter import ImagenClientEngine
from imagen_lab.image.imagen.interfaces import ImagenEngineProtocol
from imagen_lab.image.store import ImageArtifactStore
from imagen_lab.prompting import DEFAULT_REQUIRED_TERMS, PromptComposer
from imagen_lab.scene.builder.interfaces import SceneBuilderProtocol
from imagen_lab.scene.builder.probabilistic import ProbabilisticSceneBuilder
from imagen_lab.scoring.core.engine import CoreScoringEngine
from imagen_lab.scoring.core.interfaces import ScoringEngineProtocol
from imagen_lab.style_guide import StyleGuide


@dataclass
class PipelineContainer:
    scene_builder: SceneBuilderProtocol
    caption_engine: CaptionEngineProtocol
    imagen_engine: ImagenEngineProtocol
    scoring_engine: ScoringEngineProtocol | None
    repository: RepositoryProtocol
    ga_engine: GAEngineProtocol
    catalog: Catalog
    required_terms: list[str]
    composer: PromptComposer


def _required_terms(config: PipelineConfig, catalog: Catalog) -> tuple[list[str], PromptComposer]:
    configured_terms = [term for term in config.prompting.required_terms if term]
    fallback_terms = configured_terms or list(DEFAULT_REQUIRED_TERMS)
    style = StyleGuide.from_catalog(catalog, fallback_terms)
    required = configured_terms or list(style.required_terms)
    composer = PromptComposer(style, required)
    return required, composer


def create_pipeline_container(
    config: PipelineConfig,
    *,
    output_dir: Optional[Path] = None,
    enable_scoring: bool = True,
) -> PipelineContainer:
    catalog = Catalog.load(config.paths.catalog)
    required_terms, composer = _required_terms(config, catalog)

    ollama_client = OllamaClient(base_url=config.ollama.url, model=config.ollama.model)
    caption_engine = OllamaCaptionEngine(composer=composer, client=ollama_client)

    imagen_client = genai.Client()
    imagen_engine = ImagenClientEngine(client=imagen_client, model=config.imagen.model)

    bias_engine = SimpleBiasEngine()
    scene_builder = ProbabilisticSceneBuilder(catalog=catalog, bias_engine=bias_engine)

    repository = SQLiteRepository(Path(config.paths.database))
    artifacts = ImageArtifactStore(output_dir or config.paths.output_dir)
    scoring_engine: ScoringEngineProtocol | None = CoreScoringEngine(
        repository=repository,
        artifacts=artifacts,
        temperatures=config.scoring.temperatures,
        compute_metrics=enable_scoring,
    )

    ga_engine = DefaultGAEngine(
        builder=scene_builder,
        catalog=catalog,
        caption_engine=caption_engine,
        imagen_engine=imagen_engine,
        scoring_engine=scoring_engine,
        feedback=None,
    )

    return PipelineContainer(
        scene_builder=scene_builder,
        caption_engine=caption_engine,
        imagen_engine=imagen_engine,
        scoring_engine=scoring_engine,
        repository=repository,
        ga_engine=ga_engine,
        catalog=catalog,
        required_terms=required_terms,
        composer=composer,
    )
