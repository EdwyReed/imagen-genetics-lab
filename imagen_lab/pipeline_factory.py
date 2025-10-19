from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from google import genai

from scorer import DualScorer

from imagen_lab.caption.ollama import OllamaClient, OllamaCaptionEngine
from imagen_lab.caption.ollama.interfaces import CaptionEngineProtocol
from imagen_lab.config import PipelineConfig
from imagen_lab.embeddings import EmbeddingCache, EmbeddingHistoryConfig
from imagen_lab.scene_builder import SceneBuilder
from imagen_lab.scoring import DEFAULT_STYLE_WEIGHTS, WeightProfileTable
from imagen_lab.learning import StyleFeedback
from imagen_lab.prompting import DEFAULT_REQUIRED_TERMS, PromptComposer
from imagen_lab.style_guide import StyleGuide
from imagen_lab.storage import ArtifactWriter, PromptLogger
from imagen_lab.catalog import Catalog
from imagen_lab.characters import CharacterLibrary

from .scene.builder.adapter import SceneBuilderAdapter
from .scene.builder.interfaces import SceneBuilderProtocol
from .image.imagen.adapter import ImagenClientEngine
from .image.imagen.interfaces import ImagenEngineProtocol
from .scoring.core.adapter import ScoringEngineAdapter
from .scoring.core.interfaces import ScoringEngineProtocol
from .db.repo.interfaces import RepositoryProtocol
from .db.repo.sqlite import SQLiteRepository
from .ga.engine.adapter import DefaultGAEngine
from .ga.engine.interfaces import GAEngineProtocol


@dataclass
class PipelineContainer:
    scorer: DualScorer | None
    scene_builder: SceneBuilderProtocol
    caption_engine: CaptionEngineProtocol
    imagen_engine: ImagenEngineProtocol
    scoring_engine: ScoringEngineProtocol
    repository: RepositoryProtocol
    ga_engine: GAEngineProtocol
    catalog: Catalog
    options_catalog: Catalog | None
    characters: CharacterLibrary | None
    writer: ArtifactWriter
    client: genai.Client
    feedback: StyleFeedback | None
    style: StyleGuide
    required_terms: list[str]
    composer: PromptComposer


def create_pipeline_container(
    config: PipelineConfig,
    *,
    output_dir: Optional[Path] = None,
    enable_scoring: bool = True,
) -> PipelineContainer:
    history_cfg = EmbeddingHistoryConfig(
        enabled=config.history.enabled,
        max_embeddings=config.history.max_embeddings,
    )
    history_cache = EmbeddingCache(history_cfg)

    weight_table = None
    profiles_path = getattr(config.scoring, "weight_profiles_path", None)
    if profiles_path:
        defaults = config.scoring.weights or DEFAULT_STYLE_WEIGHTS
        weight_table = WeightProfileTable.load(
            profiles_path,
            defaults=defaults,
            create=True,
        )

    scorer: DualScorer | None = None
    if enable_scoring:
        scorer = DualScorer(
            device=config.scoring.device,
            batch=config.scoring.batch_size,
            db_path=config.paths.database,
            jsonl_path=config.paths.scores_jsonl,
            weights=config.scoring.weights,
            tau=config.scoring.tau,
            cal_style=config.scoring.cal_style,
            cal_illu=config.scoring.cal_illu,
            auto_weights=config.scoring.auto_weights.as_dict(),
            weight_table=weight_table,
            weight_profile=getattr(config.scoring, "weight_profile", "default"),
            persist_profile_updates=getattr(config.scoring, "persist_profile_updates", False),
            composition_enabled=getattr(config.scoring, "composition_metrics", True),
        )
        setattr(scorer, "embedding_cache", history_cache)

    catalog = Catalog.load(config.paths.catalog)
    options_catalog = None
    options_path = getattr(config.paths, "options_catalog", None)
    if options_path:
        try:
            options_catalog = Catalog.load(options_path)
        except FileNotFoundError:
            options_catalog = None

    character_library: CharacterLibrary | None = None
    char_path = getattr(config.paths, "character_catalog", None)
    if char_path:
        try:
            character_library = CharacterLibrary.load(char_path)
        except FileNotFoundError:
            character_library = None

    configured_terms = [t for t in config.prompting.required_terms if t]
    fallback_terms = configured_terms or list(DEFAULT_REQUIRED_TERMS)
    style = StyleGuide.from_catalog(catalog, fallback_terms)
    required_terms = configured_terms or list(style.required_terms)
    composer = PromptComposer(style, required_terms)

    catalog_dict = catalog.to_dict()
    default_character = catalog_dict.get("default_character")
    variant_defaults: dict[str, str] = {}
    for variant in catalog_dict.get("brand_variants", []):
        if isinstance(variant, dict):
            variant_id = variant.get("id")
            char_id = variant.get("default_character")
            if (
                isinstance(variant_id, str)
                and variant_id.strip()
                and isinstance(char_id, str)
                and char_id.strip()
            ):
                variant_defaults[variant_id.strip()] = char_id.strip()

    scene_builder = SceneBuilderAdapter(
        SceneBuilder(
            catalog,
            required_terms=required_terms,
            template_ids=config.prompting.template_ids,
            character_library=character_library,
            default_character=default_character if isinstance(default_character, str) else None,
            variant_character_defaults=variant_defaults,
        )
    )

    logger = PromptLogger(config.paths.database)
    writer = ArtifactWriter(output_dir or config.paths.output_dir)
    imagen_client = genai.Client()

    feedback = StyleFeedback(config.feedback) if config.feedback.enabled else None

    ollama_client = OllamaClient(
        base_url=config.ollama.url,
        model=config.ollama.model,
    )
    caption_engine = OllamaCaptionEngine(
        composer=composer,
        client=ollama_client,
    )

    imagen_engine = ImagenClientEngine(
        client=imagen_client,
        model=config.imagen.model,
    )

    scoring_engine = ScoringEngineAdapter(
        writer=writer,
        logger=logger,
        scorer=scorer,
    )

    repository = SQLiteRepository(Path(config.paths.database))

    ga_engine = DefaultGAEngine(
        builder=scene_builder,
        catalog=catalog,
        caption_engine=caption_engine,
        imagen_engine=imagen_engine,
        scoring_engine=scoring_engine,
        feedback=feedback,
    )

    return PipelineContainer(
        scorer=scorer,
        scene_builder=scene_builder,
        caption_engine=caption_engine,
        imagen_engine=imagen_engine,
        scoring_engine=scoring_engine,
        repository=repository,
        ga_engine=ga_engine,
        catalog=catalog,
        options_catalog=options_catalog,
        characters=character_library,
        writer=writer,
        client=imagen_client,
        feedback=feedback,
        style=style,
        required_terms=required_terms,
        composer=composer,
    )
