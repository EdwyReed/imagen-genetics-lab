from __future__ import annotations

from imagen_pipeline.core.build import build_struct
from imagen_pipeline.core.constraints import ScenarioContext
from imagen_pipeline.core.evolve import Genome, mutate_genome
from imagen_pipeline.core.preferences import BiasConfig, LockSet
from imagen_pipeline.core.scenarios import Stage
from imagen_pipeline.core.selector import AssetSelector


def test_mutate_genome_respects_pinned_lock(base_assets, bias_config):
    selector = AssetSelector(base_assets, bias_config)
    stage = Stage(
        stage_id="evo",
        cycles=1,
        temperature=0.2,
        style_profile="retro_glossy",
        locks={},
        required_terms=["evolve"],
        inject_rules=["base_guideline"],
    )
    build = build_struct(
        assets=base_assets,
        selector=selector,
        bias=bias_config,
        stage=stage,
        default_style_profile="retro_glossy",
        style_token_limit=2,
    )
    genome = Genome.from_build(build)
    context = ScenarioContext(locks={"models": LockSet(allow=[genome.model])})
    mutated = mutate_genome(
        genome,
        selector=selector,
        assets=base_assets,
        context=context,
        mutation_rate=1.0,
    )
    assert mutated.model == genome.model
