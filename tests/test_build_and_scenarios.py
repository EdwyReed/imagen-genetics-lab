from __future__ import annotations

import json
from pathlib import Path

from imagen_pipeline.core.build import build_struct
from imagen_pipeline.core.preferences import BiasConfig
from imagen_pipeline.core.scenarios import ScenarioLoader, Stage
from imagen_pipeline.core.selector import AssetSelector


def test_scenario_two_stages_apply_metadata(tmp_path: Path, base_assets, bias_config):
    scenario_path = tmp_path / "two_stage.json"
    scenario_path.write_text(
        json.dumps(
            {
                "id": "demo",
                "stages": [
                    {
                        "stage_id": "stage_one",
                        "cycles": 1,
                        "temperature": 0.3,
                        "style_profile": "retro_glossy",
                        "required_terms": ["stage one term"],
                        "inject_rules": ["base_guideline"],
                    },
                    {
                        "stage_id": "stage_two",
                        "cycles": 1,
                        "temperature": 0.2,
                        "style_profile": "y2k",
                        "locks": {"models": {"allow": ["sdxl_base"]}},
                        "required_terms": ["stage two term"],
                        "inject_rules": ["no_vignette"],
                    },
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    loader = ScenarioLoader()
    scenario = loader.load(scenario_path)
    selector = AssetSelector(base_assets, bias_config)
    stages = list(scenario.stages)
    result_one = build_struct(
        assets=base_assets,
        selector=selector,
        bias=bias_config,
        stage=stages[0],
        default_style_profile="retro_glossy",
        style_token_limit=2,
    )
    assert result_one.meta["stage_id"] == "stage_one"
    assert "stage one term" in result_one.system_prompt.required_terms

    result_two = build_struct(
        assets=base_assets,
        selector=selector,
        bias=bias_config,
        stage=stages[1],
        default_style_profile="retro_glossy",
        style_token_limit=2,
    )
    assert result_two.meta["stage_id"] == "stage_two"
    assert result_two.meta["style_profile"] == "y2k"
    assert "no_vignette" in result_two.gene_ids["rules"]


def test_build_struct_gene_ids_and_meta(base_assets, bias_config):
    selector = AssetSelector(base_assets, bias_config)
    stage = Stage(
        stage_id="smoke",
        cycles=1,
        temperature=0.1,
        style_profile="retro_glossy",
        locks={},
        required_terms=["smoke"],
        inject_rules=["base_guideline"],
    )
    result = build_struct(
        assets=base_assets,
        selector=selector,
        bias=bias_config,
        stage=stage,
        default_style_profile="retro_glossy",
        style_token_limit=2,
    )
    assert set(result.gene_ids.keys()) == {
        "model",
        "characters",
        "wardrobe_set",
        "wardrobe_main",
        "pose",
        "palette",
        "lighting",
        "camera",
        "style_tokens",
        "rules",
    }
    assert result.meta["stage_id"] == "smoke"
    assert result.meta["run_cfg"]["temperature"] == 0.1
    assert result.meta["active_asset_packs"]
