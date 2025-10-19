from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

pytest.importorskip("yaml")

from imagen_lab.config import PipelineConfig, load_config


def test_pipeline_config_supports_storage_profiles(tmp_path: Path) -> None:
    data = {
        "schema_version": 2,
        "storage": {
            "database": "scores.sqlite",
            "scores_log": "scores.jsonl",
            "artifacts_dir": "output",
            "catalogs": {"primary": "catalogs/main.json"},
            "profiles": {"directory": "profiles", "active": "profiles/studio.json"},
        },
        "runtime": {
            "prompting": {"required_terms": [], "template_ids": []},
            "ollama": {"url": "http://localhost:11434", "model": "test", "temperature": 0.5, "top_p": 0.9},
            "imagen": {"model": "imagen-3.0-generate-002", "person_mode": "allow_adult"},
        },
        "scoring": {},
    }
    config = PipelineConfig.from_dict(data)
    assert config.paths.catalog == Path("catalogs/main.json")
    assert config.paths.profiles_dir == Path("profiles")
    assert config.paths.profile_template == Path("profiles/studio.json")
    assert config.bias.macro_weights == {}
    assert config.bias.rules == ()


def test_load_config_from_jsonc(tmp_path: Path) -> None:
    config_path = tmp_path / "config.jsonc"
    config_path.write_text(
        """
        {
          // storage block is minimal for this test
          "storage": {
            "catalogs": {"primary": "catalogs/all-together.json"}
          },
          "runtime": {
            "ollama": {"url": "http://localhost:11434", "model": "demo"},
            "imagen": {"model": "imagen", "person_mode": "allow_adult"}
          },
          "scoring": {}
        }
        """,
        encoding="utf-8",
    )

    config = load_config(config_path)
    assert isinstance(config, PipelineConfig)
    assert config.paths.catalog == Path("catalogs/all-together.json")
