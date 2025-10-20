from __future__ import annotations

from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from imagen_pipeline.core.assets import AssetLibrary
from imagen_pipeline.core.preferences import BiasConfig
from imagen_pipeline.core.selector import AssetSelector


@pytest.fixture(scope="session")
def base_assets() -> AssetLibrary:
    root = Path("data/catalog").resolve()
    example = Path("data/packs/examples").resolve()
    return AssetLibrary([root, example], schema_dir=Path("schemas"))


@pytest.fixture()
def bias_config() -> BiasConfig:
    return BiasConfig()


@pytest.fixture()
def selector(base_assets: AssetLibrary, bias_config: BiasConfig) -> AssetSelector:
    return AssetSelector(base_assets, bias_config)
