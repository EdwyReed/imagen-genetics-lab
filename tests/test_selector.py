from __future__ import annotations

import json
from pathlib import Path

import pytest

from imagen_pipeline.core.assets import AssetConflictError, AssetLibrary
from imagen_pipeline.core.preferences import BiasConfig, LockSet
from imagen_pipeline.core.selector import AssetSelector, EmptyPoolError
from imagen_pipeline.core.constraints import ScenarioContext


def write_asset(root: Path, group: str, name: str, payload: dict) -> None:
    path = root / group / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def test_locks_limit_pool_and_raise_on_empty(tmp_path: Path) -> None:
    catalog = tmp_path / "catalog"
    write_asset(
        catalog,
        "models",
        "model_a",
        {"id": "model_a", "label": "A", "weight": 1.0, "tags": [], "requires": [], "excludes": [], "meta": {}},
    )
    write_asset(
        catalog,
        "models",
        "model_b",
        {"id": "model_b", "label": "B", "weight": 1.0, "tags": [], "requires": [], "excludes": [], "meta": {}},
    )
    bias = BiasConfig(locks={"models": LockSet(allow=["model_a"])})
    library = AssetLibrary([catalog], schema_dir=Path("schemas"))
    selector = AssetSelector(library, bias)
    picked = selector.pick_one("models", selected={}, context=ScenarioContext())
    assert picked["id"] == "model_a"

    context = ScenarioContext(locks={"models": LockSet(allow=["missing"])})
    with pytest.raises(EmptyPoolError) as exc:
        selector.pick_one("models", selected={}, context=context)
    message = str(exc.value)
    assert "models" in message
    assert "missing" in message


def test_asset_pack_conflict_logging_and_fail(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    base = tmp_path / "base"
    override = tmp_path / "override"
    write_asset(
        base,
        "poses",
        "duplicate",
        {"id": "duplicate", "label": "Base pose", "weight": 1.0, "tags": [], "requires": [], "excludes": [], "meta": {}},
    )
    write_asset(
        override,
        "poses",
        "duplicate",
        {"id": "duplicate", "label": "Override pose", "weight": 1.0, "tags": [], "requires": [], "excludes": [], "meta": {}},
    )
    caplog.set_level("WARNING")
    library = AssetLibrary([base, override], schema_dir=Path("schemas"))
    asset = library.get("poses", "duplicate")
    assert asset["label"] == "Override pose"
    assert any("replaced" in record.message for record in caplog.records)

    with pytest.raises(AssetConflictError):
        AssetLibrary([base, override], schema_dir=Path("schemas"), fail_on_conflict=True)
