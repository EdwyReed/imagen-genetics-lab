from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from imagen_lab.db.repo.interfaces import (
    GeneStatRecord,
    ProfileRecord,
    PromptRecord,
    RunRecord,
    ScoreRecord,
)
from imagen_lab.db.repo.sqlite import SQLiteRepository


def _make_repo(tmp_path: Path) -> SQLiteRepository:
    db_path = tmp_path / "repo.sqlite"
    return SQLiteRepository(db_path)


def test_schema_created_with_indexes(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    conn = sqlite3.connect(repo.db_path)
    try:
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert {"runs", "prompts", "scores", "profiles", "gene_stats"}.issubset(tables)

        index_names = {row[1] for row in conn.execute("PRAGMA index_list('prompts')")}
        assert "idx_prompts_session_id" in index_names
        assert "idx_prompts_fitness" in index_names

        gene_indexes = {row[1] for row in conn.execute("PRAGMA index_list('gene_stats')")}
        assert "idx_gene_stats_gene_id" in gene_indexes
    finally:
        conn.close()


def test_run_prompt_and_score_roundtrip(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    repo.log_run(
        RunRecord(
            session_id="s-1",
            mode="plain",
            payload={"cycles": 1},
            macro_snapshot={"sfw": 0.5},
            meso_snapshot={"fitness": 0.7},
            seed=42,
        )
    )

    prompt = PromptRecord(
        path="image.jpg",
        session_id="s-1",
        prompt="hello",
        params={"scene": "test"},
        fitness=1.0,
        status="no_image",
    )
    repo.log_prompt(prompt)

    score = ScoreRecord(
        prompt_path="image.jpg",
        micro_metrics={"style": 0.8},
        meso_metrics={"fitness_visual": 0.75},
        fitness_visual=0.75,
        clip_alignment=0.6,
    )
    repo.log_score(score)

    stored_run = repo.get_run("s-1")
    assert stored_run is not None
    assert stored_run["cfg"] == {"cycles": 1}
    assert stored_run["macro_snapshot"] == {"sfw": 0.5}

    stored_prompt = repo.get_prompt("image.jpg")
    assert stored_prompt is not None
    assert stored_prompt["status"] == "no_image"
    assert stored_prompt["fitness"] == 0.0

    stored_score = repo.get_score("image.jpg")
    assert stored_score is not None
    assert stored_score["micro"] == {"style": 0.8}
    assert stored_score["clip_alignment"] == 0.6


def test_cycle_transaction_rolls_back_on_failure(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    repo.log_run(RunRecord(session_id="sess", mode="plain", payload={}))

    prompt = PromptRecord(
        path="p1.jpg",
        session_id="sess",
        prompt="prompt",
        params={},
        fitness=0.9,
    )
    bad_score = ScoreRecord(
        prompt_path="other_path.jpg",  # violates FK
        micro_metrics={},
        meso_metrics={},
    )

    with pytest.raises(sqlite3.IntegrityError):
        repo.record_cycle(prompt=prompt, score=bad_score)

    assert repo.get_prompt("p1.jpg") is None


def test_profile_and_gene_stats_crud(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    repo.upsert_profile(ProfileRecord(profile_id="profile-a", vector={"w": 1.0}))
    profile = repo.get_profile("profile-a")
    assert profile is not None
    assert profile["vector"] == {"w": 1.0}

    repo.upsert_gene_stats(
        [
            GeneStatRecord(
                gene_id="gene-1",
                ema_fitness=0.5,
                count=3,
                last_seen_ts=10,
                confidence=0.8,
            )
        ]
    )
    stat = repo.get_gene_stat("gene-1")
    assert stat is not None
    assert stat["ema_fitness"] == pytest.approx(0.5)
    assert stat["count"] == 3

    stats = list(repo.iter_gene_stats())
    assert stats and stats[0]["gene_id"] == "gene-1"
