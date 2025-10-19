from dataclasses import dataclass
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from imagen_lab.ga.session_state import SessionGeneStats


@dataclass
class DummyRepo:
    records: list

    def upsert_gene_stats(self, records):  # type: ignore[override]
        self.records.extend(records)


def test_session_gene_stats_updates_and_penalties():
    baseline = [
        {
            "gene_id": "pose_a",
            "ema_fitness": 40.0,
            "count": 4,
            "last_seen_ts": 10,
            "confidence": 0.5,
        }
    ]
    tracker = SessionGeneStats(
        baseline,
        ema_alpha=0.5,
        penalty_increment=0.4,
        penalty_decay=0.5,
        threshold=0.3,
    )

    genes = {"pose": "pose_a"}
    tracker.record_failure(genes)
    assert tracker.should_penalize(genes)

    tracker.record_success(genes, 80.0, timestamp=100)
    assert not tracker.should_penalize(genes)

    snapshot = tracker.ema_snapshot()
    assert snapshot["pose_a"] == pytest.approx(60.0)

    repo = DummyRepo(records=[])
    tracker.flush(repo)  # type: ignore[arg-type]

    assert len(repo.records) == 1
    record = repo.records[0]
    assert record.gene_id == "pose_a"
    assert record.ema_fitness == pytest.approx(60.0)
    assert record.count == 5
    assert record.last_seen_ts == 100

    tracker.record_failure(genes)
    tracker.decay_penalties()
    assert not tracker.should_penalize(genes)
