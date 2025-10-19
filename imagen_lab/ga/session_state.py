"""Session-level tracking helpers for genetic statistics.

The evolutionary loops record per-gene fitness observations and soft penalties
for failing combinations.  At the end of the session these observations are
blended into the persistent ``gene_stats`` table via an exponential moving
average (EMA).  During the session the tracker exposes lightweight bias signals
that can be consumed by the bias engine to steer sampling probabilities.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Iterable, Mapping, MutableMapping, Optional

from imagen_lab.db.repo.interfaces import GeneStatRecord, RepositoryProtocol


def _as_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return default


@dataclass(slots=True)
class GeneStatBaseline:
    """Immutable view of a stored ``gene_stats`` row."""

    ema_fitness: float = 0.0
    count: int = 0
    last_seen_ts: int = 0
    confidence: float = 0.0


class SessionGeneStats:
    """Collects EMA updates and transient penalties for a session.

    Parameters
    ----------
    baseline:
        Optional iterable with existing rows from ``gene_stats``.
    ema_alpha:
        Blending factor for the exponential moving average.  A larger value
        gives more weight to the observations of the current session.
    penalty_increment:
        Amount added to the penalty memory whenever a combination fails.
    penalty_decay:
        Multiplicative decay applied to penalties after every iteration to keep
        the memory short-lived within the session.
    threshold:
        Minimum accumulated penalty required for ``should_penalize`` to flag a
        gene set.  The value is interpreted on a 0..1 scale.
    """

    def __init__(
        self,
        baseline: Optional[Iterable[Mapping[str, object]]] = None,
        *,
        ema_alpha: float = 0.35,
        penalty_increment: float = 0.25,
        penalty_decay: float = 0.6,
        threshold: float = 0.9,
    ) -> None:
        self._ema_alpha = max(0.0, min(1.0, float(ema_alpha)))
        self._penalty_increment = max(0.0, float(penalty_increment))
        self._penalty_decay = max(0.0, min(1.0, float(penalty_decay)))
        self._penalty_threshold = max(0.0, float(threshold))

        self._baseline: dict[str, GeneStatBaseline] = {}
        if baseline is not None:
            for row in baseline:
                gene_id = str(row.get("gene_id"))
                if not gene_id:
                    continue
                self._baseline[gene_id] = GeneStatBaseline(
                    ema_fitness=_as_float(row.get("ema_fitness"), 0.0),
                    count=int(row.get("count", 0) or 0),
                    last_seen_ts=int(row.get("last_seen_ts", 0) or 0),
                    confidence=_as_float(row.get("confidence"), 0.0),
                )

        self._session_totals: MutableMapping[str, float] = {}
        self._session_counts: MutableMapping[str, int] = {}
        self._last_seen: MutableMapping[str, int] = {}
        self._penalties: MutableMapping[str, float] = {}

    # ------------------------------------------------------------------
    # Penalty handling
    # ------------------------------------------------------------------
    def record_failure(self, gene_ids: Mapping[str, object]) -> None:
        """Registers a failed attempt for the provided genes."""

        if not gene_ids:
            return
        for gene in gene_ids.values():
            if not gene:
                continue
            gid = str(gene)
            current = self._penalties.get(gid, 0.0)
            updated = min(1.0, current + self._penalty_increment)
            self._penalties[gid] = updated

    def decay_penalties(self) -> None:
        """Applies exponential decay to the in-memory penalty scores."""

        if not self._penalties:
            return
        for gid in list(self._penalties.keys()):
            value = self._penalties[gid] * self._penalty_decay
            if value < 1e-2:
                self._penalties.pop(gid, None)
            else:
                self._penalties[gid] = value

    def should_penalize(self, gene_ids: Mapping[str, object]) -> bool:
        """Returns ``True`` when the genes accumulated a strong penalty."""

        if not self._penalties or not gene_ids:
            return False
        score = 0.0
        for gene in gene_ids.values():
            if not gene:
                continue
            score += self._penalties.get(str(gene), 0.0)
        return score >= self._penalty_threshold

    def penalty_snapshot(self) -> Mapping[str, float]:
        """Current penalty map for bias engines."""

        return dict(self._penalties)

    # ------------------------------------------------------------------
    # EMA handling
    # ------------------------------------------------------------------
    def record_success(
        self,
        gene_ids: Mapping[str, object],
        fitness: float,
        *,
        timestamp: Optional[int] = None,
    ) -> None:
        """Tracks a successful evaluation for the provided genes."""

        if not gene_ids:
            return
        ts = int(timestamp or time.time())
        fitness_value = max(0.0, float(fitness))
        for gene in gene_ids.values():
            if not gene:
                continue
            gid = str(gene)
            self._session_totals[gid] = self._session_totals.get(gid, 0.0) + fitness_value
            self._session_counts[gid] = self._session_counts.get(gid, 0) + 1
            self._last_seen[gid] = ts
            # Successful observations ease off the penalty memory.
            if gid in self._penalties:
                reduced = self._penalties[gid] * 0.5
                if reduced < 1e-2:
                    self._penalties.pop(gid, None)
                else:
                    self._penalties[gid] = reduced

    def ema_snapshot(self) -> Mapping[str, float]:
        """Predicts the EMA after blending the current session observations."""

        snapshot: dict[str, float] = {
            gene_id: baseline.ema_fitness for gene_id, baseline in self._baseline.items()
        }
        for gene_id, total in self._session_totals.items():
            count = self._session_counts.get(gene_id, 0)
            if count <= 0:
                continue
            session_avg = total / float(count)
            base = snapshot.get(gene_id, session_avg)
            snapshot[gene_id] = (1.0 - self._ema_alpha) * base + self._ema_alpha * session_avg
        return snapshot

    def flush(self, repository: RepositoryProtocol) -> None:
        """Persists aggregated EMA updates into the repository."""

        if not self._session_totals:
            self._penalties.clear()
            return

        now = int(time.time())
        records: list[GeneStatRecord] = []
        for gene_id, total in list(self._session_totals.items()):
            count = self._session_counts.get(gene_id, 0)
            if count <= 0:
                continue
            session_avg = total / float(count)
            baseline = self._baseline.get(gene_id)
            if baseline is None or baseline.count <= 0:
                ema_value = session_avg
                total_count = count
            else:
                ema_value = (1.0 - self._ema_alpha) * baseline.ema_fitness + self._ema_alpha * session_avg
                total_count = baseline.count + count

            seen_ts = self._last_seen.get(gene_id, now)
            confidence = 1.0 - math.exp(-float(total_count) / 6.0)
            confidence = min(0.999, max(0.0, confidence))

            records.append(
                GeneStatRecord(
                    gene_id=gene_id,
                    ema_fitness=ema_value,
                    count=total_count,
                    last_seen_ts=seen_ts,
                    confidence=confidence,
                    updated_at=now,
                )
            )

            self._baseline[gene_id] = GeneStatBaseline(
                ema_fitness=ema_value,
                count=total_count,
                last_seen_ts=seen_ts,
                confidence=confidence,
            )

        repository.upsert_gene_stats(records)

        self._session_totals.clear()
        self._session_counts.clear()
        self._last_seen.clear()
        self._penalties.clear()


__all__ = ["SessionGeneStats", "GeneStatBaseline"]

