"""Aggregation snapshots must not accumulate forever.

Both agents replaced rows WITHIN a window from the start; neither pruned across
them. Measured 2026-07-28: `skill_demand_stats` held 4 windows / 4,120 rows to
serve 1,139 current, `skill_supply_stats` 3 windows / 18,334 to serve 10,202 —
and every consumer query runs `window_end = (SELECT max(window_end) …)` over the
lot. At Agent B's 12-hour cycle that is roughly 730 windows a year.

The floor is the interesting half: retention below 2 windows silently breaks
Agent B's trend calculation, which reads `prior_frequency_count` from the STORED
prior window.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from shared.config import Config


def _seed_windows(store, n: int) -> list[date]:
    """n distinct daily snapshots, oldest first."""
    today = date.today()
    ends = [today - timedelta(days=i) for i in range(n)][::-1]
    conn = store.connect()
    with conn.cursor() as cur:
        for w_end in ends:
            cur.execute(
                """
                INSERT INTO skill_demand_stats
                    (sector, skill, skill_key, window_start, window_end,
                     frequency_count, trend)
                VALUES ('2', 'Accounting', 'accounting', %s, %s, 5, 'stable')
                """,
                (w_end - timedelta(days=30), w_end),
            )
    conn.commit()
    return ends


def _windows(store) -> list[date]:
    with store.connect().cursor() as cur:
        cur.execute("SELECT DISTINCT window_end FROM skill_demand_stats ORDER BY window_end")
        return [r["window_end"] for r in cur.fetchall()]


def test_only_the_newest_windows_survive(store):
    _seed_windows(store, 12)
    assert len(_windows(store)) == 12

    removed = store.prune_stats_windows(keep=8)
    store.connect().commit()

    kept = _windows(store)
    assert len(kept) == 8 and removed == 4
    assert kept == sorted(kept)[-8:], "pruning kept the wrong end of the history"


def test_pruning_is_idempotent(store):
    _seed_windows(store, 10)
    store.prune_stats_windows(keep=8)
    store.connect().commit()
    assert store.prune_stats_windows(keep=8) == 0


def test_a_history_shorter_than_the_retention_is_left_alone(store):
    _seed_windows(store, 3)
    assert store.prune_stats_windows(keep=8) == 0
    assert len(_windows(store)) == 3


def test_retention_below_two_is_refused(store):
    """Not a preference — Agent B's trend reads `prior_frequency_count` from the
    stored prior window, so keeping one would relabel every skill from a prior of
    zero and silently resurrect the fabricated-trend bug its audit fixed."""
    _seed_windows(store, 5)
    with pytest.raises(ValueError, match="floor|prior window"):
        store.prune_stats_windows(keep=1)
    assert len(_windows(store)) == 5, "a refused prune still deleted rows"


def test_the_config_floors_it_before_the_store_ever_sees_it():
    assert Config(stats_retention_windows=1).stats_windows_to_keep() == 2
    assert Config(stats_retention_windows=0).stats_windows_to_keep() == 2
    assert Config(stats_retention_windows=8).stats_windows_to_keep() == 8


def test_the_prior_window_a_trend_needs_is_still_there_after_pruning(store):
    """The concrete reason for the floor, asserted end to end."""
    _seed_windows(store, 12)
    store.prune_stats_windows(keep=Config().stats_windows_to_keep())
    store.connect().commit()

    kept = _windows(store)
    assert len(kept) >= 2, "no prior window survived for the trend calculation"
