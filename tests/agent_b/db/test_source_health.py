"""source_health bookkeeping against real Postgres.

The property that matters: recording a failure never does anything except record
it. `degraded_since` marks WHEN degradation began and does not move on further
failures, so "degraded for more than N days" stays answerable; a success clears
it, because recovery is recovery.
"""

from __future__ import annotations


def _rec(store, source, *, success, error=None, degraded_after=3):
    return store.record_source_health(
        source, success=success, error=error, degraded_after=degraded_after
    )


def test_a_first_success_starts_a_clean_row(store):
    row = _rec(store, "blog", success=True)
    assert row["consecutive_failures"] == 0
    assert row["last_success_at"] is not None
    assert row["degraded_since"] is None
    assert row["total_cycles"] == 1


def test_failures_accumulate_and_a_success_resets_them(store):
    _rec(store, "blog", success=False, error="timeout")
    row = _rec(store, "blog", success=False, error="timeout")
    assert row["consecutive_failures"] == 2
    assert row["total_cycles"] == 2

    recovered = _rec(store, "blog", success=True)
    assert recovered["consecutive_failures"] == 0
    assert recovered["last_error"] is None
    assert recovered["total_cycles"] == 3


def test_degraded_since_is_stamped_at_the_threshold_not_before(store):
    a = _rec(store, "blog", success=False, error="x", degraded_after=3)
    b = _rec(store, "blog", success=False, error="x", degraded_after=3)
    assert a["degraded_since"] is None
    assert b["degraded_since"] is None

    c = _rec(store, "blog", success=False, error="x", degraded_after=3)
    assert c["consecutive_failures"] == 3
    assert c["degraded_since"] is not None


def test_degraded_since_does_not_move_on_further_failures(store):
    for _ in range(3):
        _rec(store, "blog", success=False, error="x", degraded_after=3)
    first = store.get_source_health("blog")["degraded_since"]

    _rec(store, "blog", success=False, error="x", degraded_after=3)
    assert store.get_source_health("blog")["degraded_since"] == first


def test_recovery_clears_degraded(store):
    for _ in range(3):
        _rec(store, "blog", success=False, error="x", degraded_after=3)
    assert store.get_source_health("blog")["degraded_since"] is not None

    _rec(store, "blog", success=True)
    assert store.get_source_health("blog")["degraded_since"] is None
