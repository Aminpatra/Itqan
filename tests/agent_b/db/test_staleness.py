"""The staleness lifecycle against real Postgres.

The rule is "age by CYCLES, not by clock", and the property that matters is that
each transition is reversible in exactly one direction: seen → active, missed →
stale, stale-too-long → gone. Rejected rows sit outside all of it.
"""

from __future__ import annotations

from agents.agent_b_job_ingest.records import PersistedPosting


def _row(pid: str, **kw) -> PersistedPosting:
    base = dict(
        posting_id=pid, source=kw.pop("source", "el7far"),
        source_group="el7far_network", source_type="blogger_feed",
        source_url=f"https://example.test/{pid}", title="A Role",
        raw_description="desc", content_hash=f"h_{pid}",
    )
    base.update(kw)
    return PersistedPosting(**base)


def _set(store, pid, **cols):
    conn = store.connect()
    sets = ", ".join(f"{k} = %({k})s" for k in cols)
    with conn.cursor() as cur:
        cur.execute(f"UPDATE job_postings SET {sets} WHERE posting_id = %(pid)s", {**cols, "pid": pid})
    conn.commit()


def _col(store, pid, col):
    conn = store.connect()
    with conn.cursor() as cur:
        cur.execute(f"SELECT {col} FROM job_postings WHERE posting_id = %s", (pid,))
        row = cur.fetchone()
        return row[col] if row else None


# ---------------------------------------------------------------------------
def test_age_missed_only_touches_the_named_sources(store):
    """A source that did not fetch cleanly must not have its inventory aged —
    otherwise a single 429 pushes its live jobs toward deletion."""
    store.upsert_batch([_row("a", source="el7far"), _row("b", source="dubizzle")])

    store.age_missed(["el7far"])

    assert _col(store, "a", "missed_cycles") == 1
    assert _col(store, "b", "missed_cycles") == 0


def test_seen_postings_net_to_zero_across_age_then_touch(store):
    """The cycle order that makes 'count cycles' work: age blanket-increments,
    then the pipeline resets the ones actually seen."""
    store.upsert_batch([_row("seen"), _row("gone")])

    store.age_missed(["el7far"])          # both -> 1
    store.touch_seen(["seen"])            # seen -> 0

    assert _col(store, "seen", "missed_cycles") == 0
    assert _col(store, "gone", "missed_cycles") == 1


def test_mark_stale_fires_at_the_threshold_and_stamps_stale_since(store):
    store.upsert_batch([_row("young"), _row("old")])
    _set(store, "young", missed_cycles=2)
    _set(store, "old", missed_cycles=3)

    moved = store.mark_stale(threshold=3)

    assert moved == 1
    assert store.get_status("young") == "active"
    assert store.get_status("old") == "stale"
    assert _col(store, "old", "stale_since") is not None


def test_stale_since_is_not_reset_on_a_second_mark(store):
    store.upsert_batch([_row("x")])
    _set(store, "x", missed_cycles=5)
    store.mark_stale(threshold=3)
    first = _col(store, "x", "stale_since")

    _set(store, "x", missed_cycles=6)
    store.mark_stale(threshold=3)
    assert _col(store, "x", "stale_since") == first, "the prune clock was reset"


def test_prune_deletes_only_long_stale_rows(store):
    store.upsert_batch([_row("fresh_stale", status="stale"), _row("old_stale", status="stale")])
    conn = store.connect()
    with conn.cursor() as cur:
        cur.execute("UPDATE job_postings SET stale_since = now() - interval '10 days' WHERE posting_id='fresh_stale'")
        cur.execute("UPDATE job_postings SET stale_since = now() - interval '90 days' WHERE posting_id='old_stale'")
    conn.commit()

    deleted = store.prune(older_than_days=60)

    assert deleted == 1
    assert store.get_status("old_stale") is None
    assert store.get_status("fresh_stale") == "stale"


def test_prune_never_deletes_a_rejected_posting(store):
    """Rejected rows are audit evidence, retained forever. The prune is
    WHERE status='stale' precisely so a scam can never age out of the record."""
    store.upsert_batch([_row("scam", status="rejected")])
    conn = store.connect()
    with conn.cursor() as cur:
        cur.execute("UPDATE job_postings SET stale_since = now() - interval '999 days' WHERE posting_id='scam'")
    conn.commit()

    deleted = store.prune(older_than_days=60)

    assert deleted == 0
    assert store.get_status("scam") == "rejected"


def test_pruning_a_canonical_promotes_its_duplicate(store):
    """ON DELETE SET NULL: the orphan becomes standalone, which is exactly true
    once its canonical is gone — not deleted alongside it."""
    store.upsert_batch([
        _row("canon", status="stale"),
        _row("dup", duplicate_of="canon"),
    ])
    conn = store.connect()
    with conn.cursor() as cur:
        cur.execute("UPDATE job_postings SET stale_since = now() - interval '90 days' WHERE posting_id='canon'")
    conn.commit()

    store.prune(older_than_days=60)

    assert store.get_status("canon") is None
    assert store.get_status("dup") == "active"
    assert _col(store, "dup", "duplicate_of") is None
