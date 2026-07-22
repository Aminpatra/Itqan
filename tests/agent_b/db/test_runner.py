"""The cycle runner and operator commands against real Postgres.

The graph itself is covered in test_graph_cycle.py; what is added here is the
runner's own responsibilities — the advisory lock that stops two cycles
overlapping, and the two destructive/export operator commands.
"""

from __future__ import annotations

import psycopg

from agents.agent_b_job_ingest.db.store import CYCLE_LOCK_KEY
from agents.agent_b_job_ingest.records import PersistedPosting
from agents.agent_b_job_ingest.runner import run_cycle
from shared.config import Config


def _row(pid, **kw):
    base = dict(
        posting_id=pid, source=kw.pop("source", "el7far"), source_group="g",
        source_type="blogger_feed", source_url=f"https://e.test/{pid}",
        title="A Role", raw_description="desc", content_hash=f"h_{pid}",
    )
    base.update(kw)
    return PersistedPosting(**base)


# ---------------------------------------------------------------------------
# advisory lock
# ---------------------------------------------------------------------------
def test_a_cycle_skips_cleanly_when_another_holds_the_lock(store, monkeypatch):
    """Two overlapping cycles would double-increment missed_cycles and stale
    every posting a cycle early. The second must exit 0 and do nothing — not
    error, not wait. Held here on a separate connection to simulate a running
    cycle; the runner never even reaches the network."""
    config = Config(database_url=store.dsn)

    holder = psycopg.connect(store.dsn)
    try:
        with holder.cursor() as cur:
            cur.execute("SELECT pg_advisory_lock(hashtext(%s))", (CYCLE_LOCK_KEY,))

        result = run_cycle(config, source_names=["el7far"], fake_llm=True)

        assert result.ran is False
        assert result.exit_code == 0
        assert result.state is None
    finally:
        with holder.cursor() as cur:
            cur.execute("SELECT pg_advisory_unlock(hashtext(%s))", (CYCLE_LOCK_KEY,))
        holder.close()


# ---------------------------------------------------------------------------
# --purge-source
# ---------------------------------------------------------------------------
def test_purge_source_deletes_only_that_source(store):
    store.upsert_batch([_row("a", source="dubizzle"), _row("b", source="el7far")])

    deleted = store.purge_source("dubizzle")

    assert deleted == 1
    assert store.get_status("a") is None
    assert store.get_status("b") == "active"


def test_purge_promotes_a_duplicate_in_another_source(store):
    """A cross-source duplicate that pointed at a purged canonical becomes
    standalone (ON DELETE SET NULL), not deleted with it."""
    store.upsert_batch([_row("canon", source="dubizzle")])
    store.upsert_batch([_row("dup", source="el7far", duplicate_of="canon")])

    store.purge_source("dubizzle")

    assert store.get_status("dup") == "active"
    conn = store.connect()
    with conn.cursor() as cur:
        cur.execute("SELECT duplicate_of FROM job_postings WHERE posting_id='dup'")
        assert cur.fetchone()["duplicate_of"] is None


# ---------------------------------------------------------------------------
# --label-sample
# ---------------------------------------------------------------------------
def test_sample_for_labelling_returns_rows_with_the_fields_a_reviewer_needs(store):
    for i in range(5):
        store.upsert_batch([_row(f"p{i}", legitimacy_score=0.8)])

    sample = store.sample_for_labelling(3)

    assert len(sample) == 3
    for row in sample:
        assert {"posting_id", "title", "raw_description", "legitimacy_score", "status"} <= set(row)


def test_sample_size_is_capped_at_what_exists(store):
    store.upsert_batch([_row("only")])
    assert len(store.sample_for_labelling(50)) == 1
