"""Schema semantics.

`CREATE TABLE` succeeding proves nothing about whether the constraints behave.
These tests exercise the decisions that were made deliberately in the migrations
— the FK delete action, the CHECKs, the vector round-trip, and the partial index
predicates — because each of them is silent when wrong.
"""

from __future__ import annotations

import psycopg
import pytest

# Skipping when no test database is configured is handled by the
# pytest_collection_modifyitems hook in this directory's conftest.


def _insert(store, posting_id: str, **overrides):
    row = {
        "posting_id": posting_id,
        "source": "test_source",
        "source_group": "test_group",
        "source_type": "blogger_feed",
        "source_url": f"https://example.test/{posting_id}",
        "title": "Synthetic Test Posting",
        "raw_description": "Synthetic fixture. Not a real job posting.",
        "content_hash": f"hash_{posting_id}",
        "status": "active",
    }
    row.update(overrides)
    cols = ", ".join(row)
    vals = ", ".join(f"%({k})s" for k in row)
    conn = store.connect()
    with conn.cursor() as cur:
        cur.execute(f"INSERT INTO job_postings ({cols}) VALUES ({vals})", row)
    conn.commit()


# ---------------------------------------------------------------------------
# duplicate_of — the FK action was chosen over two alternatives that both lose
# ---------------------------------------------------------------------------
def test_deleting_a_canonical_orphans_its_duplicate_rather_than_deleting_it(store):
    """ON DELETE SET NULL, not CASCADE.

    CASCADE would let the 60-day prune of a canonical silently delete duplicates
    that may still be active and are independent audit evidence — data loss
    driven by a cron job. SET NULL promotes the orphan back to standalone, which
    is exactly true once its canonical no longer exists.
    """
    _insert(store, "canonical")
    _insert(store, "dup", duplicate_of="canonical")

    conn = store.connect()
    with conn.cursor() as cur:
        cur.execute("DELETE FROM job_postings WHERE posting_id = 'canonical'")
    conn.commit()

    with conn.cursor() as cur:
        cur.execute("SELECT posting_id, duplicate_of FROM job_postings")
        rows = cur.fetchall()

    assert len(rows) == 1, "the duplicate must survive its canonical's deletion"
    assert rows[0]["posting_id"] == "dup"
    assert rows[0]["duplicate_of"] is None


def test_a_posting_cannot_be_its_own_duplicate(store):
    _insert(store, "self")
    conn = store.connect()
    with pytest.raises(psycopg.errors.CheckViolation):
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE job_postings SET duplicate_of = 'self' WHERE posting_id = 'self'"
            )
    conn.rollback()


def test_duplicate_of_must_reference_a_real_posting(store):
    conn = store.connect()
    with pytest.raises(psycopg.errors.ForeignKeyViolation):
        _insert(store, "orphan", duplicate_of="does_not_exist")
    conn.rollback()


# ---------------------------------------------------------------------------
# CHECK constraints
# ---------------------------------------------------------------------------
def test_status_accepts_rejected_and_refuses_anything_else(store):
    for status in ("active", "stale", "needs_review", "rejected"):
        _insert(store, f"s_{status}", status=status)

    conn = store.connect()
    with pytest.raises(psycopg.errors.CheckViolation):
        _insert(store, "bogus", status="deleted")
    conn.rollback()


def test_source_type_is_constrained_to_the_three_known_kinds(store):
    for kind in ("html_scrape", "blogger_feed", "telegram"):
        _insert(store, f"k_{kind}", source_type=kind)

    conn = store.connect()
    with pytest.raises(psycopg.errors.CheckViolation):
        _insert(store, "bad_kind", source_type="carrier_pigeon")
    conn.rollback()


@pytest.mark.parametrize("score", [0, 0.301, 1])
def test_legitimacy_score_accepts_the_unit_interval(store, score):
    _insert(store, f"legit_{score}", legitimacy_score=score)


@pytest.mark.parametrize("score", [-0.1, 1.1])
def test_legitimacy_score_rejects_values_outside_it(store, score):
    conn = store.connect()
    with pytest.raises(psycopg.errors.CheckViolation):
        _insert(store, f"bad_{score}", legitimacy_score=score)
    conn.rollback()


def test_one_source_url_cannot_yield_two_rows(store):
    _insert(store, "first", source_url="https://example.test/same")
    conn = store.connect()
    with pytest.raises(psycopg.errors.UniqueViolation):
        _insert(store, "second", source_url="https://example.test/same")
    conn.rollback()


# ---------------------------------------------------------------------------
# vector round-trip
# ---------------------------------------------------------------------------
def test_embedding_round_trips_as_floats_not_a_string(store):
    """The pgvector adapter is registered, so a list[float] goes in and comes
    back as a typed vector — not through string formatting, which is where
    silent precision and format bugs come from.

    Note the return type is pgvector's ``Vector``, not a list: it has
    ``.to_list()`` / ``.to_numpy()`` / ``.dimensions`` but no ``len()``, and
    ``np.asarray`` on it yields object dtype. That is exactly why the store
    normalizes to ``list[float]`` on read rather than letting a library type
    leak into the nodes.
    """
    vec = [0.0] * 1536
    vec[0], vec[1] = 1.0, -0.5
    _insert(store, "vec", embedding=vec)

    conn = store.connect()
    with conn.cursor() as cur:
        cur.execute("SELECT embedding FROM job_postings WHERE posting_id = 'vec'")
        got = (cur.fetchone() or {})["embedding"]

    assert got is not None
    assert not isinstance(got, str), "vector came back as text; register_vector did not run"
    assert got.dimensions() == 1536

    as_list = got.to_list()
    assert len(as_list) == 1536
    assert as_list[0] == pytest.approx(1.0)
    assert as_list[1] == pytest.approx(-0.5)


def test_store_normalizes_vectors_to_plain_floats(store):
    """Nodes must never receive a pgvector ``Vector``. Keeping the library type
    behind the store boundary is what lets the vector backend change without
    every consumer learning a new API."""
    from agents.agent_b_job_ingest.db.store import to_float_list

    vec = [0.25] * 1536
    _insert(store, "vec_norm", embedding=vec)

    conn = store.connect()
    with conn.cursor() as cur:
        cur.execute("SELECT embedding FROM job_postings WHERE posting_id = 'vec_norm'")
        raw = (cur.fetchone() or {})["embedding"]

    normalized = to_float_list(raw)
    assert isinstance(normalized, list)
    assert all(isinstance(x, float) for x in normalized[:8])
    assert normalized[0] == pytest.approx(0.25)
    # Idempotent, and tolerant of a plain list or None.
    assert to_float_list(normalized) == normalized
    assert to_float_list(None) is None


def test_embedding_dimension_is_enforced(store):
    conn = store.connect()
    with pytest.raises(psycopg.errors.DataException):
        _insert(store, "wrong_dims", embedding=[0.1, 0.2, 0.3])
    conn.rollback()


# ---------------------------------------------------------------------------
# listing provenance (migration 0006)
# ---------------------------------------------------------------------------
def test_provenance_defaults_exclude_a_posting_from_aggregation(store):
    """Both columns default to 'unknown', and only ('vacancy', 'company')
    aggregates.

    Defaulting either to its useful value would mean a classifier that fails to
    run silently publishes exactly what these columns exist to keep out — and
    the bad rows would be indistinguishable from ordinary demand data.
    """
    _insert(store, "prov_default")

    conn = store.connect()
    with conn.cursor() as cur:
        cur.execute(
            "SELECT listing_intent, poster_type FROM job_postings WHERE posting_id = 'prov_default'"
        )
        row = cur.fetchone() or {}

    assert row["listing_intent"] == "unknown"
    assert row["poster_type"] == "unknown"


@pytest.mark.parametrize("intent", ["vacancy", "seeking", "service", "unknown"])
def test_listing_intent_accepts_the_four_known_kinds(store, intent):
    _insert(store, f"intent_{intent}", listing_intent=intent)


def test_listing_intent_rejects_anything_else(store):
    conn = store.connect()
    with pytest.raises(psycopg.errors.CheckViolation):
        _insert(store, "intent_bad", listing_intent="maybe_a_job")
    conn.rollback()


def test_poster_type_rejects_anything_else(store):
    conn = store.connect()
    with pytest.raises(psycopg.errors.CheckViolation):
        _insert(store, "poster_bad", poster_type="probably_a_company")
    conn.rollback()


def test_the_aggregable_index_requires_all_four_conditions(store):
    """A seeker ad counted as a posting measures SUPPLY and publishes it as
    demand. The predicate is written verbatim here because callers must repeat
    it exactly to get the index — and because it is the only place the
    eligibility rule is stated in SQL."""
    conn = store.connect()
    with conn.cursor() as cur:
        cur.execute(
            "SELECT indexdef FROM pg_indexes WHERE indexname = 'job_postings_aggregable_idx'"
        )
        definition = (cur.fetchone() or {})["indexdef"]

    assert "status = 'active'" in definition
    assert "duplicate_of IS NULL" in definition
    assert "listing_intent = 'vacancy'" in definition
    assert "poster_type = 'company'" in definition


# ---------------------------------------------------------------------------
# skill_demand_stats
# ---------------------------------------------------------------------------
def _insert_stat(store, **overrides):
    row = {
        "sector": "2",
        "skill": "Synthetic Skill",
        "skill_key": "synthetic skill",
        "window_start": "2026-06-22",
        "window_end": "2026-07-21",
        "frequency_count": 12,
        "trend": "stable",
    }
    row.update(overrides)
    cols = ", ".join(row)
    vals = ", ".join(f"%({k})s" for k in row)
    conn = store.connect()
    with conn.cursor() as cur:
        cur.execute(f"INSERT INTO skill_demand_stats ({cols}) VALUES ({vals})", row)
    conn.commit()


def test_stats_keep_history_across_windows(store):
    """window_end is in the PK, so a later cycle adds a row rather than
    destroying the prior one — which is what makes the trend column auditable
    against its own past."""
    _insert_stat(store, window_end="2026-07-21")
    _insert_stat(store, window_end="2026-07-22")

    conn = store.connect()
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) AS n FROM skill_demand_stats")
        assert (cur.fetchone() or {})["n"] == 2


def test_low_confidence_defaults_to_false(store):
    _insert_stat(store)
    conn = store.connect()
    with conn.cursor() as cur:
        cur.execute("SELECT low_confidence FROM skill_demand_stats")
        assert (cur.fetchone() or {})["low_confidence"] is False


def test_trend_is_constrained(store):
    conn = store.connect()
    with pytest.raises(psycopg.errors.CheckViolation):
        _insert_stat(store, trend="skyrocketing")
    conn.rollback()


def test_window_cannot_end_before_it_starts(store):
    conn = store.connect()
    with pytest.raises(psycopg.errors.CheckViolation):
        _insert_stat(store, window_start="2026-07-21", window_end="2026-06-22")
    conn.rollback()


# ---------------------------------------------------------------------------
# partial indexes — callers must repeat the predicates verbatim to get them
# ---------------------------------------------------------------------------
def test_hnsw_index_is_partial_on_two_terminal_states(store):
    conn = store.connect()
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT indexdef FROM pg_indexes
             WHERE indexname = 'job_postings_embedding_hnsw'
            """
        )
        definition = (cur.fetchone() or {})["indexdef"]

    assert "hnsw" in definition.lower()
    assert "vector_cosine_ops" in definition, "Agent C must query with <=>; ops class must match"
    assert "duplicate_of IS NULL" in definition
    assert "status <> 'rejected'" in definition


def test_retrieval_index_excludes_duplicates(store):
    """Serving a duplicate to Agent C would double-count one job as two
    opportunities."""
    conn = store.connect()
    with conn.cursor() as cur:
        cur.execute(
            "SELECT indexdef FROM pg_indexes WHERE indexname = 'job_postings_sector_date_idx'"
        )
        definition = (cur.fetchone() or {})["indexdef"]

    assert "duplicate_of IS NULL" in definition
    assert "status = 'active'" in definition
