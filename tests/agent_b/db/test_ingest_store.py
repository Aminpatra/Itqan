"""The ingestion store methods against real Postgres + pgvector.

The offline pipeline tests use a FakeStore, which proves the pipeline LOGIC. What
cannot be faked is the SQL itself: the FK-ordered upsert, the ON CONFLICT
lifecycle reset, and above all the pgvector cosine search — whose
distance-vs-similarity direction and partial-index predicates are exactly the
things that are silent when wrong. Those are tested here.
"""

from __future__ import annotations

from agents.agent_b_job_ingest.records import PersistedPosting


def _row(pid: str, **kw) -> PersistedPosting:
    base = dict(
        posting_id=pid,
        source=kw.pop("source", "el7far"),
        source_group=kw.pop("source_group", "el7far_network"),
        source_type=kw.pop("source_type", "blogger_feed"),
        source_url=kw.pop("source_url", f"https://example.test/{pid}"),
        title=kw.pop("title", "A Role"),
        raw_description=kw.pop("raw_description", "A description of the role."),
        content_hash=kw.pop("content_hash", f"hash_{pid}"),
    )
    base.update(kw)
    return PersistedPosting(**base)


def _unit_vector(index: int) -> list[float]:
    """A one-hot 1536-d vector. Two different indices are orthogonal (cosine 0);
    the same index is identical (cosine 1). That gives hand-computable
    similarities without depending on any embedder."""
    vec = [0.0] * 1536
    vec[index] = 1.0
    return vec


def _blend(i: int, j: int, wi: float, wj: float) -> list[float]:
    import math

    vec = [0.0] * 1536
    vec[i] += wi
    vec[j] += wj
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


# ---------------------------------------------------------------------------
# upsert + change detection
# ---------------------------------------------------------------------------
def test_upsert_then_lookup_hash_reports_unchanged(store):
    store.upsert_batch([_row("a", content_hash="h1")])

    assert store.lookup_hashes(["a"]) == {"a": "h1"}
    assert store.lookup_hashes(["missing"]) == {}


def test_touch_seen_revives_a_stale_posting_but_not_a_rejected_one(store):
    store.upsert_batch([_row("stale_one", status="stale"), _row("scam", status="rejected")])
    # mimic staleness having been set
    conn = store.connect()
    with conn.cursor() as cur:
        cur.execute("UPDATE job_postings SET stale_since = now() WHERE posting_id = 'stale_one'")
    conn.commit()

    store.touch_seen(["stale_one", "scam"])

    assert store.get_status("stale_one") == "active"
    assert store.get_status("scam") == "rejected", "re-seeing a scam must not revive it"
    with conn.cursor() as cur:
        cur.execute("SELECT stale_since FROM job_postings WHERE posting_id = 'stale_one'")
        assert cur.fetchone()["stale_since"] is None


def test_upsert_preserves_first_seen_at_but_moves_last_seen(store):
    store.upsert_batch([_row("a")])
    conn = store.connect()
    with conn.cursor() as cur:
        cur.execute("SELECT first_seen_at, last_seen_at FROM job_postings WHERE posting_id = 'a'")
        before = cur.fetchone()
        cur.execute(
            "UPDATE job_postings SET first_seen_at = now() - interval '10 days', "
            "last_seen_at = now() - interval '10 days', missed_cycles = 5 WHERE posting_id = 'a'"
        )
    conn.commit()

    store.upsert_batch([_row("a", content_hash="h2")])
    with conn.cursor() as cur:
        cur.execute(
            "SELECT first_seen_at, last_seen_at, missed_cycles FROM job_postings WHERE posting_id='a'"
        )
        after = cur.fetchone()

    assert after["first_seen_at"] < before["last_seen_at"], "first_seen_at was reset"
    assert after["missed_cycles"] == 0, "a re-seen posting must reset its missed-cycle counter"


def test_upsert_writes_a_duplicate_after_its_in_batch_canonical(store):
    """FK ordering under one transaction: the duplicate is listed first but must
    still commit, because upsert sorts canonicals ahead of duplicates."""
    dup = _row("dup", duplicate_of="canon", source_url="https://example.test/dup")
    canon = _row("canon", source_url="https://example.test/canon")

    store.upsert_batch([dup, canon])  # deliberately dup-first

    conn = store.connect()
    with conn.cursor() as cur:
        cur.execute("SELECT duplicate_of FROM job_postings WHERE posting_id = 'dup'")
        assert cur.fetchone()["duplicate_of"] == "canon"


# ---------------------------------------------------------------------------
# the pgvector near-dup search — the part nothing else can cover
# ---------------------------------------------------------------------------
def test_similarity_is_one_minus_distance_not_distance(store):
    """The single most likely bug in the feature. An identical vector must come
    back as similarity ~1.0, an orthogonal one as ~0.0 — if the SQL returned
    distance, these would be swapped and every near-dup decision inverted."""
    store.upsert_batch([_row("v0", embedding=_unit_vector(0))])

    same = store.find_neardup_candidates(
        _unit_vector(0), recent_days=14, limit=5, exclude_id="other"
    )
    assert same and abs(same[0]["similarity"] - 1.0) < 1e-6

    orthogonal = store.find_neardup_candidates(
        _unit_vector(1), recent_days=14, limit=5, exclude_id="other"
    )
    assert orthogonal and abs(orthogonal[0]["similarity"]) < 1e-6


def test_neardup_search_excludes_duplicates_and_rejected_and_self(store):
    """The candidate set must match the partial HNSW index predicate verbatim,
    or a merged/rejected posting resurfaces as a false duplicate."""
    store.upsert_batch([
        _row("canon", embedding=_unit_vector(0)),
        _row("already_dup", embedding=_unit_vector(0), duplicate_of="canon",
             source_url="https://example.test/already_dup"),
        _row("rejected", embedding=_unit_vector(0), status="rejected",
             source_url="https://example.test/rejected"),
        _row("self", embedding=_unit_vector(0), source_url="https://example.test/self"),
    ])

    hits = store.find_neardup_candidates(
        _unit_vector(0), recent_days=14, limit=10, exclude_id="self"
    )
    ids = {h["posting_id"] for h in hits}

    assert "canon" in ids
    assert "already_dup" not in ids, "a duplicate resurfaced as a candidate"
    assert "rejected" not in ids, "a rejected posting resurfaced as a candidate"
    assert "self" not in ids, "a posting matched itself"


def test_neardup_search_respects_the_recency_window(store):
    store.upsert_batch([_row("old", embedding=_unit_vector(0))])
    conn = store.connect()
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE job_postings SET first_seen_at = now() - interval '30 days' "
            "WHERE posting_id = 'old'"
        )
    conn.commit()

    hits = store.find_neardup_candidates(
        _unit_vector(0), recent_days=14, limit=5, exclude_id="x"
    )
    assert "old" not in {h["posting_id"] for h in hits}


def test_neardup_orders_by_similarity(store):
    store.upsert_batch([
        _row("far", embedding=_blend(0, 1, 0.3, 0.7)),
        _row("near", embedding=_blend(0, 1, 0.95, 0.05), source_url="https://example.test/near"),
    ])

    hits = store.find_neardup_candidates(
        _unit_vector(0), recent_days=14, limit=5, exclude_id="q"
    )
    assert [h["posting_id"] for h in hits][:2] == ["near", "far"]
    assert hits[0]["similarity"] > hits[1]["similarity"]


def test_find_by_source_urls_resolves_the_link_dedup_target(store):
    store.upsert_batch([_row("blog", source_url="https://oman.el7far.com/job.html")])

    found = store.find_by_source_urls(
        ["https://oman.el7far.com/job.html", "https://oman.el7far.com/missing.html"]
    )
    assert found == {"https://oman.el7far.com/job.html": "blog"}
