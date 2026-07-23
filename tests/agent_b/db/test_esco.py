"""The ESCO layer against real Postgres: sync, mapping SQL, and the aggregation
join that finally fills the reserved esco_code column.

The canonicalization test at the bottom is the reason the whole layer exists:
two raw skills that phrase one concept differently must end up sharing an
esco_code, so grouping by it merges their counts.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

from agents.agent_b_job_ingest.aggregate import recompute_stats
from agents.agent_b_job_ingest.esco_map import map_new_skills, sync_taxonomy
from shared.config import Config
from tests.fake_embedder import FakeEmbedder

FIXTURE = Path(__file__).parents[1] / "fixtures" / "esco_skills_sample.csv"
AS_OF = date(2026, 7, 22)


def _sync(store):
    # No transaction wrapper: sync manages its own commits (that per-batch
    # commit IS the resumability contract, so tests exercise it as shipped).
    return sync_taxonomy(store, FakeEmbedder(), path=FIXTURE, version="test-1")


def _ins_posting(store, pid, skills):
    conn = store.connect()
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO job_postings (posting_id,source,source_group,source_type,source_url,"
            "title,raw_description,content_hash,sector,required_skills,country,listing_intent,"
            "poster_type,status,posted_date) VALUES "
            "(%s,'s','g','blogger_feed',%s,'t','d',%s,'2',%s,'OM','vacancy','company','active','2026-07-01')",
            (pid, f"https://e.test/{pid}", f"h_{pid}", list(skills)),
        )
    conn.commit()


def _map(store):
    with store.transaction():
        return map_new_skills(store, FakeEmbedder(), Config())


# ---------------------------------------------------------------------------
# sync
# ---------------------------------------------------------------------------
def test_sync_loads_concepts_labels_and_embeddings(store):
    summary = _sync(store)

    assert summary.skills == 5
    status = store.esco_status()
    assert status["skills"] == 5
    # 5 preferred + 8 alt labels = 13
    assert status["labels"] == 13
    assert status["embedded"] == 13, "every label must end up with a vector"
    assert status["version"] == "test-1"


def test_sync_is_idempotent_and_does_not_re_embed(store):
    _sync(store)
    second = _sync(store)

    assert second.embedded == 0, "already-embedded labels were re-paid for"
    assert store.esco_status()["labels"] == 13


# ---------------------------------------------------------------------------
# mapping through the real SQL
# ---------------------------------------------------------------------------
def test_lexical_and_unmapped_paths_write_auditable_rows(store):
    _sync(store)
    _ins_posting(store, "p1", ["Manage Time", "bookkeeping", "quantum basket weaving"])

    summary = _map(store)

    assert summary.exact == 1        # manage time -> preferred label
    assert summary.alt_label == 1    # bookkeeping -> alt of accounting
    assert summary.unmapped == 1     # nothing resembles quantum basket weaving

    conn = store.connect()
    with conn.cursor() as cur:
        cur.execute("SELECT skill_key, method, esco_uri, similarity FROM skill_esco_map ORDER BY skill_key")
        rows = {r["skill_key"]: r for r in cur.fetchall()}

    assert rows["manage time"]["method"] == "exact"
    assert rows["bookkeeping"]["esco_uri"].endswith("/0004")
    unmapped = rows["quantum basket weaving"]
    assert unmapped["method"] == "unmapped" and unmapped["esco_uri"] is None
    assert unmapped["similarity"] is not None, "the near-miss score is the tuning evidence"


def test_mapped_keys_are_not_reconsidered_next_cycle(store):
    _sync(store)
    _ins_posting(store, "p1", ["manage time", "quantum basket weaving"])
    _map(store)

    second = _map(store)
    assert second.considered == 0, "an already-decided key was re-examined"


def test_unmapped_keys_retry_only_on_a_new_taxonomy_version(store):
    _sync(store)
    _ins_posting(store, "p1", ["quantum basket weaving"])
    _map(store)
    assert _map(store).considered == 0

    # A new release is the one event that can turn a miss into a hit.
    sync_taxonomy(store, FakeEmbedder(), path=FIXTURE, version="test-2")
    retry = _map(store)
    assert retry.considered == 1


def test_preferred_label_beats_alt_label_on_collision(store):
    """'data analysis' is uri 0002's preferred label; if some other concept ever
    lists it as an alt, DISTINCT ON must keep the preferred one. Simulated by
    inserting a competing alt label directly."""
    _sync(store)
    conn = store.connect()
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO esco_labels (label_key, label, esco_uri, is_preferred) "
            "VALUES ('analyse data', 'analyse data', 'http://example.test/esco/skill/0003', false)"
        )
    conn.commit()

    hit = store.find_esco_by_labels(["analyse data"])["analyse data"]
    assert hit["esco_uri"].endswith("/0002")
    assert hit["is_preferred"] is True


# ---------------------------------------------------------------------------
# the payoff: aggregation fills esco_code, variants share it
# ---------------------------------------------------------------------------
def test_two_phrasings_of_one_concept_share_an_esco_code(store):
    _sync(store)
    _ins_posting(store, "p1", ["manage time"])          # preferred label
    _ins_posting(store, "p2", ["prioritise tasks"])     # alt label, same concept
    _ins_posting(store, "p3", ["quantum basket weaving"])

    _map(store)
    with store.transaction():
        recompute_stats(store, Config(), as_of=AS_OF)

    conn = store.connect()
    with conn.cursor() as cur:
        cur.execute("SELECT skill_key, esco_code FROM skill_demand_stats")
        rows = {r["skill_key"]: r["esco_code"] for r in cur.fetchall()}

    assert rows["manage time"] == rows["prioritise tasks"], (
        "two phrasings of one concept must share an esco_code — this merge is "
        "the entire point of the layer"
    )
    assert rows["manage time"].endswith("/0001")
    assert rows["quantum basket weaving"] is None, "an unmapped skill must stay NULL, never guessed"

    # And the canonical view a consumer would take:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT esco_code, sum(frequency_count) AS n FROM skill_demand_stats "
            "WHERE esco_code IS NOT NULL GROUP BY esco_code"
        )
        merged = {r["esco_code"]: r["n"] for r in cur.fetchall()}
    assert merged["http://example.test/esco/skill/0001"] == 2


def test_aggregation_without_any_mapping_leaves_esco_code_null(store):
    """The layer is optional at runtime: no sync, no map — stats still compute,
    esco_code just stays NULL exactly as before this layer existed."""
    _ins_posting(store, "p1", ["manage time"])
    with store.transaction():
        recompute_stats(store, Config(), as_of=AS_OF)

    conn = store.connect()
    with conn.cursor() as cur:
        cur.execute("SELECT esco_code FROM skill_demand_stats")
        assert cur.fetchone()["esco_code"] is None
