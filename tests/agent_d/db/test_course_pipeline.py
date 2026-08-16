"""Agent D against real Postgres: store SQL, ESCO mapping tiers, supply
aggregation, and the demand-vs-supply join that is the whole point.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from agents.agent_b_job_ingest.esco_map import sync_taxonomy
from agents.agent_d_course_ingest.aggregate import recompute_supply
from agents.agent_d_course_ingest.db.store import CourseStore
from agents.agent_d_course_ingest.esco_map import map_new_course_skills
from agents.agent_d_course_ingest.pipeline import CoursePipeline
from agents.agent_d_course_ingest.prompts.extraction import EXTRACTION_PROMPT
from agents.agent_d_course_ingest.records import PersistedCourse
from agents.agent_d_course_ingest.schemas import CourseExtraction
from agents.agent_d_course_ingest.sources.base import RawCourse
from shared.config import Config
from shared.llm import structured
from tests.fake_embedder import FakeEmbedder
from tests.fake_llm import FakeStructuredLLM

ESCO_FIXTURE = Path(__file__).parents[2] / "agent_b" / "fixtures" / "esco_skills_sample.csv"
# today, not a hardcoded date: courses are first_seen at the server's now(), and
# the supply window ends at as_of — a fixed past date would exclude today's rows.
AS_OF = date.today()


def _cfg(store):
    # map_skills_to_esco opens its OWN connection via config.database_url, so in
    # tests it must be pointed at the test DB, not the env's dev database.
    return Config(database_url=store.dsn)


def _sync_esco(store):
    # The taxonomy sync is Agent B's (it owns esco_skills/esco_labels); Agent D
    # only READS them via the shared resolver. Sync through a JobStore on the
    # same database, exactly as production does via `agent-b --esco-sync`.
    from agents.agent_b_job_ingest.db import JobStore

    with JobStore(store.dsn) as jb:
        sync_taxonomy(jb, FakeEmbedder(), path=ESCO_FIXTURE, version="test-1")


def _course(url, **kw):
    return RawCourse(
        source=kw.pop("source", "coursera"), source_group=kw.pop("group", "coursera"),
        source_type=kw.pop("stype", "api"), source_url=url,
        name=kw.pop("name", "Intro to Accounting"),
        raw_description=kw.pop("body", "A course covering accounting and data analysis."),
        provider=kw.pop("provider", "IBM"), primary_language="en",
    )


def _pipeline(store, llm=None, embedder=None):
    llm = llm or FakeStructuredLLM(
        CourseExtraction=CourseExtraction(taught_skills=["accounting", "data analysis"])
    )
    return CoursePipeline(
        store=store, extractor=EXTRACTION_PROMPT | structured(llm, CourseExtraction),
        embedder=embedder or FakeEmbedder(), config=Config(), model_name="fake",
    )


def _row(cid, **kw):
    base = dict(course_id=cid, source="coursera", source_group="coursera", source_type="api",
                source_url=f"https://c.test/{cid}", name=f"Course {cid}",
                raw_description="d", content_hash=f"h_{cid}")
    base.update(kw)
    return PersistedCourse(**base)


# ---------------------------------------------------------------------------
# store SQL
# ---------------------------------------------------------------------------
def test_upsert_then_a_second_run_reports_unchanged(store):
    llm, emb = FakeStructuredLLM(CourseExtraction=CourseExtraction(taught_skills=["accounting"])), FakeEmbedder()
    batch = [_course("https://c.test/a"), _course("https://c.test/b", name="Data 101")]

    first = _pipeline(store, llm, emb).run(batch)
    assert first.written == 2

    llm2, emb2 = FakeStructuredLLM(), FakeEmbedder()
    second = _pipeline(store, llm2, emb2).run(batch)
    assert second.unchanged == 2 and second.extractions == 0 and second.embeddings == 0
    assert llm2.calls == [] and emb2.embed_calls == 0


def test_refresh_volatile_updates_price_and_rating_without_touching_embedding(store):
    """The lightweight every-cycle path: volatile columns + price_observed_at
    change; the embedding (the expensive content-gated artifact) does not."""
    vec = [0.25] * 1536
    store.upsert_batch([_row("c", embedding=vec, rating=4.0)])
    store.connect().commit()

    store.refresh_volatile([{
        "course_id": "c", "volatile_observed": True,
        "rating": 4.8, "review_count": 500, "enrollment_count": 9000,
        "last_updated": None, "price_amount": 0.0, "price_currency": None, "price_is_free": True,
    }])
    store.connect().commit()

    conn = store.connect()
    with conn.cursor() as cur:
        cur.execute("SELECT rating, review_count, enrollment_count, price_is_free, price_amount, "
                    "price_observed_at, embedding FROM courses WHERE course_id='c'")
        row = cur.fetchone()
    assert float(row["rating"]) == 4.8
    assert row["review_count"] == 500 and row["enrollment_count"] == 9000
    assert row["price_is_free"] is True and float(row["price_amount"]) == 0.0
    assert row["price_observed_at"] is not None
    assert row["embedding"] is not None, "refresh_volatile disturbed the embedding"


def test_an_unobserved_refresh_preserves_the_stored_quality_signals(store):
    """THE regression that matters: enrichment failing must not erase good data.

    A robots refusal, a timeout or a Coursera layout change yields a course with
    every volatile field None. Writing those Nones destroyed ratings the system
    could not re-derive — and silently, because the row still looked fine. On the
    live corpus 426 of 676 courses already had no rating; one bad cycle would
    have taken the remaining 250.
    """
    vec = [0.25] * 1536
    store.upsert_batch([_row("c", embedding=vec, rating=4.7)])
    store.connect().commit()

    # Exactly what a failed enrichment produces: nothing observed, all None.
    store.refresh_volatile([{
        "course_id": "c", "volatile_observed": False,
        "rating": None, "review_count": None, "enrollment_count": None,
        "last_updated": None, "price_amount": None, "price_currency": None,
        "price_is_free": None,
    }])
    store.connect().commit()

    conn = store.connect()
    with conn.cursor() as cur:
        cur.execute("SELECT rating, last_seen_at, missed_cycles FROM courses WHERE course_id='c'")
        row = cur.fetchone()
    assert float(row["rating"]) == 4.7, "a failed lookup erased a stored rating"
    # The seen/staleness touch still applies — the course WAS observed to exist,
    # we just could not read its page.
    assert row["missed_cycles"] == 0


def test_neardup_similarity_direction(store):
    """1 - distance; identical vector ~1.0, orthogonal ~0.0 â€” the classic bug."""
    v0 = [0.0] * 1536; v0[0] = 1.0
    v1 = [0.0] * 1536; v1[1] = 1.0
    store.upsert_batch([_row("x", embedding=v0)])
    store.connect().commit()

    same = store.find_neardup_candidates(v0, recent_days=30, limit=5, exclude_id="q")
    assert same and abs(same[0]["similarity"] - 1.0) < 1e-6
    orth = store.find_neardup_candidates(v1, recent_days=30, limit=5, exclude_id="q")
    assert orth and abs(orth[0]["similarity"]) < 1e-6


# ---------------------------------------------------------------------------
# ESCO mapping tiers (course_esco_map, never skill_esco_map)
# ---------------------------------------------------------------------------
def test_course_skills_map_through_the_shared_tiers_into_course_esco_map(store):
    _sync_esco(store)
    _pipeline(store).run([_course("https://c.test/a")])  # skills: accounting, data analysis

    summary = map_new_course_skills(store, FakeEmbedder(), _cfg(store))
    assert summary.exact >= 1        # accounting -> preferred label
    assert summary.alt_label >= 1    # data analysis -> alt of "analyse data"

    conn = store.connect()
    with conn.cursor() as cur:
        cur.execute("SELECT skill_key, method, esco_uri FROM course_esco_map ORDER BY skill_key")
        rows = {r["skill_key"]: r for r in cur.fetchall()}
        cur.execute("SELECT count(*) AS n FROM skill_esco_map")
        assert cur.fetchone()["n"] == 0, "Agent D wrote into Agent B's skill_esco_map"

    assert rows["accounting"]["method"] == "exact"
    assert rows["data analysis"]["method"] == "alt_label"


# ---------------------------------------------------------------------------
# supply aggregation + esco_code fill
# ---------------------------------------------------------------------------
def test_supply_counts_courses_and_fills_esco_code(store):
    _sync_esco(store)
    _pipeline(store).run([
        _course("https://c.test/a", name="Accounting A", provider="IBM"),
        _course("https://c.test/b", name="Accounting B", provider="Google"),
    ])
    map_new_course_skills(store, FakeEmbedder(), _cfg(store))
    with store.transaction():
        recompute_supply(store, Config(), as_of=AS_OF)

    conn = store.connect()
    with conn.cursor() as cur:
        cur.execute("SELECT skill, course_count, provider_count, esco_code, low_confidence "
                    "FROM skill_supply_stats WHERE skill_key = 'accounting'")
        row = cur.fetchone()
    assert row["course_count"] == 2
    assert row["provider_count"] == 2               # IBM + Google
    assert row["esco_code"].endswith("/0004")       # accounting concept
    assert row["low_confidence"] is True            # 2 < course_low_confidence_min_courses (3)


def test_a_course_older_than_the_window_is_still_supply(store):
    """SUPPLY IS A STOCK, NOT A FLOW. The aggregation copied Agent B's
    posting-date window, which is right for vacancies (one from 90 days ago is
    not current demand) and wrong for courses (one from 90 days ago still teaches
    what it teaches). With the filter in place, every course would have dropped
    out of the table 90 days after we first saw it and course_count would decay
    toward zero while nothing about the real supply changed — on the live corpus,
    starting 2026-10-22.
    """
    _sync_esco(store)
    store.upsert_batch([_row("ancient", taught_skills=["accounting"])])
    with store.connect().cursor() as cur:
        # Older than course_window_days (90) — and still an active, listed course.
        cur.execute("UPDATE courses SET first_seen_at = now() - interval '200 days' "
                    "WHERE course_id = 'ancient'")
    store.connect().commit()

    map_new_course_skills(store, FakeEmbedder(), _cfg(store))
    with store.transaction():
        recompute_supply(store, Config(), as_of=AS_OF)

    with store.connect().cursor() as cur:
        cur.execute("SELECT course_count FROM skill_supply_stats WHERE skill_key='accounting'")
        row = cur.fetchone()
    assert row is not None, "a course aged out of the supply table it should still be in"
    assert row["course_count"] == 1


def test_a_rerun_clears_rows_it_no_longer_reproduces(store):
    """`replace_stats_window` never replaced: it only INSERT..ON CONFLICTed, so a
    skill whose last course disappeared kept its row and was served forever as
    current supply. Agent B measured 114 such phantom rows before its own fix."""
    _sync_esco(store)
    store.upsert_batch([_row("a", taught_skills=["accounting", "data analysis"])])
    store.connect().commit()
    map_new_course_skills(store, FakeEmbedder(), _cfg(store))
    with store.transaction():
        recompute_supply(store, Config(), as_of=AS_OF)

    with store.connect().cursor() as cur:
        cur.execute("SELECT count(*) AS n FROM skill_supply_stats WHERE window_end=%s", (AS_OF,))
        assert cur.fetchone()["n"] == 2

    # The course now teaches only one of them; the other has no supply at all.
    store.upsert_batch([_row("a", taught_skills=["accounting"])])
    store.connect().commit()
    with store.transaction():
        recompute_supply(store, Config(), as_of=AS_OF)

    with store.connect().cursor() as cur:
        cur.execute("SELECT skill_key FROM skill_supply_stats WHERE window_end=%s", (AS_OF,))
        keys = {r["skill_key"] for r in cur.fetchall()}
    assert keys == {"accounting"}, f"a phantom row survived the rerun: {keys}"


def test_the_supply_count_publishes_its_denominator(store):
    """`course_count = 3` is uninterpretable without "out of how many" — the same
    omission Agent B's audit fixed with sector_volume."""
    _sync_esco(store)
    store.upsert_batch([
        _row("a", taught_skills=["accounting"]),
        _row("b", taught_skills=["data analysis"], source_url="https://c.test/b"),
    ])
    store.connect().commit()
    map_new_course_skills(store, FakeEmbedder(), _cfg(store))
    with store.transaction():
        recompute_supply(store, Config(), as_of=AS_OF)

    with store.connect().cursor() as cur:
        cur.execute("SELECT course_count, total_courses, courses_without_provider "
                    "FROM skill_supply_stats WHERE skill_key='accounting'")
        row = cur.fetchone()
    assert row["course_count"] == 1 and row["total_courses"] == 2


def test_one_concept_counts_a_course_once_however_many_ways_it_phrases_it(store):
    """The fan-out fix. skill_supply_stats is keyed by skill_key and several keys
    map to one ESCO concept (measured: one concept carried SEVEN), so the
    documented `USING (esco_code)` join multiplied — 131 matched concepts became
    273 rows and inflated summed demand 237 -> 407 (+72%). No rollup of the skill
    grain can fix that: summing double-counts a course teaching two phrasings,
    max undercounts. The concept grain counts DISTINCT courses instead.
    """
    _sync_esco(store)
    # One course, one concept, two phrasings that both map to it.
    store.upsert_batch([_row("a", taught_skills=["accounting", "Accounting"])])
    store.connect().commit()
    map_new_course_skills(store, FakeEmbedder(), _cfg(store))
    with store.transaction():
        recompute_supply(store, Config(), as_of=AS_OF)

    with store.connect().cursor() as cur:
        cur.execute("SELECT esco_code, course_count, variant_keys FROM concept_supply_stats "
                    "WHERE window_end=%s AND esco_code LIKE %s", (AS_OF, "%/0004"))
        rows = cur.fetchall()
    assert len(rows) == 1, "one concept produced several supply rows"
    assert rows[0]["course_count"] == 1, "one course was counted twice for phrasing it twice"


def test_rejected_and_duplicate_courses_are_not_counted(store):
    _sync_esco(store)
    store.upsert_batch([
        _row("good", taught_skills=["accounting"]),
        _row("bad", taught_skills=["accounting"], status="rejected",
             source_url="https://c.test/bad"),
        _row("dup", taught_skills=["accounting"], duplicate_of="good",
             source_url="https://c.test/dup"),
    ])
    store.connect().commit()
    map_new_course_skills(store, FakeEmbedder(), _cfg(store))
    with store.transaction():
        recompute_supply(store, Config(), as_of=AS_OF)

    conn = store.connect()
    with conn.cursor() as cur:
        cur.execute("SELECT course_count FROM skill_supply_stats WHERE skill_key='accounting'")
        assert cur.fetchone()["course_count"] == 1


# ---------------------------------------------------------------------------
# the payoff: demand meets supply on esco_code
# ---------------------------------------------------------------------------
def test_demand_and_supply_join_on_esco_code(store):
    _sync_esco(store)
    # supply: two courses teach accounting
    _pipeline(store, FakeStructuredLLM(CourseExtraction=CourseExtraction(taught_skills=["accounting"]))).run([
        _course("https://c.test/a", name="Acc A"), _course("https://c.test/b", name="Acc B"),
    ])
    map_new_course_skills(store, FakeEmbedder(), _cfg(store))
    with store.transaction():
        recompute_supply(store, Config(), as_of=AS_OF)

    # demand: a job stat row for the SAME esco concept (seed directly)
    conn = store.connect()
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO skill_demand_stats (sector, skill, skill_key, esco_code, window_start, "
            "window_end, frequency_count, trend) VALUES "
            "('2','Accounting','accounting','http://example.test/esco/skill/0004',"
            "'2026-06-22','2026-07-22',9,'stable')"
        )
        conn.commit()
        cur.execute(
            """
            SELECT d.frequency_count AS demand, s.course_count AS supply
              FROM skill_demand_stats d
              JOIN skill_supply_stats s USING (esco_code)
             WHERE d.esco_code = 'http://example.test/esco/skill/0004'
            """
        )
        row = cur.fetchone()
    assert row["demand"] == 9 and row["supply"] == 2


# ---------------------------------------------------------------------------
# retention — the supply side of the same unbounded-growth problem
# ---------------------------------------------------------------------------
def test_supply_snapshots_are_pruned_across_both_grains(store):
    """Both grains together: a consumer joining a current concept count against
    a stale skill count would be comparing two different days."""
    from datetime import timedelta

    conn = store.connect()
    with conn.cursor() as cur:
        for i in range(10):
            w_end = AS_OF - timedelta(days=i)
            cur.execute(
                "INSERT INTO skill_supply_stats (skill, skill_key, window_start, "
                "window_end, course_count) VALUES ('A','a',%s,%s,1)",
                (w_end - timedelta(days=90), w_end))
            cur.execute(
                "INSERT INTO concept_supply_stats (esco_code, window_start, "
                "window_end, course_count) VALUES ('uri:a',%s,%s,1)",
                (w_end - timedelta(days=90), w_end))
    conn.commit()

    removed = store.prune_stats_windows(keep=8)
    conn.commit()

    with conn.cursor() as cur:
        cur.execute("SELECT count(DISTINCT window_end) AS n FROM skill_supply_stats")
        skill_windows = cur.fetchone()["n"]
        cur.execute("SELECT count(DISTINCT window_end) AS n FROM concept_supply_stats")
        concept_windows = cur.fetchone()["n"]

    assert skill_windows == 8 and concept_windows == 8
    assert removed == 4, "both grains should have lost two windows each"


def test_supply_retention_refuses_to_go_below_the_floor(store):
    import pytest as _pytest

    with _pytest.raises(ValueError, match="floor"):
        store.prune_stats_windows(keep=1)


# ---------------------------------------------------------------------------
# the connection itself
# ---------------------------------------------------------------------------
def test_a_bare_write_commits_without_an_explicit_transaction(store):
    """The bug this store had, and that Agent B was bitten by twice.

    Without autocommit, psycopg opens an implicit transaction on the FIRST
    statement — and a cycle reads before it writes — so every later
    `with conn.transaction()` nests as a SAVEPOINT, and releasing a savepoint
    commits nothing. The run logs its work, raises nothing, and the table is
    unchanged. A verification script here reported 148 courses refreshed and
    wrote none.
    """
    store.upsert_batch([_row("c_bare", name="Bare write")])

    fresh = CourseStore(store.dsn)
    try:
        assert fresh.lookup_hashes(["c_bare"]), "the write never committed"
    finally:
        fresh.close()


def test_a_failed_batch_still_writes_nothing(store):
    """Autocommit must not have cost the rollback it looks like it costs.

    Writers wrap their work in `transaction()`, which under autocommit opens a
    REAL transaction rather than a savepoint — so a failure mid-batch still
    leaves the table as it was.
    """
    before = len(store.lookup_hashes(["c_ok", "c_boom"]))

    with pytest.raises(RuntimeError):
        with store.transaction():
            store.upsert_batch([_row("c_ok", name="Fine")])
            raise RuntimeError("something failed after the write")

    assert len(store.lookup_hashes(["c_ok", "c_boom"])) == before, (
        "a failed batch left rows behind")
