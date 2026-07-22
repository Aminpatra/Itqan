"""The aggregation query against real Postgres, with hand-computed expectations.

Every count here is worked out by hand in the test body, because the whole point
of the query is that its numbers are trustworthy — a test that trusted the
query's own arithmetic would prove nothing. The eligibility exclusions get one
assertion each: each is a distinct way a real posting could silently become a
fabricated statistic.
"""

from __future__ import annotations

from datetime import date

import pytest

from agents.agent_b_job_ingest.aggregate import recompute_stats
from shared.config import Config

AS_OF = date(2026, 7, 22)
IN_WINDOW = "2026-07-01"   # inside (2026-06-22, 2026-07-22]
PRIOR = "2026-06-01"       # inside (2026-05-23, 2026-06-22]


def _ins(store, pid, *, sector="2", skills=("Python",), country="OM",
         intent="vacancy", poster="company", status="active",
         posted=IN_WINDOW, dup=None, title=None):
    conn = store.connect()
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO job_postings
              (posting_id, source, source_group, source_type, source_url, title,
               raw_description, content_hash, sector, required_skills, country,
               listing_intent, poster_type, status, posted_date, duplicate_of)
            VALUES (%s,'s','g','blogger_feed',%s,%s,'d',%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """,
            (pid, f"https://e.test/{pid}", title or f"Job {pid}", f"h_{pid}",
             sector, list(skills), country, intent, poster, status, posted, dup),
        )
    conn.commit()


def _stats(store, sector, skill_key):
    conn = store.connect()
    with conn.cursor() as cur:
        cur.execute(
            "SELECT * FROM skill_demand_stats WHERE sector=%s AND skill_key=%s "
            "AND window_end=%s",
            (sector, skill_key, AS_OF),
        )
        return cur.fetchone()


def _run(store):
    return recompute_stats(store, Config(), as_of=AS_OF)


# ---------------------------------------------------------------------------
# frequency counts count only eligible postings
# ---------------------------------------------------------------------------
def test_frequency_counts_exactly_the_eligible_postings(store):
    # 10 eligible current postings in sector 2; 6 mention Python, all 10 mention SQL.
    for i in range(6):
        _ins(store, f"c{i}", skills=("Python", "SQL"))
    for i in range(6, 10):
        _ins(store, f"c{i}", skills=("SQL",))

    # Noise that must NOT be counted — one of every ineligibility reason, each
    # carrying Python so a leak would show up as Python == 7.
    _ins(store, "canon_for_dup", skills=("SQL",))
    _ins(store, "seeker", skills=("Python",), intent="seeking")
    _ins(store, "unknown_poster", skills=("Python",), poster="unknown")
    _ins(store, "dup", skills=("Python",), dup="canon_for_dup")
    _ins(store, "abroad", skills=("Python",), country="AE")
    _ins(store, "no_sector", skills=("Python",), sector=None)
    _ins(store, "rejected", skills=("Python",), status="rejected")
    _ins(store, "stale", skills=("Python",), status="stale")

    _run(store)

    python = _stats(store, "2", "python")
    assert python["frequency_count"] == 6, "an ineligible posting leaked into the count"
    sql = _stats(store, "2", "sql")
    assert sql["frequency_count"] == 11  # 10 eligible + canon_for_dup


@pytest.mark.parametrize(
    "reason,kw",
    [
        ("seeking", {"intent": "seeking"}),
        ("unknown_poster", {"poster": "unknown"}),
        ("out_of_scope", {"country": "AE"}),
        ("null_sector", {"sector": None}),
        ("rejected", {"status": "rejected"}),
        ("stale", {"status": "stale"}),
    ],
)
def test_a_single_ineligible_posting_produces_no_stats(store, reason, kw):
    _ins(store, "only", skills=("Python",), **kw)
    summary = _run(store)
    assert summary.rows_written == 0, f"{reason} posting was aggregated"


def test_a_duplicate_is_not_counted(store):
    _ins(store, "canon", skills=("Python",))
    _ins(store, "dup", skills=("Python",), dup="canon")
    _run(store)
    assert _stats(store, "2", "python")["frequency_count"] == 1


# ---------------------------------------------------------------------------
# trend
# ---------------------------------------------------------------------------
def test_rising_when_current_exceeds_prior_by_the_ratio(store):
    for i in range(6):
        _ins(store, f"cur{i}", skills=("Python",))
    for i in range(5):
        _ins(store, f"pri{i}", skills=("Python",), posted=PRIOR)

    _run(store)
    row = _stats(store, "2", "python")
    assert row["frequency_count"] == 6 and row["prior_frequency_count"] == 5
    assert row["trend"] == "rising"  # 6/5 = 1.20


def test_falling_when_current_drops_below_the_ratio(store):
    for i in range(5):
        _ins(store, f"cur{i}", skills=("Excel",))
    for i in range(10):
        _ins(store, f"pri{i}", skills=("Excel",), posted=PRIOR)

    _run(store)
    row = _stats(store, "2", "excel")
    assert row["trend"] == "falling"  # 5/10 = 0.50


def test_a_low_volume_skill_is_always_stable(store):
    """1 -> 2 postings is +100% noise, not a rising trend. Below trend_min_volume
    the label is pinned to stable so noise is never published as a finding."""
    for i in range(2):
        _ins(store, f"cur{i}", skills=("Rust",))
    _ins(store, "pri0", skills=("Rust",), posted=PRIOR)

    _run(store)
    assert _stats(store, "2", "rust")["trend"] == "stable"  # freq 2 < trend_min 5


# ---------------------------------------------------------------------------
# low_confidence
# ---------------------------------------------------------------------------
def test_low_confidence_set_below_the_sector_floor_and_cleared_above(store):
    # sector 7: 3 postings -> below the floor of 10 -> low_confidence
    for i in range(3):
        _ins(store, f"s7_{i}", sector="7", skills=("Welding",))
    # sector 2: 10 postings -> at/above the floor -> not low_confidence
    for i in range(10):
        _ins(store, f"s2_{i}", sector="2", skills=("Python",))

    _run(store)
    assert _stats(store, "7", "welding")["low_confidence"] is True
    assert _stats(store, "2", "python")["low_confidence"] is False


def test_low_confidence_recomputed_across_the_boundary(store):
    """A sector that gains postings between runs must lose the flag — it is
    written on insert AND recomputed on conflict."""
    for i in range(9):
        _ins(store, f"a{i}", skills=("Python",))
    _run(store)
    assert _stats(store, "2", "python")["low_confidence"] is True

    _ins(store, "a9", skills=("Python",))   # now 10
    _run(store)
    assert _stats(store, "2", "python")["low_confidence"] is False


# ---------------------------------------------------------------------------
# co-occurrence + samples + evidence
# ---------------------------------------------------------------------------
def test_co_occurring_skills_and_sample_postings(store):
    for i in range(6):
        _ins(store, f"c{i}", skills=("Python", "SQL"))

    _run(store)
    python = _stats(store, "2", "python")

    co = {c["skill"]: c["count"] for c in python["co_occurring_skills"]}
    assert co == {"SQL": 6}
    # sample_postings is evidence, capped at 5, each a {title, source_url}.
    samples = python["sample_postings"]
    assert len(samples) == 5
    assert all({"title", "source_url"} <= set(s) for s in samples)


def test_display_skill_uses_the_most_common_casing(store):
    for i in range(3):
        _ins(store, f"p{i}", skills=("Python",))
    _ins(store, "p_lower", skills=("python",))

    _run(store)
    # skill_key groups them; the displayed label is the modal casing.
    row = _stats(store, "2", "python")
    assert row["frequency_count"] == 4
    assert row["skill"] == "Python"


# ---------------------------------------------------------------------------
# empty and history
# ---------------------------------------------------------------------------
def test_an_empty_window_writes_nothing_and_does_not_error(store):
    summary = _run(store)
    assert summary.rows_written == 0
    assert summary.sectors_with_current_demand == 0


def test_a_sector_that_had_demand_and_now_has_none_is_reported_zeroed(store):
    _ins(store, "pri0", skills=("Python",), posted=PRIOR)  # prior only

    summary = _run(store)
    assert summary.sectors_zeroed == 1
    assert summary.sectors_with_current_demand == 0


def test_history_is_appended_not_overwritten_across_windows(store):
    for i in range(3):
        _ins(store, f"c{i}", skills=("Python",))
    recompute_stats(store, Config(), as_of=date(2026, 7, 22))
    recompute_stats(store, Config(), as_of=date(2026, 7, 23))

    conn = store.connect()
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) AS n FROM skill_demand_stats WHERE skill_key='python'")
        assert cur.fetchone()["n"] == 2, "the second window overwrote the first"
