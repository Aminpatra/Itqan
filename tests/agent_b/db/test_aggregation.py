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
#
# The prior count comes from the PREVIOUS WINDOW'S STORED ROW, not from
# re-deriving the past out of today's live postings. These tests therefore seed a
# stats row for an earlier window rather than inserting old postings.
#
# That distinction is the whole bug: the old query read the prior from `eligible`,
# which requires status='active'. In a test, freshly-inserted "prior" postings ARE
# active, so it passed. In production those postings were delisted and had gone
# stale within ~36h, so the prior was ALWAYS 0 — measured across all 1153 rows of
# the live table, with 23 of them published as 'rising' off zero observations.
# A test that seeds live prior postings cannot see that; this one can.
# ---------------------------------------------------------------------------
PRIOR_WINDOW_END = date(2026, 7, 21)   # the window before AS_OF


def _seed_prior_stats(store, sector, skill_key, freq, *, skill=None):
    conn = store.connect()
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO skill_demand_stats
              (sector, skill, skill_key, window_start, window_end,
               frequency_count, prior_frequency_count, trend, computed_at)
            VALUES (%s,%s,%s,%s,%s,%s,0,'stable',now())
            """,
            (sector, skill or skill_key, skill_key,
             PRIOR_WINDOW_END - (AS_OF - date(2026, 6, 22)), PRIOR_WINDOW_END, freq),
        )
    conn.commit()


def test_rising_when_current_exceeds_prior_by_the_ratio(store):
    for i in range(6):
        _ins(store, f"cur{i}", skills=("Python",))
    _seed_prior_stats(store, "2", "python", 5)

    _run(store)
    row = _stats(store, "2", "python")
    assert row["frequency_count"] == 6 and row["prior_frequency_count"] == 5
    assert row["trend"] == "rising"  # 6/5 = 1.20


def test_falling_when_current_drops_below_the_ratio(store):
    for i in range(5):
        _ins(store, f"cur{i}", skills=("Excel",))
    _seed_prior_stats(store, "2", "excel", 10)

    _run(store)
    row = _stats(store, "2", "excel")
    assert row["trend"] == "falling"  # 5/10 = 0.50


def test_a_low_volume_skill_on_both_sides_is_stable(store):
    """1 -> 2 postings is +100% noise, not a rising trend."""
    for i in range(2):
        _ins(store, f"cur{i}", skills=("Rust",))
    _seed_prior_stats(store, "2", "rust", 1)

    _run(store)
    assert _stats(store, "2", "rust")["trend"] == "stable"


def test_a_collapse_is_not_hidden_by_the_volume_floor(store):
    """The floor suppresses noise, not disappearances. 20 -> 4 was reported
    'stable' because only the CURRENT count was tested against the floor."""
    for i in range(4):
        _ins(store, f"cur{i}", skills=("Cobol",))
    _seed_prior_stats(store, "2", "cobol", 20)

    _run(store)
    assert _stats(store, "2", "cobol")["trend"] == "falling"


def test_no_prior_window_is_not_reported_as_rising(store):
    """'We have never measured this' is not a rise — it used to be published as
    one, on zero observations."""
    for i in range(6):
        _ins(store, f"cur{i}", skills=("Python",))

    _run(store)
    row = _stats(store, "2", "python")
    assert row["trend"] == "no_prior_data" and row["prior_frequency_count"] == 0


def test_a_skill_absent_from_the_prior_window_is_new_not_rising(store):
    for i in range(6):
        _ins(store, f"cur{i}", skills=("Python",))
    _seed_prior_stats(store, "2", "excel", 5)   # prior window exists, python absent

    _run(store)
    assert _stats(store, "2", "python")["trend"] == "new"


# ---------------------------------------------------------------------------
# the window is REPLACED, not merged into
# ---------------------------------------------------------------------------
def test_recomputing_a_window_drops_rows_it_no_longer_supports(store):
    """Measured on the live corpus before this fix: 114 rows survived from an
    earlier run of the same window, citing sample_postings URLs that no longer
    resolved — and Agent C was weighting on them."""
    _ins(store, "a", skills=("Python", "Cobol"))
    _run(store)
    assert _stats(store, "2", "cobol") is not None

    # The posting is re-extracted and no longer mentions Cobol.
    conn = store.connect()
    with conn.cursor() as cur:
        cur.execute("UPDATE job_postings SET required_skills = %s WHERE posting_id='a'",
                    (["Python"],))
    conn.commit()

    _run(store)
    assert _stats(store, "2", "python") is not None
    assert _stats(store, "2", "cobol") is None, "a stat no posting supports survived"


def test_denominators_are_published(store):
    """A count is not a rate without them."""
    _ins(store, "a", skills=("Python",))
    _ins(store, "b", skills=("Python",))
    _ins(store, "c", skills=("Excel",))

    _run(store)
    row = _stats(store, "2", "python")
    assert row["sector_volume"] == 3      # 3 eligible postings in sector 2
    assert row["distinct_posts"] == 2     # from 2 distinct source posts


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
