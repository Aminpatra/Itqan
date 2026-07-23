"""The Agent C read surface (`shared.job_market.export_for_agent_c`) against
real Postgres.

Each ineligibility gets one seeded row WITH an embedding, so a filter leak
surfaces as a row in the results rather than passing silently.
"""

from __future__ import annotations

import math
from datetime import date

from agents.agent_b_job_ingest.records import PersistedPosting
from shared.config import Config
from shared.job_market import export_for_agent_c

import pytest


def _config(store) -> Config:
    return Config(database_url=store.dsn)


def _unit(index: int) -> list[float]:
    vec = [0.0] * 1536
    vec[index] = 1.0
    return vec


def _blend(i: int, j: int, wi: float, wj: float) -> list[float]:
    vec = [0.0] * 1536
    vec[i], vec[j] = wi, wj
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


def _seed(store, rows) -> None:
    """Upsert AND COMMIT. The export opens its own connection - the realistic
    read path - so uncommitted seeds would be invisible to it."""
    store.upsert_batch(rows)
    store.connect().commit()


def _row(pid: str, **kw) -> PersistedPosting:
    base = dict(
        posting_id=pid, source="el7far", source_group="g", source_type="blogger_feed",
        source_url=f"https://e.test/{pid}", title=f"Role {pid}",
        raw_description="desc", content_hash=f"h_{pid}",
        sector=kw.pop("sector", "2"), country=kw.pop("country", "OM"),
        listing_intent=kw.pop("listing_intent", "vacancy"),
        poster_type=kw.pop("poster_type", "company"),
        embedding=kw.pop("embedding", _unit(0)),
    )
    base.update(kw)
    return PersistedPosting(**base)


def _stat(store, *, sector="2", skill="Python", window_end="2026-07-22", freq=5):
    conn = store.connect()
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO skill_demand_stats (sector, skill, skill_key, window_start, "
            "window_end, frequency_count, trend) VALUES (%s,%s,%s,%s,%s,%s,'stable')",
            (sector, skill, skill.lower(), "2026-06-22", window_end, freq),
        )
    conn.commit()


# ---------------------------------------------------------------------------
def test_empty_database_returns_empty_lists_not_errors(store):
    out = export_for_agent_c(_unit(0), config=_config(store))
    assert out == {"job_postings": [], "skill_demand_stats": []}


@pytest.mark.parametrize(
    "reason,kw",
    [
        ("seeker", {"listing_intent": "seeking"}),
        ("unknown_poster", {"poster_type": "unknown"}),
        ("rejected", {"status": "rejected"}),
        ("stale", {"status": "stale"}),
        ("out_of_scope", {"country": "AE"}),
    ],
)
def test_each_ineligible_posting_is_excluded(store, reason, kw):
    """One eligible + one ineligible with an IDENTICAL embedding: a filter leak
    would rank the ineligible row equally and it would appear."""
    _seed(store, [_row("good"), _row("bad", **kw)])

    out = export_for_agent_c(_unit(0), config=_config(store))

    ids = [p.posting_id for p in out["job_postings"]]
    assert ids == ["good"], f"{reason} posting leaked into retrieval"


def test_a_duplicate_is_excluded(store):
    _seed(store, [_row("canon")])
    _seed(store, [_row("dup", duplicate_of="canon")])

    out = export_for_agent_c(_unit(0), config=_config(store))
    assert [p.posting_id for p in out["job_postings"]] == ["canon"]


def test_results_descend_by_similarity_and_direction_is_right(store):
    """`<=>` is DISTANCE; the export must return 1 - distance. Identical vector
    first at ~1.0, then hand-computed blends in order."""
    _seed(store, [
        _row("identical", embedding=_unit(0)),
        _row("near", embedding=_blend(0, 1, 0.9, 0.1)),
        _row("far", embedding=_blend(0, 1, 0.3, 0.7)),
    ])

    out = export_for_agent_c(_unit(0), config=_config(store))
    postings = out["job_postings"]

    assert [p.posting_id for p in postings] == ["identical", "near", "far"]
    assert postings[0].similarity == pytest.approx(1.0, abs=1e-6)
    sims = [p.similarity for p in postings]
    assert sims == sorted(sims, reverse=True)
    # hand-computed: cos = 0.9/sqrt(0.82) and 0.3/sqrt(0.58)
    assert postings[1].similarity == pytest.approx(0.9 / math.sqrt(0.82), abs=1e-6)
    assert postings[2].similarity == pytest.approx(0.3 / math.sqrt(0.58), abs=1e-6)


def test_top_k_caps_the_result(store):
    _seed(store, [_row(f"p{i}") for i in range(6)])
    out = export_for_agent_c(_unit(0), top_k=3, config=_config(store))
    assert len(out["job_postings"]) == 3


def test_stats_come_from_the_latest_window_only(store):
    """History rows exist for audit; serving an old window as current demand is
    the stale-statistic mistake the schema's consumer contract warns about."""
    _stat(store, skill="OldSkill", window_end="2026-07-01", freq=99)
    _stat(store, skill="NewSkill", window_end="2026-07-22", freq=5)

    out = export_for_agent_c(_unit(0), config=_config(store))

    skills = [s.skill for s in out["skill_demand_stats"]]
    assert skills == ["NewSkill"], "a stale window leaked into the export"


def test_stats_sector_filter(store):
    _stat(store, sector="2", skill="Python")
    _stat(store, sector="5", skill="Sales")

    out = export_for_agent_c(_unit(0), isco08_sector="5", config=_config(store))
    assert [s.skill for s in out["skill_demand_stats"]] == ["Sales"]


def test_wrong_dimension_vector_fails_loudly_naming_both_lengths(store):
    with pytest.raises(ValueError, match="3 dimensions.*1536"):
        export_for_agent_c([0.1, 0.2, 0.3], config=_config(store))


def test_bad_sector_argument_is_rejected(store):
    with pytest.raises(ValueError, match="ISCO-08"):
        export_for_agent_c(_unit(0), isco08_sector="22", config=_config(store))
