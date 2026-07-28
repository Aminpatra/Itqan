"""What the gap report claims must be true — the logical defects, pinned.

Each case here was measured on the real 2026-07-28 run before it was fixed. The
common thread is that the old arithmetic answered a question it had no data for
and returned a confident number anyway.
"""

from __future__ import annotations

import math

import pytest

from tests.agent_c.test_gap_analysis import (
    CANDIDATE_AXIS,
    PinnedEmbedder,
    _pin,
    _posting,
    _stat,
    profile_path,          # noqa: F401 - fixture
    run_graph,
)


# ---------------------------------------------------------------------------
# 0.0 used to mean three different things
# ---------------------------------------------------------------------------
def test_a_posting_with_no_parsable_requirements_scores_null_not_zero(
    profile_path, tmp_path  # noqa: F811
):
    """0.0 is the BEST value on the scale. Emitting it for "this posting listed
    nothing we could parse" was a false claim of a perfect fit — and on the live
    run 5 of 15 jobs scored 0.0 having matched nothing, two listing no
    requirements at all."""
    postings = [_posting(f"p{i}", 0.85, skills=[]) for i in range(5)]
    state = run_graph(profile_path, tmp_path, postings=postings, stats=[])

    job = state["matched_jobs"][0]
    assert job["gap_score"] is None
    assert job["insufficient_data"] is True
    assert state["aggregate"]["jobs_without_parsable_requirements"] == 5
    assert any("perfect fit" in w for w in state["warnings"])


def test_a_job_where_nothing_resolved_scores_null_not_zero(
    profile_path, tmp_path  # noqa: F811
):
    """Every requirement unresolved makes the point estimate arithmetically 0.0
    and semantically empty — not one was settled either way. Reported as null with
    a [0, 1] range, because "0.00" reads as a perfect fit at a glance."""
    embedder = PinnedEmbedder({"alpha": CANDIDATE_AXIS, "unsure": _pin(0.70)})
    postings = [_posting(f"p{i}", 0.85, skills=["unsure"]) for i in range(5)]
    state = run_graph(profile_path, tmp_path, postings=postings, stats=[],
                      embedder=embedder)

    job = state["matched_jobs"][0]
    assert job["gap_score"] is None
    assert job["gap_score_range"] == [0.0, 1.0]
    assert job["insufficient_data"] is True


def test_undefined_jobs_are_excluded_from_the_average(profile_path, tmp_path):  # noqa: F811
    """The published average was 0.5533 where the honest figure over jobs with
    real requirements was 0.757, because the undefined 0.0s were averaged in."""
    embedder = PinnedEmbedder({"alpha": CANDIDATE_AXIS, "far-off": _pin(0.10)})
    postings = [_posting("real", 0.85, skills=["far-off"])] + [
        _posting(f"empty{i}", 0.85, skills=[]) for i in range(4)
    ]
    state = run_graph(profile_path, tmp_path, postings=postings, stats=[],
                      embedder=embedder)

    # One real job, entirely missing -> 1.0. The four empties must not dilute it.
    assert state["aggregate"]["average_gap_score"] == 1.0
    assert state["aggregate"]["jobs_scored"] == 1


def test_the_score_publishes_its_uncertainty_as_an_interval(
    profile_path, tmp_path  # noqa: F811
):
    """possible_match sits in the denominator only, which is NOT neutral — it
    silently decides "not missing". The interval says what that means: at best
    `lower`, at worst `upper`."""
    embedder = PinnedEmbedder({
        "alpha": CANDIDATE_AXIS, "far-off": _pin(0.10), "unsure": _pin(0.70),
    })
    postings = [_posting(f"p{i}", 0.85, skills=["far-off", "unsure"]) for i in range(5)]
    state = run_graph(profile_path, tmp_path, postings=postings, stats=[],
                      embedder=embedder)

    job = state["matched_jobs"][0]
    lower, upper = job["gap_score_range"]
    assert lower == job["gap_score"] == 0.5   # one of two weighted equally
    assert upper == 1.0                        # the unresolved one counted as missing
    assert "unsure" in state["aggregate"]["unresolved_skills"]


# ---------------------------------------------------------------------------
# ranking
# ---------------------------------------------------------------------------
def test_one_concept_phrased_three_ways_becomes_one_gap(profile_path, tmp_path):  # noqa: F811
    """`professional communication` (190), `communication skills` (55) and
    `communication` (33) occupied three of the ten slots a candidate sees, and
    Agent E duly spent three of ten recommendations on them."""
    embedder = PinnedEmbedder({
        "alpha": CANDIDATE_AXIS,
        "communication": _pin(0.10),
        "communication skills": _pin(0.10),
        "professional communication": _pin(0.10),
        "welding": _pin(0.10),
    })
    postings = [
        _posting("p0", 0.85, skills=["communication", "welding"]),
        _posting("p1", 0.85, skills=["communication skills"]),
        _posting("p2", 0.85, skills=["professional communication"]),
        _posting("p3", 0.85, skills=["welding"]),
        _posting("p4", 0.85, skills=["welding"]),
    ]
    state = run_graph(profile_path, tmp_path, postings=postings, stats=[],
                      embedder=embedder)

    skills = state["aggregate"]["most_common_missing_skills"]
    comms = [s for s in skills if "communication" in s]
    assert len(comms) == 1, f"the same concept still occupies several slots: {comms}"
    merged = next(d for d in state["aggregate"]["missing_skill_details"]
                  if "communication" in d["skill"])
    assert merged["also_phrased_as"], "the merged phrasings are not disclosed"


def test_demand_is_scoped_to_the_jobs_own_sector(profile_path, tmp_path):  # noqa: F811
    """Summing a skill across all nine ISCO groups made `professional
    communication` weigh 95 when its largest single sector was 58 — a nationwide
    generic phrase outranking a candidate's real sector-specific gaps."""
    embedder = PinnedEmbedder({"alpha": CANDIDATE_AXIS, "widespread": _pin(0.10)})
    postings = [_posting(f"p{i}", 0.85, sector="3", skills=["widespread"])
                for i in range(5)]
    stats = [
        _stat("widespread", sector="3", freq=2),    # this job's sector
        _stat("widespread", sector="2", freq=90),   # a huge, irrelevant sector
    ]
    state = run_graph(profile_path, tmp_path, postings=postings, stats=stats,
                      embedder=embedder)

    detail = state["aggregate"]["missing_skill_details"][0]
    assert detail["priority_score"] == pytest.approx(math.log1p(2), abs=1e-4)
    assert detail["priority_score"] < math.log1p(90), "another sector's demand leaked in"


# ---------------------------------------------------------------------------
# evidence quality
# ---------------------------------------------------------------------------
def test_a_weakly_evidenced_skill_cannot_close_a_gap_outright(
    profile_path, tmp_path, monkeypatch  # noqa: F811
):
    """An unverified claim_only skill cancelled a requirement exactly as forcefully
    as a project-evidenced one — and the error shrinks the gap, which is the
    harmful direction."""
    import json
    from pathlib import Path

    from shared.contracts import CandidateProfile

    profile = CandidateProfile(
        run_id="c", generated_at="2026-07-23T00:00:00+00:00",
        candidate={"contact": {"location": "Muscat"}},
        skills={"accepted": [
            {"name": "Alpha", "quality": "low", "evidence_type": "claim_only"},
        ], "rejected": []},
        summary={"headline": "Test Candidate"},
    )
    path = Path(tmp_path) / "weak_profile.json"
    path.write_text(profile.model_dump_json(), encoding="utf-8")

    # 'alpha' is an EXACT string match, which would normally be `matched`.
    postings = [_posting(f"p{i}", 0.85, skills=["Alpha"]) for i in range(5)]
    state = run_graph(str(path), tmp_path, postings=postings, stats=[])

    job = state["matched_jobs"][0]
    assert job["matched_skills"] == [], "a claim_only skill produced a confident match"
    assert job["possible_match_skills"] == ["Alpha"]


# ---------------------------------------------------------------------------
# the report is usable
# ---------------------------------------------------------------------------
def test_a_job_can_actually_be_opened(profile_path, tmp_path):  # noqa: F811
    """A report whose purpose is "roles you nearly fit" published a job_id and no
    link; source_url, posted_date, seniority and location were all retrieved and
    discarded."""
    postings = [_posting(f"p{i}", 0.85) for i in range(5)]
    state = run_graph(profile_path, tmp_path, postings=postings, stats=[])

    job = state["matched_jobs"][0]
    assert job["source_url"].startswith("https://")
    for field in ("posted_date", "seniority_level", "location"):
        assert field in job


def test_every_verdict_records_how_it_was_reached(profile_path, tmp_path):  # noqa: F811
    """So a human can tell a near miss from a genuine gap, and audit which tier
    (or the LLM) decided each requirement."""
    embedder = PinnedEmbedder({"alpha": CANDIDATE_AXIS, "far-off": _pin(0.10)})
    postings = [_posting(f"p{i}", 0.85, skills=["far-off"]) for i in range(5)]
    state = run_graph(profile_path, tmp_path, postings=postings, stats=[],
                      embedder=embedder)

    entry = state["matched_jobs"][0]["skill_resolution"][0]
    assert entry["resolved_by"] in {"esco", "exact", "containment", "cosine", "llm"}
    assert entry["best_similarity"] is not None


def test_the_envelope_records_its_own_calibration(profile_path, tmp_path):  # noqa: F811
    """An output could not be reproduced or re-interpreted from itself."""
    import json
    from pathlib import Path

    postings = [_posting(f"p{i}", 0.85) for i in range(5)]
    run_graph(profile_path, tmp_path, postings=postings, stats=[])
    out = json.loads((Path(tmp_path) / "t" / "skill_gap.json").read_text(encoding="utf-8"))

    cal = out["calibration"]
    for key in ("agent_c_match_threshold", "agent_c_skill_match",
                "agent_c_skill_possible", "agent_c_min_usable_postings",
                "agent_c_llm_matching"):
        assert key in cal
