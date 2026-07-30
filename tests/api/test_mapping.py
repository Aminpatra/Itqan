"""The honest-gap layer, unit-tested.

No database and no HTTP: these are pure functions, and they are where an
integration lies if it is going to. Each case below is a shape the pipeline
really produces — several were found by running a real CV through the API.
"""

from __future__ import annotations

import pytest

from api.mapping import analysis_result, courses, dashboard, job_matches


# ---------------------------------------------------------------------------
# absence must survive the mapping
# ---------------------------------------------------------------------------
def test_a_missing_price_stays_null_and_never_becomes_free():
    """Measured: 0 of 1,999 Coursera courses publish a price. `0` would render
    as FREE — a different claim, and one the user would act on."""
    out = courses({"recommendations": [{
        "skill": "power bi", "no_course_found": False,
        "course": {"course_id": "c", "title": "T", "provider": "P", "url": "u",
                   "covers_other_skills": [], "quality": {"price": None}},
    }]})
    assert out[0]["price"] is None and out[0]["currency"] is None


def test_hours_stays_null_because_agent_d_stores_no_duration():
    """Coursera's `workload` is free text and multilingual — '2 heures',
    '4 weeks of study, 2-4 hours a week'. Reducing that to one number means
    choosing a point inside an 8-16 hour range."""
    out = courses({"recommendations": [{
        "skill": "x", "no_course_found": False,
        "course": {"course_id": "c", "title": "T", "provider": "P", "url": "u",
                   "covers_other_skills": [], "quality": {}},
    }]})
    assert out[0]["hours"] is None


def test_a_null_gap_score_becomes_a_null_readiness_not_a_zero():
    """Agent C publishes null when it had nothing to compute from. `0` would
    read as 'you match nothing' — the exact misreading its audit removed."""
    out = dashboard({}, {"aggregate": {"average_gap_score": None,
                                       "missing_skill_details": []}}, {})
    assert out["readiness"] is None
    assert "not enough" in out["readinessNote"].lower()


def test_readiness_is_the_complement_of_the_gap():
    out = dashboard({}, {"aggregate": {"average_gap_score": 0.42,
                                       "missing_skill_details": []}}, {})
    assert out["readiness"] == 58


def test_birth_date_is_always_null():
    assert analysis_result({"candidate": {"full_name": "X"}})["birthDate"] is None


# ---------------------------------------------------------------------------
# graduation date — found by running a real CV through the API
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("raw,want", [
    # THE bug: slicing [:7] published "Expecte". The month is now read as well —
    # the CV wrote it, and a bare year cannot populate an <input type="month">,
    # which is why an extracted graduation date still rendered as blank.
    ("Expected June 2026", "2026-06"),
    ("2025-06", "2025-06"),
    ("Sept 2025", "2025-09"),
    ("Jan. 2027", "2027-01"),
    ("2024 - 2026", "2026"),          # a range ends at its later year
    ("2026", "2026"),                 # year only: published as a year, never padded
    ("ongoing", None),                # not a date; show nothing rather than junk
    ("", None),
])
def test_only_a_real_date_is_published_as_a_graduation_date(raw, want):
    got = analysis_result({"candidate": {"education": [{"end_date": raw}]}})
    assert (got["graduationDate"] or {}).get("value") == want


# ---------------------------------------------------------------------------
# evidence, not prose
# ---------------------------------------------------------------------------
def test_a_job_with_no_evidence_chain_is_dropped_not_shipped():
    """The frontend's own rule: a recommendation the user cannot check is one the
    sceptical user will not trust."""
    gap = {"matched_jobs": [
        {"job_id": "a", "job_title": "No evidence", "gap_score": 0.2,
         "skill_resolution": [{"skill": "X", "verdict": "missing", "satisfied_by": None}]},
        {"job_id": "b", "job_title": "Evidenced", "gap_score": 0.3,
         "skill_resolution": [{"skill": "SQL", "verdict": "matched", "satisfied_by": "SQL"}]},
    ]}
    out = job_matches(gap)
    assert [j["id"] for j in out] == ["b"]
    assert "SQL covers their requirement for SQL" in out[0]["why"]


def test_a_job_score_is_the_complement_and_null_survives():
    gap = {"matched_jobs": [
        {"job_id": "a", "job_title": "T", "gap_score": None,
         "skill_resolution": [{"skill": "SQL", "verdict": "matched", "satisfied_by": "SQL"}]},
    ]}
    assert job_matches(gap)[0]["score"] is None


def test_skill_confidence_uses_the_real_grounding_score_when_there_is_one():
    profile = {"skills": {"accepted": [{"name": "SQL", "quality": "low"}]},
               "provenance": {"grounding": {"skills[0].name": {"score": 0.97}}}}
    assert analysis_result(profile)["skills"][0]["confidence"] == 0.97


def test_a_categorical_quality_never_lands_above_the_trust_threshold_dishonestly():
    """The UI states anything >= 0.85 as fact (TRUST_THRESHOLD). Only `high`
    reaches it, and only just."""
    for quality, expected in (("high", 0.86), ("medium", 0.70), ("low", 0.50)):
        profile = {"skills": {"accepted": [{"name": "S", "quality": quality}]}}
        assert analysis_result(profile)["skills"][0]["confidence"] == expected


def test_a_course_that_was_not_found_never_becomes_a_course():
    out = courses({"recommendations": [
        {"skill": "welding", "no_course_found": True, "course": None},
    ]})
    assert out == []
