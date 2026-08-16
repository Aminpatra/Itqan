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


def _course(**quality):
    return {"recommendations": [{
        "skill": "power bi", "no_course_found": False,
        "course": {"course_id": "c", "title": "T", "provider": "Rutgers",
                   "source": quality.pop("source", None), "url": "u",
                   "covers_other_skills": [], "quality": quality},
    }]}


# ---------------------------------------------------------------------------
# what can be SAID about a price we never saw
# ---------------------------------------------------------------------------
def test_an_unpriced_coursera_course_is_labelled_paid():
    """A card showing nothing where the price belongs is useless to someone
    choosing what to study, and this is the normal case rather than an edge one:
    measured, 0 of 1,999 Coursera courses publish a price anywhere.

    The label is a claim about the PLATFORM's catalogue, which is true, and the
    amount stays null because no amount was ever observed."""
    out = courses(_course(price=None, source="coursera"))[0]
    assert out["priceLabel"] == "paid"
    assert out["price"] is None, "labelling it must not invent an amount"
    assert out["currency"] is None


def test_a_free_course_is_labelled_free_not_paid():
    out = courses(_course(price={"amount": 0.0, "currency": None, "is_free": True},
                          source="freecodecamp"))[0]
    assert out["priceLabel"] == "free" and out["price"] == 0.0


def test_an_unpriced_course_on_a_free_platform_is_not_called_paid():
    """The label is per-platform, and freeCodeCamp's catalogue is not sold. An
    unlabelled unknown is honest; 'Paid' there would be a plain falsehood."""
    assert courses(_course(price=None, source="freecodecamp"))[0]["priceLabel"] is None


def test_an_unknown_platform_gets_no_label():
    """A source this mapping has not been told about must not inherit Coursera's
    commercial model — exactly the assumption that ages badly as sources are
    added."""
    assert courses(_course(price=None, source="edraak"))[0]["priceLabel"] is None
    assert courses(_course(price=None))[0]["priceLabel"] is None


def test_a_real_amount_needs_no_label():
    """With a number to show the label would be redundant — and it must not
    override the amount."""
    out = courses(_course(price={"amount": 18.0, "currency": "OMR", "is_free": False},
                          source="coursera"))[0]
    assert out["priceLabel"] is None
    assert out["price"] == 18.0 and out["currency"] == "OMR"


def test_an_unstated_duration_is_null_across_all_three_fields():
    """This test used to assert that Agent D stored NO duration, which was true
    and was the bug: `workload` was fetched and discarded, the API published
    `hours: null`, and the card rendered "0 hours" because the front-end type
    said a number was always there.

    Agent D stores it now — but "nothing stated" must still be null in every
    field, because that is what the card renders as silence.
    """
    out = courses(_course())[0]

    assert out["hoursMin"] is None
    assert out["hoursMax"] is None
    assert out["durationText"] is None


def test_a_range_is_published_as_a_range():
    """'4 weeks of study, 2-4 hours a week' is 8 to 16. Publishing 12 would be a
    figure no provider ever stated — the reason durations were left unstored
    rather than guessed at."""
    out = courses(_course(hours_min=8.0, hours_max=16.0,
                          duration_text="4 weeks of study, 2-4 hours a week"))[0]

    assert (out["hoursMin"], out["hoursMax"]) == (8.0, 16.0)
    assert out["durationText"] == "4 weeks of study, 2-4 hours a week"


def test_one_stated_figure_has_equal_ends():
    """Not an average of anything: the provider said '2 heures'."""
    out = courses(_course(hours_min=2.0, hours_max=2.0,
                          duration_text="2 heures"))[0]

    assert out["hoursMin"] == out["hoursMax"] == 2.0


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


_EVIDENCE = [{"skill": "SQL", "verdict": "matched", "satisfied_by": "SQL"}]


def _one_job(**fields):
    gap = {"matched_jobs": [dict(job_id="a", job_title="T", gap_score=0.2,
                                 skill_resolution=_EVIDENCE, **fields)]}
    return job_matches(gap)[0]


def test_the_apply_link_is_the_employer_page_when_we_resolved_one():
    """The whole point of recording a destination. Sending someone to the
    aggregator's article when we know the employer's own vacancy page is the
    difference between a lead and an application."""
    job = _one_job(source_url="https://oman.el7far.com/2026/07/eni.html",
                   final_url="https://jobs.eni.com/en/sites/CX_1004/job/33730")
    assert job["source"]["url"] == "https://jobs.eni.com/en/sites/CX_1004/job/33730"


def test_a_posting_with_no_destination_still_links_somewhere():
    """`final_url` is NULL on every row ingested before migration 0011, so this
    is the majority case today, not an edge case. A blank apply link would be a
    regression on the whole existing corpus."""
    job = _one_job(source_url="https://oman.el7far.com/2026/07/eni.html")
    assert job["source"]["url"] == "https://oman.el7far.com/2026/07/eni.html"


@pytest.mark.parametrize("fields,want", [
    ({"work_arrangement": "remote"}, "Remote"),
    ({"work_arrangement": "remote", "employment_type": "internship"},
     "Remote · Internship"),
    ({"employment_type": "full_time"}, "Full time"),
    # The fallback: what every posting written before the destination crawl has.
    ({"seniority_level": "Senior"}, "Senior"),
    # A stated arrangement beats the fallback — that is the upgrade.
    ({"work_arrangement": "hybrid", "seniority_level": "Senior"}, "Hybrid"),
    # Nothing stated, nothing shown. Never a guessed "On-site".
    ({}, ""),
])
def test_the_arrangement_chip_says_only_what_the_posting_stated(fields, want):
    assert _one_job(**fields)["arrangement"] == want


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
