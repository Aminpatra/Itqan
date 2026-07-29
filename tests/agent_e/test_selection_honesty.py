"""What a recommendation is allowed to claim, and how it must admit what it is.

Two things Agent E published on trust before this suite existed:

* **Which course.** On the live corpus `communication skills` had 13 candidates
  and not one carried a rating, a price or a date, so the entire tie-break chain
  fell through to the final `course_id` element — a SHA-256 hash. The output
  presented that as a recommendation.
* **Why.** The rationale was the only model output in the whole pipeline
  published without verification, against the repo's own stated thesis that
  prompt instructions alone do not prevent hallucination.
"""

from __future__ import annotations

import json

from agents.agent_e_course_recommend.nodes import (
    Deps,
    deterministic_rationale,
    greedy_assign,
    shrunk_rating,
    verify_claims,
    verify_rationale,
)
from shared.config import Config
from shared.contracts import CourseCandidate, CoursePrice
from tests.agent_e.test_recommend import FakePlainLLM, FakeReader, cc, detail, run_graph

TIEBREAK = tuple(Config().agent_e_tiebreak)


# ---------------------------------------------------------------------------
# a rating you can trust vs one you cannot
# ---------------------------------------------------------------------------
def test_a_well_reviewed_49_beats_a_thinly_reviewed_50():
    """THE live inversion. Raw `rating -> review_count` is lexicographic, so
    review_count only broke an EXACT rating tie and a 5.0 from 10 reviews
    outranked a 4.9 from 30,000. Every top-rated row on this corpus is a 5.0 with
    10-14 reviews, so this was not hypothetical."""
    thin = cc("thin", rating=5.0, reviews=10)
    solid = cc("solid", rating=4.9, reviews=30000)
    prior_mean = 4.5

    assert shrunk_rating(thin, prior_mean, 50) < shrunk_rating(solid, prior_mean, 50)


def test_a_rating_with_no_review_count_is_all_prior():
    """A rating nobody has voted on carries no evidence, so it sits at the corpus
    mean rather than being trusted at face value."""
    unbacked = cc("u", rating=5.0, reviews=None)
    assert shrunk_rating(unbacked, 4.0, 50) == 4.0


def test_an_unrated_course_still_has_no_shrunk_rating():
    """None means 'no rating', and must not become the corpus mean — that would
    invent a score for a course nobody rated."""
    assert shrunk_rating(cc("x"), 4.5, 50) is None


# ---------------------------------------------------------------------------
# saying so when nothing decided it
# ---------------------------------------------------------------------------
def test_a_pick_with_nothing_to_choose_between_candidates_is_marked_arbitrary():
    missing = [{"skill": "s", "esco_code": "u", "priority_score": 1.0}]
    bare = {c: cc(c) for c in ("a", "b", "c")}        # no rating/price/date at all
    _assigned, _none, basis = greedy_assign(
        missing, {"s": list(bare)}, bare, TIEBREAK)

    won = next(iter(basis.values()))
    assert won["basis"] == "arbitrary"
    assert won["equivalent_candidates"] == 3


def test_a_pick_decided_by_a_real_signal_is_marked_quality():
    missing = [{"skill": "s", "esco_code": "u", "priority_score": 1.0}]
    courses = {"a": cc("a"), "b": cc("b", rating=4.5, reviews=900)}
    assigned, _none, basis = greedy_assign(
        missing, {"s": ["a", "b"]}, courses, TIEBREAK)

    assert "b" in assigned, "the rated course did not win"
    assert basis["b"]["basis"] == "quality"


def test_the_envelope_counts_how_many_picks_were_arbitrary(tmp_path):
    reader = FakeReader(by_esco={"uri:a": [cc("a"), cc("b")]})
    _, out, _ = run_graph(tmp_path, [detail("skill-a", "uri:a", 1.0)], reader=reader)

    cal = out["calibration"]
    assert cal["recommendations_by_arbitrary_pick"] == ["skill-a"]
    assert cal["tiebreak"] == list(TIEBREAK)
    assert cal["candidates_per_skill"] == {"skill-a": 2}


def test_enrollment_breaks_a_tie_that_nothing_else_could():
    """Agent D collects enrollment_count for 252 courses; Agent E read it into
    CourseCandidate and then used it nowhere."""
    quiet = CourseCandidate(course_id="a", title="A", provider="P", url="u",
                            taught_skills=[], enrollment_count=12)
    popular = CourseCandidate(course_id="b", title="B", provider="P", url="u",
                              taught_skills=[], enrollment_count=90000)
    missing = [{"skill": "s", "esco_code": "u", "priority_score": 1.0}]
    assigned, _none, basis = greedy_assign(
        missing, {"s": ["a", "b"]}, {"a": quiet, "b": popular}, TIEBREAK)

    assert "b" in assigned
    assert basis["b"]["basis"] == "quality"


# ---------------------------------------------------------------------------
# the rationale must not assert what it was not told
# ---------------------------------------------------------------------------
FACTS = "Skill: python\nCourse: Data Science 101 (Acme)\nRating: 4.5\nReviews: 200"


def test_an_invented_figure_voids_the_rationale():
    """The fabrication that reads as most authoritative is a specific number."""
    reason = verify_rationale(
        "Data Science 101 covers python across 40 hours of video.", FACTS)
    assert reason and "40" in reason


def test_restating_the_given_numbers_is_fine():
    assert verify_rationale(
        "Data Science 101 is rated 4.5 from 200 reviews, and covers python.",
        FACTS) is None


def test_comma_and_trailing_zero_formatting_is_not_an_invention():
    facts = "Reviews: 30000\nRating: 4.50"
    assert verify_rationale("Rated 4.5 from 30,000 reviews.", facts) is None


def test_an_internal_term_voids_the_rationale():
    assert verify_rationale("This closes an ESCO-coded gap.", FACTS)


def test_an_empty_rationale_is_not_silently_published():
    assert verify_rationale("   ", FACTS) == "empty"


def test_a_hallucinating_model_falls_back_to_the_template(tmp_path):
    llm = FakePlainLLM(canned="A superb 12-week bootcamp with 5 expert mentors.")
    _, out, _ = run_graph(tmp_path, [detail("skill-a", "uri:a", 1.0)],
                          reader=FakeReader(by_esco={"uri:a": [cc("x", "Real Course",
                                                                 rating=4.0)]}),
                          llm=llm)
    course = out["recommendations"][0]["course"]
    assert course["rationale_source"] == "template"
    assert "bootcamp" not in course["rationale"]
    assert "Real Course" in course["rationale"]


def test_a_clean_model_rationale_is_kept(tmp_path):
    llm = FakePlainLLM(canned="Real Course covers skill-a and is a sensible place to start.")
    _, out, _ = run_graph(tmp_path, [detail("skill-a", "uri:a", 1.0)],
                          reader=FakeReader(by_esco={"uri:a": [cc("x", "Real Course")]}),
                          llm=llm)
    assert out["recommendations"][0]["course"]["rationale_source"] == "model"


def test_the_template_omits_facts_it_does_not_have():
    """Same rule the prompt gives the model, enforced instead of requested: a
    null is left out, never rendered as a guess."""
    rec = {
        "skill": "python", "priority_bucket": "high",
        "supply": {"courses_available": 9, "thin": False},
        "selection": {"basis": "quality"},
        "demand": {},
        "course": {"title": "T", "provider": None, "covers_other_skills": [],
                   "quality": {"rating": None, "review_count": None,
                               "enrollment_count": None, "price": None}},
    }
    text = deterministic_rationale(rec, used_fallback=False)
    assert "T covers python." in text
    for absent in ("None", "not available", "rated", "free"):
        assert absent not in text


def test_the_template_says_when_the_pick_was_arbitrary():
    rec = {
        "skill": "python", "priority_bucket": "high",
        "supply": {"courses_available": 13, "thin": False},
        "selection": {"basis": "arbitrary", "equivalent_candidates": 13},
        "demand": {},
        "course": {"title": "T", "provider": None, "covers_other_skills": [],
                   "quality": {"rating": None, "review_count": None,
                               "enrollment_count": None, "price": None}},
    }
    text = deterministic_rationale(rec, used_fallback=False)
    assert "13 courses matched" in text and "representative" in text


# ---------------------------------------------------------------------------
# the bucket is a rank, not a market measurement
# ---------------------------------------------------------------------------
def test_a_single_gap_is_not_automatically_your_biggest(tmp_path):
    """With one missing skill `lo == hi`, every threshold collapsed and the sole
    gap was ALWAYS labelled 'high' — a ranking of one thing against nothing."""
    from agents.agent_e_course_recommend.nodes import _compute_buckets

    assert _compute_buckets([{"skill": "only", "priority_score": 0.2}]) == {"only": "moderate"}


def test_gaps_that_all_weigh_the_same_are_not_all_high():
    from agents.agent_e_course_recommend.nodes import _compute_buckets

    same = [{"skill": s, "priority_score": 1.0} for s in ("a", "b", "c")]
    assert set(_compute_buckets(same).values()) == {"moderate"}


def test_agent_c_market_evidence_reaches_the_output_and_the_model(tmp_path):
    """`jobs_missing_in` is Agent C's real, checkable market measurement, and
    Agent E dropped it on the floor while asking the model to imply it from a
    relative rank instead."""
    gap = {
        "user_id": "u", "used_fallback": False,
        "aggregate": {"missing_skill_details": [
            {"skill": "python", "esco_code": "uri:a", "priority_score": 2.0,
             "jobs_missing_in": 7, "low_confidence": False},
        ]},
    }
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "skill_gap.json").write_text(json.dumps(gap), encoding="utf-8")

    llm = FakePlainLLM()
    from agents.agent_e_course_recommend.graph import build_recommend_graph
    deps = Deps(config=Config(), llm=llm,
                courses_reader=FakeReader(by_esco={"uri:a": [cc("x", "Course X")]}))
    build_recommend_graph(deps).invoke({
        "gap_path": str(tmp_path / "skill_gap.json"),
        "output_dir": str(tmp_path), "run_id": "t"})
    out = json.loads((tmp_path / "t" / "course_recommendations.json").read_text(encoding="utf-8"))

    assert out["recommendations"][0]["demand"]["jobs_missing_in"] == 7
    _system, human = llm.calls[0]
    assert "7 of the roles you matched asked for it" in human


# ---------------------------------------------------------------------------
# claims the record can settle — found by running the REAL model, not by guessing
# ---------------------------------------------------------------------------
def _rec(*, available=20, thin=False, basis="quality", equivalent=1):
    return {
        "skill": "python", "priority_bucket": "high",
        "supply": {"courses_available": available, "thin": thin},
        "selection": {"basis": basis, "equivalent_candidates": equivalent},
        "demand": {},
        "course": {"title": "T", "provider": None, "covers_other_skills": [],
                   "quality": {"rating": None, "review_count": None,
                               "enrollment_count": None, "price": None}},
    }


def test_calling_a_healthy_field_scarce_is_rejected():
    """A numbers-only check passed this, and on the first real-model run SIX of
    eight rationales called a field of 9-38 courses "one of a small number of
    options". Not a fabricated figure — a false statement to someone choosing
    what to study."""
    reason = verify_claims(
        "This is one of a small number of options for python.", _rec(available=38), False)
    assert reason and "38" in reason


def test_calling_a_genuinely_thin_field_scarce_is_fine():
    assert verify_claims(
        "Few courses cover python, so options are limited.",
        _rec(available=1, thin=True), False) is None


def test_the_general_demand_hedge_is_rejected_when_the_run_matched_real_postings():
    """Three rationales offered the aggregate-stats hedge on a run where
    used_fallback was False — understating real evidence, straight out of a
    prompt rule whose condition did not hold."""
    assert verify_claims("A solid option based on general demand in your field.",
                         _rec(), used_fallback=False)
    assert verify_claims("A solid option based on general demand in your field.",
                         _rec(), used_fallback=True) is None


def test_an_arbitrary_pick_must_admit_it():
    """The one thing the message must never do is present a hash-order pick as a
    ranked best — which is exactly what the model wrote before this check."""
    arbitrary = _rec(basis="arbitrary", equivalent=13)
    assert verify_claims("This course aligns with the demand in roles you matched.",
                         arbitrary, False)
    assert verify_claims(
        "13 courses matched equally well, so this is a representative pick.",
        arbitrary, False) is None


def test_vague_praise_still_passes_and_that_limit_is_deliberate():
    """Only claims the record can adjudicate are checked. "A solid course" has no
    deterministic answer, and pretending to verify it would be theatre."""
    assert verify_claims("A solid, well-structured course.", _rec(), False) is None
