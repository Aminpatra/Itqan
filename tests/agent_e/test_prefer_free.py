"""The candidate asked for free courses. What that is allowed to do.

Two independent halves, and the second is the one that earns its keep:

* **Selection.** Free-ness moves to the front of the tie-break. It reorders; it
  never removes. Measured on the live corpus, 0 of 1,999 Coursera courses publish
  a price and only freeCodeCamp's 98 are flagged free, so a filter would cut the
  catalogue by 95% and turn most gaps into `no_course_found` — the user chose
  ranking over that (2026-07-30).
* **The claim.** The fact sheet now tells the model the learner wants free
  courses, which is exactly the setup that produces a friendly fabrication:
  "and it's free, as you asked" about a course whose price nobody knows. The
  earlier audit measured this failure class — 6 of 8 rationales called a normal
  field scarce while passing a numbers-only check — so telling the model about a
  preference without extending the verifier would reintroduce it.
"""

from __future__ import annotations

from agents.agent_e_course_recommend.nodes import (
    _asked_for_line,
    _field_key,
    effective_tiebreak,
    greedy_assign,
    verify_claims,
)
from shared.config import Config
from shared.contracts import CourseCandidate, CoursePrice

TIEBREAK = tuple(Config().agent_e_tiebreak)
FREE = CoursePrice(amount=0.0, currency=None, is_free=True)
PAID = CoursePrice(amount=49.0, currency="USD", is_free=False)


def _course(cid, *, price=None, rating=None, reviews=None):
    return CourseCandidate(course_id=cid, title=cid.upper(), provider="P", url="u",
                           taught_skills=[], price=price, rating=rating,
                           review_count=reviews)


# ---------------------------------------------------------------------------
# the tie-break chain
# ---------------------------------------------------------------------------
def test_without_the_preference_the_chain_is_untouched():
    assert effective_tiebreak(TIEBREAK, prefer_free=False) == TIEBREAK


def test_free_is_prepended_not_substituted():
    """Prepending is the design: free decides FIRST, and every configured signal
    still decides among the courses that tie on it. Replacing the chain would
    throw away the ranking work rather than reordering it."""
    chain = effective_tiebreak(TIEBREAK, prefer_free=True)
    assert chain[0] == "price_is_free"
    assert chain[1:] == TIEBREAK


def test_an_unknown_price_sorts_between_free_and_paid():
    """The three-valued middle is the whole point. "Not free" and "we never
    looked" are different claims, and coercing the second into the first would
    bury most of the catalogue on a fact nobody established — the same
    null-is-not-zero rule the price column itself follows."""
    free = _field_key(_course("a", price=FREE), "price_is_free")
    unknown = _field_key(_course("b"), "price_is_free")
    paid = _field_key(_course("c", price=PAID), "price_is_free")
    assert free < unknown < paid


def test_a_free_course_wins_over_a_better_rated_paid_one():
    """The behavioural claim. Without the preference the 4.9 wins on rating; with
    it, the free 4.2 does — and that reversal is the answer being honoured."""
    good_paid = _course("paid", price=PAID, rating=4.9, reviews=5000)
    plain_free = _course("free", price=FREE, rating=4.2, reviews=5000)
    missing = [{"skill": "s", "esco_code": "u", "priority_score": 1.0}]
    courses = {"paid": good_paid, "free": plain_free}

    without, _n, _b = greedy_assign(missing, {"s": ["paid", "free"]}, courses, TIEBREAK)
    with_pref, _n, _b = greedy_assign(missing, {"s": ["paid", "free"]}, courses,
                                      effective_tiebreak(TIEBREAK, prefer_free=True))
    assert "paid" in without
    assert "free" in with_pref


def test_a_paid_course_is_still_recommended_when_nothing_free_covers_the_skill():
    """RANK, NEVER EXCLUDE. A gap whose only course costs money still needs an
    answer; reporting `no_course_found` because of a pricing preference would be
    obeying the answer by making the product useless."""
    missing = [{"skill": "s", "esco_code": "u", "priority_score": 1.0}]
    assigned, no_course, _b = greedy_assign(
        missing, {"s": ["paid"]}, {"paid": _course("paid", price=PAID)},
        effective_tiebreak(TIEBREAK, prefer_free=True))
    assert "paid" in assigned and no_course == []


# ---------------------------------------------------------------------------
# what the model is told, and what it may then say
# ---------------------------------------------------------------------------
def _rec(price):
    return {"skill": "python",
            "course": {"title": "Data Science 101", "provider": "Acme",
                       "covers_other_skills": [],
                       "quality": {"rating": None, "review_count": None,
                                   "price": price}},
            "supply": {"courses_available": 20, "thin": False},
            "selection": {"basis": "quality", "equivalent_candidates": 1},
            "demand": {}}


def test_the_fact_sheet_states_whether_this_course_actually_qualifies():
    """Telling a model "the learner wants free courses" and nothing else is an
    invitation to congratulate itself."""
    assert "may or may not" in _asked_for_line(_rec(None), True)
    assert "IS free" in _asked_for_line(_rec({"is_free": True, "amount": 0.0}), True)
    assert "NOT free" in _asked_for_line(_rec({"is_free": False, "amount": 49.0}), True)
    assert _asked_for_line(_rec(None), False) == "nothing stated"


def test_calling_an_unpriced_course_free_is_rejected():
    """The guard that makes the prompt change safe. An unpublished price is not
    evidence of a zero one, and this is the single most actionable claim in the
    whole rationale."""
    reason = verify_claims("Data Science 101 covers python and it is free.",
                          _rec(None), False)
    assert reason and "no price is published" in reason


def test_calling_a_paid_course_free_is_rejected_with_the_amount():
    reason = verify_claims("It is free of charge.",
                          _rec({"is_free": False, "amount": 49.0}), False)
    assert reason and "49.0" in reason


def test_a_genuinely_free_course_may_be_called_free():
    assert verify_claims("It is free.", _rec({"is_free": True, "amount": 0.0}),
                         False) is None


def test_a_course_whose_own_title_says_free_does_not_fail_itself():
    """`Free Fall Physics` is a legitimate course name. Excising the title before
    the check is what stops a course being rejected for what it is called."""
    rec = _rec(None)
    rec["course"]["title"] = "Free Fall Physics"
    assert verify_claims("Free Fall Physics covers python well.", rec, False) is None


def test_the_provider_freecodecamp_is_not_a_price_claim():
    """`\\bfree\\b` does not match inside "freeCodeCamp" — no word boundary between
    "free" and "C" — and freeCodeCamp is a real provider on this corpus, named in
    every fact sheet its courses appear in."""
    assert verify_claims("A freeCodeCamp curriculum covering python.",
                         _rec(None), False) is None


# ---------------------------------------------------------------------------
# end to end through the real graph
# ---------------------------------------------------------------------------
def test_the_preference_reaches_the_graph_and_the_envelope(tmp_path):
    """Both halves of the wiring at once: the chain the run ACTUALLY used is what
    gets published, so a ranking stays reproducible from its own output rather
    than from the configured default it diverged from."""
    from tests.agent_e.test_recommend import FakeReader, detail, run_graph

    free = _course("free", price=FREE, rating=4.2, reviews=500)
    paid = _course("paid", price=PAID, rating=4.9, reviews=500)
    reader = FakeReader(by_esco={"u": [free, paid]})
    details = [detail("python", "u", 5.0)]

    _s, plain, _l = run_graph(tmp_path / "a", details, reader=reader)
    _s, biased, _l = run_graph(tmp_path / "b", details, reader=reader, prefer_free=True)

    assert plain["calibration"]["prefer_free"] is False
    assert plain["calibration"]["tiebreak"] == list(TIEBREAK)
    assert plain["recommendations"][0]["course"]["course_id"] == "paid"

    assert biased["calibration"]["prefer_free"] is True
    assert biased["calibration"]["tiebreak"][0] == "price_is_free"
    assert biased["recommendations"][0]["course"]["course_id"] == "free"
