"""Agent E offline: the whole graph driven by fakes injected through Deps.

No database, no network. A FakeReader returns hand-built CourseCandidates and a
FakePlainLLM echoes the exact fact sheet it was handed — so the grep test proves
the ONLY thing the model can see is free of internal tokens.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from agents.agent_e_course_recommend.graph import build_recommend_graph
from agents.agent_e_course_recommend.nodes import (
    Deps,
    _compute_buckets,
    _render_user_message,
    greedy_assign,
)
from shared.config import Config
from shared.contracts import CourseCandidate, CoursePrice


# ---------------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------------
def cc(cid, title="Course", *, rating=None, reviews=None, last_updated=None,
       price=None, provider="Prov", taught=None) -> CourseCandidate:
    return CourseCandidate(
        course_id=cid, title=title, provider=provider, url=f"https://x/{cid}",
        taught_skills=taught or [], rating=rating, review_count=reviews,
        last_updated=last_updated, price=price,
    )


class FakeReader:
    """Stand-in for shared.course_market.courses_for_skills."""

    def __init__(self, by_esco=None, by_key=None):
        self.by_esco = by_esco or {}
        self.by_key = by_key or {}
        self.calls = []

    def __call__(self, esco_codes, skill_keys, *, config=None):
        self.calls.append((tuple(esco_codes), tuple(skill_keys)))
        return {
            "by_esco": {c: self.by_esco.get(c, []) for c in esco_codes},
            "by_key": {k: self.by_key.get(k, []) for k in skill_keys},
        }


class FakePlainLLM:
    """Echoes the human fact sheet as the rationale, so a test can grep exactly
    what the model was allowed to see. Records every call."""

    def __init__(self, canned=None):
        self.calls = []                 # list of (system, human)
        self.canned = canned

    def invoke(self, messages):
        system, human = messages[0].content, messages[1].content
        self.calls.append((system, human))
        return SimpleNamespace(content=self.canned if self.canned is not None else human)


def detail(skill, esco, priority):
    return {"skill": skill, "esco_code": esco, "priority_score": priority}


def run_graph(tmp_path, missing_details, *, reader, llm=None, used_fallback=False, user_id="u1"):
    llm = llm or FakePlainLLM()
    tmp_path.mkdir(parents=True, exist_ok=True)
    gap = {
        "user_id": user_id,
        "used_fallback": used_fallback,
        "aggregate": {
            "missing_skill_details": missing_details,
            "most_common_missing_skills": [d["skill"] for d in missing_details],
        },
    }
    (tmp_path / "skill_gap.json").write_text(json.dumps(gap), encoding="utf-8")
    deps = Deps(config=Config(), llm=llm, courses_reader=reader)
    state = build_recommend_graph(deps).invoke({
        "gap_path": str(tmp_path / "skill_gap.json"),
        "output_dir": str(tmp_path), "run_id": "t",
    })
    out = json.loads((tmp_path / "t" / "course_recommendations.json").read_text(encoding="utf-8"))
    return state, out, llm


# ---------------------------------------------------------------------------
# coverage / no_course_found
# ---------------------------------------------------------------------------
def test_zero_candidate_skill_is_no_course_found_and_skips_the_llm(tmp_path):
    reader = FakeReader(by_esco={})           # nothing teaches the skill
    llm = FakePlainLLM()
    _, out, llm = run_graph(tmp_path, [detail("welding", "uri:W", 5.0)],
                            reader=reader, llm=llm)

    assert len(out["recommendations"]) == 1
    rec = out["recommendations"][0]
    assert rec["no_course_found"] is True and rec["course"] is None
    assert rec["skill"] == "welding" and rec["esco_code"] == "uri:W"
    assert llm.calls == [], "generate_rationale ran for a no_course_found skill"


def test_one_course_covering_three_skills_is_recommended_once(tmp_path):
    x = cc("x", "All-in-one", rating=4.5)
    reader = FakeReader(by_esco={"uri:a": [x], "uri:b": [x], "uri:c": [x]})
    _, out, _ = run_graph(tmp_path, [
        detail("skill-a", "uri:a", 3.0),
        detail("skill-b", "uri:b", 2.0),
        detail("skill-c", "uri:c", 1.0),
    ], reader=reader)

    withcourse = [r for r in out["recommendations"] if r["course"]]
    assert len(withcourse) == 1, "the shared course was recommended more than once"
    rec = withcourse[0]
    assert rec["course"]["course_id"] == "x"
    assert rec["skill"] == "skill-a"          # highest priority is the primary slot
    assert rec["course"]["covers_other_skills"] == ["skill-b", "skill-c"]
    # no separate entries for b and c
    assert all(r["skill"] != "skill-b" for r in out["recommendations"])


def test_selection_is_deterministic_across_runs(tmp_path):
    def build():
        p = tmp_path / "r"
        p.mkdir(exist_ok=True)
        x = cc("x", rating=4.0, reviews=10)
        y = cc("y", rating=4.0, reviews=99)    # same rating, more reviews -> wins tie
        reader = FakeReader(by_esco={"uri:a": [x, y]})
        _, out, _ = run_graph(p, [detail("skill-a", "uri:a", 1.0)], reader=reader)
        return out
    first, second = build(), build()
    assert first["recommendations"] == second["recommendations"]
    assert first["recommendations"][0]["course"]["course_id"] == "y"


# ---------------------------------------------------------------------------
# tie-break: nulls last, never coerced to zero
# ---------------------------------------------------------------------------
def test_a_rated_course_beats_an_unrated_one(tmp_path):
    rated = cc("b", rating=3.2)                # higher course_id, but has a rating
    unrated = cc("a", rating=None)            # lower course_id, but null rating
    reader = FakeReader(by_esco={"uri:a": [unrated, rated]})
    _, out, _ = run_graph(tmp_path, [detail("skill-a", "uri:a", 1.0)], reader=reader)
    assert out["recommendations"][0]["course"]["course_id"] == "b"


def test_zero_rating_is_a_real_value_and_beats_null(tmp_path):
    """Null must not be coerced to 0: a genuine 0.0 rating outranks a missing
    one even though it loses the course_id tiebreak."""
    zero = cc("z", rating=0.0)                 # real 0.0
    missing = cc("a", rating=None)             # null; lower id
    reader = FakeReader(by_esco={"uri:a": [missing, zero]})
    _, out, _ = run_graph(tmp_path, [detail("skill-a", "uri:a", 1.0)], reader=reader)
    assert out["recommendations"][0]["course"]["course_id"] == "z"


def test_two_null_courses_break_on_lowest_course_id(tmp_path):
    reader = FakeReader(by_esco={"uri:a": [cc("b"), cc("a")]})
    _, out, _ = run_graph(tmp_path, [detail("skill-a", "uri:a", 1.0)], reader=reader)
    assert out["recommendations"][0]["course"]["course_id"] == "a"


def test_free_course_wins_on_price_over_a_paid_one(tmp_path):
    free = cc("free", price=CoursePrice(amount=0.0, currency=None, is_free=True))
    paid = cc("paid", price=CoursePrice(amount=49.0, currency="USD", is_free=False))
    # identical on rating/reviews/last_updated (all null) -> price decides
    reader = FakeReader(by_esco={"uri:a": [paid, free]})
    _, out, _ = run_graph(tmp_path, [detail("skill-a", "uri:a", 1.0)], reader=reader)
    assert out["recommendations"][0]["course"]["course_id"] == "free"


def test_null_price_loses_to_any_real_price_and_does_not_crash(tmp_path):
    priced = cc("p", price=CoursePrice(amount=10.0, currency="USD", is_free=False))
    nulled = cc("a", price=None)               # lower id, but null price sorts last
    reader = FakeReader(by_esco={"uri:a": [nulled, priced]})
    _, out, _ = run_graph(tmp_path, [detail("skill-a", "uri:a", 1.0)], reader=reader)
    assert out["recommendations"][0]["course"]["course_id"] == "p"


# ---------------------------------------------------------------------------
# rationale fencing
# ---------------------------------------------------------------------------
def test_rationale_input_and_output_carry_no_internal_tokens(tmp_path):
    x = cc("x", "Data Science 101", rating=4.5, reviews=1200,
           price=CoursePrice(amount=0.0, currency=None, is_free=True))
    reader = FakeReader(by_esco={"uri:a": [x]})
    # a distinctive priority so we can prove the raw float never leaks
    _, out, llm = run_graph(tmp_path, [detail("python", "uri:a", 47.0)], reader=reader)

    rationale = out["recommendations"][0]["course"]["rationale"]
    _, human = llm.calls[0]
    for blob in (rationale, human):
        low = blob.lower()
        assert "esco" not in low
        assert "gap_score" not in low
        assert "priority_score" not in low
        assert "uri:a" not in low
        assert "47" not in blob, "the raw priority_score float leaked to the model"
    # priority reached the model only as a plain bucket word
    assert "Demand level for this skill:" in human


def test_no_llm_leaves_rationale_empty_without_warnings(tmp_path):
    """--no-rationale mode: selection still runs, rationales stay null, and it is
    NOT treated as a failure (no warnings)."""
    x = cc("x", rating=4.0)
    reader = FakeReader(by_esco={"uri:a": [x]})
    gap = {"used_fallback": False, "aggregate": {
        "missing_skill_details": [detail("skill-a", "uri:a", 1.0)]}}
    (tmp_path / "skill_gap.json").write_text(json.dumps(gap), encoding="utf-8")
    deps = Deps(config=Config(), llm=None, courses_reader=reader)
    state = build_recommend_graph(deps).invoke({
        "gap_path": str(tmp_path / "skill_gap.json"),
        "output_dir": str(tmp_path), "run_id": "t"})
    out = json.loads((tmp_path / "t" / "course_recommendations.json").read_text(encoding="utf-8"))
    assert out["recommendations"][0]["course"]["rationale"] is None
    assert not [w for w in state.get("warnings", []) if "rationale" in w]


def test_used_fallback_softens_the_recommendation_basis(tmp_path):
    x = cc("x", rating=4.0)
    reader = FakeReader(by_esco={"uri:a": [x]})

    _, _, llm_fb = run_graph(tmp_path / "fb", [detail("skill-a", "uri:a", 1.0)],
                             reader=reader, used_fallback=True)
    _, _, llm_dir = run_graph(tmp_path / "dir", [detail("skill-a", "uri:a", 1.0)],
                              reader=reader, used_fallback=False)

    assert "general demand data for your field" in llm_fb.calls[0][1]
    assert "specific job postings you matched with" in llm_dir.calls[0][1]


# ---------------------------------------------------------------------------
# pure helpers
# ---------------------------------------------------------------------------
def test_render_user_message_omits_nulls_and_marks_free():
    rec = {
        "skill": "python", "priority_bucket": "high",
        "course": {
            "title": "T", "provider": "P", "covers_other_skills": [],
            "quality": {"rating": None, "review_count": None,
                        "price": {"amount": 0.0, "currency": None, "is_free": True},
                        "last_updated": None},
        },
    }
    msg = _render_user_message(rec, used_fallback=False)
    assert "Rating: not available" in msg
    assert "Reviews: not available" in msg
    assert "Price: free" in msg
    assert "Last updated: not available" in msg
    assert "Also covers: nothing else — single-skill match" in msg


def test_priority_buckets_split_into_thirds():
    missing = [
        {"skill": "hi", "priority_score": 90.0},
        {"skill": "mid", "priority_score": 50.0},
        {"skill": "lo", "priority_score": 10.0},
    ]
    buckets = _compute_buckets(missing)
    assert buckets == {"hi": "high", "mid": "moderate", "lo": "some"}


def test_greedy_prefers_the_course_that_covers_more_weight():
    """A course covering two skills (summed priority 5) beats one covering a
    single higher-priority skill (4) — coverage value is count x weight."""
    a = cc("a")   # covers s1 (3) + s2 (2) = weight 5, 2 skills
    b = cc("b")   # covers s3 (4) only
    missing = [
        {"skill": "s1", "esco_code": "u1", "priority_score": 3.0},
        {"skill": "s2", "esco_code": "u2", "priority_score": 2.0},
        {"skill": "s3", "esco_code": "u3", "priority_score": 4.0},
    ]
    skill_candidates = {"s1": ["a"], "s2": ["a"], "s3": ["b"]}
    courses = {"a": a, "b": b}
    assigned, no_course = greedy_assign(missing, skill_candidates, courses,
                                        ("rating", "review_count", "last_updated", "price"))
    assert assigned == {"a": ["s1", "s2"], "b": ["s3"]}
    assert no_course == []
