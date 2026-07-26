"""Coursework-derived skills: the transcript's unclaimed skills, promoted.

Every test drives the node directly with a fake judge, since derivation reuses
the same skill judge to filter its candidates.
"""

from __future__ import annotations

from agents.agent_a_cv_extraction.nodes.derive_skills import (
    collect_derived_candidates,
    make_derive_coursework_skills_node,
)
from agents.agent_a_cv_extraction.schemas import SkillJudgement, SkillVerdict
from shared.config import Config
from tests.fake_llm import FakeStructuredLLM


# A CV that claims SQL (with a project) and nothing else; a transcript that also
# taught Apache Spark and a generic "databases".
CV = {"skills": [{"name": "SQL"}], "projects": [{"name": "ETL tool", "technologies": ["SQL"]}]}
CURRICULUM = [
    {
        "credential_name": "Big Data Systems", "credential_kind": "course",
        "grade_achieved": "A",
        "typical_skills": ["Apache Spark", "SQL", "databases"],
    },
]


def _state(**kw):
    base = {
        "cv_extraction": CV,
        "curriculum": CURRICULUM,
        "accepted_skills": [
            {"name": "SQL", "keep": True, "quality": "high", "evidence_type": "project"}
        ],
        "rejected_skills": [],
    }
    base.update(kw)
    return base


def _judge(*verdicts, config=None):
    llm = FakeStructuredLLM(SkillJudgement=SkillJudgement(verdicts=list(verdicts)))
    return make_derive_coursework_skills_node(llm, config or Config())


def test_unclaimed_coursework_skill_is_promoted_and_flagged():
    node = _judge(
        SkillVerdict(name="Apache Spark", keep=True, quality="medium",
                     category="tool", evidence_type="course", rationale="taught")
    )
    out = node(_state())
    derived = [s for s in out["accepted_skills"] if s.get("origin") == "coursework_derived"]
    assert len(derived) == 1
    spark = derived[0]
    assert spark["name"] == "Apache Spark"
    assert spark["quality"] == "medium"
    assert spark["evidence_type"] == "course"
    assert spark["corroborating_credential"] == "Big Data Systems"
    assert "coursework" in spark["rationale"].lower()
    assert out["warnings"], "adding derived skills should be surfaced"


def test_already_claimed_skill_is_not_duplicated():
    # The fake judge even tries to return SQL; it must never become a second entry.
    node = _judge(
        SkillVerdict(name="SQL", keep=True, quality="high", category="technical",
                     evidence_type="course", rationale="also taught"),
        SkillVerdict(name="Apache Spark", keep=True, quality="medium", category="tool",
                     evidence_type="course", rationale="taught"),
    )
    out = node(_state())
    sql_entries = [s for s in out["accepted_skills"] if s["name"].casefold() == "sql"]
    assert len(sql_entries) == 1
    assert sql_entries[0].get("origin") != "coursework_derived"   # the original claim, untouched


def test_generic_candidate_rejected_by_the_reused_judge_is_not_added():
    node = _judge(
        SkillVerdict(name="Apache Spark", keep=True, quality="medium", category="tool",
                     evidence_type="course", rationale="taught"),
        SkillVerdict(name="databases", keep=False, quality="low", category="generic",
                     evidence_type="course", rationale="too vague"),
    )
    out = node(_state())
    names = {s["name"].casefold() for s in out["accepted_skills"]}
    assert "databases" not in names
    assert "apache spark" in names


def test_derived_skill_is_never_high_even_if_the_judge_says_so():
    """Studied, not applied — coursework caps at medium, so it always ranks below
    a project- or certificate-backed claim."""
    node = _judge(
        SkillVerdict(name="Apache Spark", keep=True, quality="high", category="tool",
                     evidence_type="project", rationale="model over-rated it")
    )
    out = node(_state())
    spark = next(s for s in out["accepted_skills"] if s["name"] == "Apache Spark")
    assert spark["quality"] == "medium"
    # the original project-backed claim stays high — existing behaviour intact
    sql = next(s for s in out["accepted_skills"] if s["name"] == "SQL")
    assert sql["quality"] == "high"


def test_disabled_config_derives_nothing_and_leaves_accepted_untouched():
    node = _judge(
        SkillVerdict(name="Apache Spark", keep=True, quality="medium", category="tool",
                     evidence_type="course", rationale="taught"),
        config=Config(derive_coursework_skills=False),
    )
    out = node(_state())
    assert "accepted_skills" not in out          # node made no change to the skill set
    assert "disabled" in out["trace"][0]


def test_cap_is_respected():
    curriculum = [{
        "credential_name": "Survey Course", "credential_kind": "course",
        "typical_skills": ["Alpha", "Bravo", "Charlie", "Delta"],
    }]
    verdicts = [
        SkillVerdict(name=n, keep=True, quality="medium", category="tool",
                     evidence_type="course", rationale="taught")
        for n in ("Alpha", "Bravo", "Charlie", "Delta")
    ]
    node = _judge(*verdicts, config=Config(max_coursework_derived_skills=2))
    out = node(_state(curriculum=curriculum, accepted_skills=[], rejected_skills=[],
                      cv_extraction={"skills": []}))
    derived = [s for s in out["accepted_skills"] if s.get("origin") == "coursework_derived"]
    assert len(derived) == 2


def test_failure_is_fail_open():
    class RaisingLLM:
        def with_structured_output(self, schema, **kw):
            from langchain_core.runnables import RunnableLambda

            def boom(_):
                raise RuntimeError("judge exploded")
            return RunnableLambda(boom)

    node = make_derive_coursework_skills_node(RaisingLLM(), Config())
    out = node(_state())
    assert "accepted_skills" not in out          # claimed skills untouched
    assert out["warnings"] and "derivation failed" in out["warnings"][0]


def test_corroborating_credential_prefers_certification_then_graded_course():
    curriculum = [
        {"credential_name": "Intro Course", "credential_kind": "course",
         "typical_skills": ["Docker"]},
        {"credential_name": "Graded Course", "credential_kind": "course",
         "grade_achieved": "A", "typical_skills": ["Docker"]},
        {"credential_name": "Cloud Cert", "credential_kind": "certification",
         "typical_skills": ["Docker"]},
    ]
    cands = collect_derived_candidates(curriculum, known=set())
    assert cands["docker"]["corroborating_credential"] == "Cloud Cert"
