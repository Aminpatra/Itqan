"""The fenced skill-matching tier: what it fixes, and what it cannot do.

The tier exists because cosine similarity is symmetric and skill matching is not:
a candidate holding TensorFlow, PyTorch and scikit-learn was told a Senior Machine
Learning Engineer role was a 100% gap, because "machine learning" scored 0.5982
against their nearest skill.

Since this is the only place a model can move a published number, most of these
tests are about the fence rather than the feature — a model that invents, guesses
or over-reaches must not be able to change anything.
"""

from __future__ import annotations

from typing import Any

from agents.agent_c_gap_analysis.schemas import SkillMatchBatch, SkillVerdict
from agents.agent_c_gap_analysis.skill_resolver import resolve_skills
from shared.config import Config

CANDIDATE = ["Python", "TensorFlow", "PyTorch", "scikit-learn", "SQL", "JavaScript"]
RECORDS = [
    {"name": "Python", "quality": "high", "evidence_type": "project"},
    {"name": "TensorFlow", "quality": "medium", "evidence_type": "course"},
    {"name": "PyTorch", "quality": "medium", "evidence_type": "course"},
    {"name": "scikit-learn", "quality": "medium", "evidence_type": "course"},
    {"name": "SQL", "quality": "medium", "evidence_type": "course"},
    {"name": "JavaScript", "quality": "medium", "evidence_type": "course"},
    {"name": "HDFS", "quality": "low", "evidence_type": "claim_only"},
]


class FakeLLM:
    """Returns canned verdicts; records that it was called."""

    def __init__(self, verdicts: list[SkillVerdict]):
        self.verdicts, self.calls = verdicts, 0

    def with_structured_output(self, schema: type, **kw: Any):
        from langchain_core.runnables import RunnableLambda

        def _invoke(_payload: Any):
            self.calls += 1
            return SkillMatchBatch(verdicts=self.verdicts)

        return RunnableLambda(_invoke)


def _resolve(verdicts, *, unresolved=("machine learning",), config=None, candidate=None):
    llm = FakeLLM(list(verdicts))
    out = resolve_skills(
        unresolved=[{"skill": s, "key": s.lower()} for s in unresolved],
        candidate_skills=list(candidate if candidate is not None else CANDIDATE),
        skill_records=RECORDS,
        config=config or Config(),
        llm=llm,
    )
    return out, llm


# ---------------------------------------------------------------------------
# what it fixes
# ---------------------------------------------------------------------------
def test_a_component_skill_satisfies_the_umbrella_requirement():
    """THE case this tier exists for: cosine put 'machine learning' at 0.5982 —
    below the 0.60 floor — so the ML role scored a 100% gap."""
    out, _ = _resolve([
        SkillVerdict(requirement="machine learning", decision="satisfied",
                     satisfied_by="TensorFlow", reason="TensorFlow is an ML framework")
    ])
    assert out["machine learning"]["verdict"] == "matched"
    assert out["machine learning"]["resolved_by"] == "llm"
    assert out["machine learning"]["satisfied_by"] == "TensorFlow"


# ---------------------------------------------------------------------------
# the fence
# ---------------------------------------------------------------------------
def test_a_hallucinated_candidate_skill_voids_the_verdict():
    """The model cites a skill the candidate does not have. The whole verdict is
    discarded and the deterministic answer stands — the same discipline as Agent
    A's grounded spans and Agent B's verified quotes."""
    out, _ = _resolve([
        SkillVerdict(requirement="machine learning", decision="satisfied",
                     satisfied_by="Kubernetes", reason="invented")
    ])
    assert out == {}


def test_a_weak_evidence_skill_cannot_close_a_gap():
    """HDFS is claim_only — listed, never demonstrated. Even if the model accepts
    it, an unverified claim must not cancel a requirement."""
    out, _ = _resolve(
        [SkillVerdict(requirement="hadoop", decision="satisfied",
                      satisfied_by="HDFS", reason="HDFS is part of Hadoop")],
        unresolved=("hadoop",),
        candidate=CANDIDATE + ["HDFS"],
    )
    assert out == {}


def test_uncertain_keeps_the_deterministic_verdict():
    out, _ = _resolve([
        SkillVerdict(requirement="machine learning", decision="uncertain",
                     satisfied_by=None, reason="cannot tell from names")
    ])
    assert out == {}


def test_not_satisfied_cannot_manufacture_a_gap():
    """The deterministic tiers already decided missing-or-possible. Letting the
    model DOWNGRADE possible_match to missing would let it invent gaps, so a
    'not_satisfied' is simply no new information."""
    out, _ = _resolve([
        SkillVerdict(requirement="machine learning", decision="not_satisfied",
                     satisfied_by=None, reason="no")
    ])
    assert out == {}


def test_a_requirement_nobody_asked_about_is_ignored():
    out, _ = _resolve([
        SkillVerdict(requirement="underwater basket weaving", decision="satisfied",
                     satisfied_by="Python", reason="invented requirement")
    ])
    assert out == {}


def test_an_llm_failure_leaves_the_deterministic_verdicts_intact():
    class Exploding:
        def with_structured_output(self, schema, **kw):
            from langchain_core.runnables import RunnableLambda

            def _boom(_):
                raise RuntimeError("model unavailable")
            return RunnableLambda(_boom)

    out = resolve_skills(
        unresolved=[{"skill": "machine learning", "key": "machine learning"}],
        candidate_skills=CANDIDATE, skill_records=RECORDS,
        config=Config(), llm=Exploding(),
    )
    assert out == {}


# ---------------------------------------------------------------------------
# it stays optional
# ---------------------------------------------------------------------------
def test_disabled_config_makes_no_call():
    out, llm = _resolve(
        [SkillVerdict(requirement="machine learning", decision="satisfied",
                      satisfied_by="TensorFlow", reason="x")],
        config=Config(agent_c_llm_matching=False),
    )
    assert out == {} and llm.calls == 0


def test_no_llm_injected_is_a_no_op():
    assert resolve_skills(
        unresolved=[{"skill": "machine learning", "key": "machine learning"}],
        candidate_skills=CANDIDATE, skill_records=RECORDS, config=Config(), llm=None,
    ) == {}


def test_one_batched_call_for_many_requirements():
    """Per-skill calls would be ~40 per run; the candidate list is fixed, so one
    call answers all of them."""
    reqs = ("machine learning", "java", "mysql", "software development")
    _out, llm = _resolve(
        [SkillVerdict(requirement=r, decision="uncertain", satisfied_by=None, reason="x")
         for r in reqs],
        unresolved=reqs,
    )
    assert llm.calls == 1


# ---------------------------------------------------------------------------
# the prompt carries its anti-hallucination guards
# ---------------------------------------------------------------------------
def test_the_prompt_shows_both_directions_and_states_the_rules():
    from agents.agent_c_gap_analysis.prompts import SKILL_MATCH_PROMPT

    rendered = SKILL_MATCH_PROMPT.format(candidate_skills="Python", requirements="java")
    # few-shot examples covering the boundary that no rule can state
    assert "machine learning" in rendered and "JavaScript" in rendered
    assert "MySQL" in rendered            # general does not imply the specific
    # the standing guards
    assert "Never invent" in rendered
    assert "uncertain" in rendered
    assert "weak evidence" in rendered
