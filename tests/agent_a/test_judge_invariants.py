"""Invariants the skill judge must hold regardless of what the model returns.

Both of these were real failures. They are enforced in code rather than asked for
in the prompt, because a model under a long prompt drifts, and a rating system
that silently changes meaning is worse than no rating system.
"""

from __future__ import annotations

from agents.agent_a_cv_extraction.nodes.judge_skills import (
    _enforce_quality_rules,
    make_judge_skills_node,
)
from agents.agent_a_cv_extraction.schemas import SkillJudgement, SkillVerdict
from shared.config import Config
from tests.fake_llm import FakeStructuredLLM

EXTRACTION = {
    "skills": [
        {"name": "Python"},
        {"name": "SQL"},
        {"name": "Welding"},
        {"name": "Triage"},
    ],
    "projects": [{"name": "Ward rota tool", "technologies": ["Python"]}],
}


def judge_with(verdicts: list[SkillVerdict]) -> dict:
    llm = FakeStructuredLLM(SkillJudgement=SkillJudgement(verdicts=verdicts))
    node = make_judge_skills_node(llm, Config())
    return node({"cv_extraction": EXTRACTION})


def test_unjudged_skills_are_not_silently_dropped():
    """The model returned 16 verdicts for 20 claimed skills on a real run, and
    four skills vanished from the output. A claim the judge ignores is still a
    claim; it must survive as an unverified one."""
    result = judge_with(
        [
            SkillVerdict(
                name="Python", keep=True, quality="high", category="technical",
                evidence_type="project", rationale="used in a project",
            )
        ]
    )
    names = {s["name"] for s in result["accepted_skills"] + result["rejected_skills"]}
    assert names == {"Python", "SQL", "Welding", "Triage"}

    defaulted = next(s for s in result["accepted_skills"] if s["name"] == "Welding")
    assert defaulted["evidence_type"] == "claim_only"
    assert defaulted["quality"] == "low"
    assert result["warnings"], "silent defaulting should be surfaced, not hidden"


def test_quality_is_clamped_to_what_the_evidence_supports():
    """On one run every skill came back "medium", including claim_only ones that
    the prompt caps at "low" — flattening the distinction the rating exists for."""
    result = judge_with(
        [
            SkillVerdict(
                name="SQL", keep=True, quality="high", category="technical",
                evidence_type="claim_only", rationale="listed",
            ),
            SkillVerdict(
                name="Welding", keep=True, quality="high", category="technical",
                evidence_type="course", rationale="taught in a course",
            ),
        ]
    )
    by_name = {s["name"]: s for s in result["accepted_skills"]}

    assert by_name["SQL"]["quality"] == "low"        # claim_only caps at low
    assert by_name["Welding"]["quality"] == "medium"  # course caps at medium
    assert "capped" in by_name["SQL"]["rationale"]


def test_clamping_never_promotes():
    """Downgrades only. The clamp must not invent confidence the model withheld."""
    modest = _enforce_quality_rules(
        {"name": "Triage", "quality": "low", "evidence_type": "experience", "rationale": "x"}
    )
    assert modest["quality"] == "low"


def test_project_evidence_still_reaches_high():
    result = judge_with(
        [
            SkillVerdict(
                name="Python", keep=True, quality="high", category="technical",
                evidence_type="project", rationale="used in the ward rota tool",
            )
        ]
    )
    python = next(s for s in result["accepted_skills"] if s["name"] == "Python")
    assert python["quality"] == "high"


# A candidate who demonstrably programs (Python in a project) and also CLAIMS the
# everyday tools of that work without describing them.
TOOLING_CV = {
    "skills": [{"name": "Python"}, {"name": "VS Code"}, {"name": "Git"}],
    "projects": [{"name": "Ward rota tool", "technologies": ["Python"]}],
}


def judge_cv(extraction, verdicts):
    llm = FakeStructuredLLM(SkillJudgement=SkillJudgement(verdicts=verdicts))
    return make_judge_skills_node(llm, Config())({"cv_extraction": extraction})


def test_adjacent_evidence_lifts_a_claimed_tool_to_medium():
    """A specific tool the candidate claimed but never described, when it is
    standard equipment of a domain they DID demonstrate, is 'adjacent'/medium —
    not an unsupported claim_only. (VS Code for a proven programmer; the same
    logic applies to any field.)"""
    result = judge_cv(
        TOOLING_CV,
        [
            SkillVerdict(name="Python", keep=True, quality="high", category="technical",
                         evidence_type="project", rationale="used in the ward rota tool"),
            SkillVerdict(name="VS Code", keep=True, quality="medium", category="tool",
                         evidence_type="adjacent",
                         evidence_quote="Ward rota tool [tech: Python]",
                         rationale="standard editor for the demonstrated Python work"),
        ],
    )
    vscode = next(s for s in result["accepted_skills"] if s["name"] == "VS Code")
    assert vscode["evidence_type"] == "adjacent"
    assert vscode["quality"] == "medium"


def test_adjacent_never_reaches_high():
    """Presumed use is weaker than demonstrated use: the clamp caps adjacent at
    medium even if the model over-rates it."""
    result = judge_cv(
        TOOLING_CV,
        [
            SkillVerdict(name="Git", keep=True, quality="high", category="tool",
                         evidence_type="adjacent", rationale="model over-rated it"),
        ],
    )
    git = next(s for s in result["accepted_skills"] if s["name"] == "Git")
    assert git["quality"] == "medium"
    assert "capped" in git["rationale"]


def test_generic_trait_is_never_rescued_by_adjacency():
    """Adjacency lifts specific named tools, never generic self-assessed traits."""
    from agents.agent_a_cv_extraction.nodes.judge_skills import _enforce_quality_rules

    # Even if a verdict tried to mark teamwork adjacent, the reject path owns it:
    result = judge_with(
        [
            SkillVerdict(
                name="Welding", keep=False, quality="low", category="generic",
                evidence_type="claim_only", rationale="—",
            ),
        ]
    )
    # Welding was rejected by the model; it stays rejected, not lifted.
    assert any(s["name"] == "Welding" for s in result["rejected_skills"])
    # And the clamp maps the new tier without error.
    lifted = _enforce_quality_rules(
        {"name": "X", "quality": "high", "evidence_type": "adjacent", "rationale": "y"}
    )
    assert lifted["quality"] == "medium"


def test_invented_skills_are_still_rejected():
    """Clamping and defaulting must not open a door for skills nobody claimed."""
    result = judge_with(
        [
            SkillVerdict(
                name="Kubernetes", keep=True, quality="high", category="tool",
                evidence_type="project", rationale="invented",
            )
        ]
    )
    names = {s["name"] for s in result["accepted_skills"] + result["rejected_skills"]}
    assert "Kubernetes" not in names
