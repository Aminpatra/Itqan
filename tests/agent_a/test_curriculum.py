"""Curriculum research tests.

This node is the only place the pipeline admits knowledge from outside the
documents, so its guardrails get tested harder than its happy path. The failure
mode that matters is a fabricated syllabus silently promoting an unearned skill.
"""

from __future__ import annotations

import json
from pathlib import Path

from agents.agent_a_cv_extraction.graph import build_graph
from agents.agent_a_cv_extraction.nodes.curriculum import (
    MAX_CREDENTIALS,
    collect_credentials,
    make_research_curriculum_node,
)
from agents.agent_a_cv_extraction.nodes.judge_skills import build_evidence_context
from shared.config import Config
from tests.fake_llm import FakeStructuredLLM

FIXTURE = Path(__file__).resolve().parent.parent / "fixtures" / "sample_cv.txt"
TRANSCRIPT = Path(__file__).resolve().parent.parent / "fixtures" / "sample_transcript.txt"


def test_collects_courses_and_certificates_without_duplicates():
    state = {
        "cv_extraction": {
            "certifications": [{"name": "CCNA v7: Introduction to Networks"}],
            "courses": [{"title": "Machine Learning"}],
        },
        "transcript_extraction": {
            "courses": [{"title": "Machine Learning"}, {"title": "Database Systems"}]
        },
    }
    found = collect_credentials(state)
    names = [c["credential_name"] for c in found]

    assert "CCNA v7: Introduction to Networks" in names
    assert "Database Systems" in names
    assert names.count("Machine Learning") == 1, "transcript duplicate should collapse"


def test_ignores_ocr_debris_and_caps_volume():
    state = {
        "cv_extraction": {
            "courses": [{"title": "ML"}, {"title": ""}, {"title": None}]
            + [{"title": f"Real Course Number {i}"} for i in range(60)]
        }
    }
    found = collect_credentials(state)
    assert all(len(c["credential_name"]) >= 4 for c in found)
    assert len(found) <= MAX_CREDENTIALS


def test_courses_matching_a_claimed_skill_survive_the_cap():
    """A transcript arrives chronologically, so a naive cut drops whatever is at
    the end. Courses that share a word with a claimed skill are promoted ahead of
    the truncation, whatever the discipline."""
    early = [{"title": f"Unrelated Subject {i}"} for i in range(MAX_CREDENTIALS + 5)]
    state = {
        "cv_extraction": {
            "skills": [{"name": "Pharmacology"}],
            "courses": early + [{"title": "Clinical Pharmacology"}],
        }
    }
    names = [c["credential_name"] for c in collect_credentials(state)]
    assert "Clinical Pharmacology" in names


def test_prioritisation_has_no_opinion_about_subject_matter():
    """Guards against reintroducing a stoplist of "low-yield" subjects. Such a
    list encodes one discipline's assumptions and would bury exactly the courses
    that matter for a linguist, a lawyer or a nurse."""
    state = {
        "cv_extraction": {
            "skills": [],
            "courses": [
                {"title": "Advanced Arabic Translation"},
                {"title": "Medical Ethics"},
                {"title": "Public Speaking"},
                {"title": "Distributed Systems"},
            ],
        }
    }
    names = [c["credential_name"] for c in collect_credentials(state)]
    # Order preserved, nothing demoted for being non-technical.
    assert names == [
        "Advanced Arabic Translation",
        "Medical Ethics",
        "Public Speaking",
        "Distributed Systems",
    ]


def test_certifications_are_never_cut():
    state = {
        "cv_extraction": {
            "certifications": [{"name": "CCNA v7: Introduction to Networks"}],
            "courses": [{"title": f"Filler Course {i}"} for i in range(MAX_CREDENTIALS + 10)],
        }
    }
    names = [c["credential_name"] for c in collect_credentials(state)]
    assert "CCNA v7: Introduction to Networks" in names


def test_unrecognised_credentials_are_discarded():
    """A credential the model admits not knowing must not reach the judge. An
    unknown local hackathon carries no curriculum, and passing it through with
    empty lists would still let the judge treat it as researched."""
    node = make_research_curriculum_node(FakeStructuredLLM(), Config())
    result = node(
        {
            "cv_extraction": {
                "skills": [{"name": "scikit-learn"}],
                "certifications": [{"name": "Campus Coding Marathon 2024"}],
                "courses": [{"title": "Machine Learning"}],
            }
        }
    )
    names = {entry["credential_name"] for entry in result["curriculum"]}
    assert "Machine Learning" in names
    assert "Campus Coding Marathon 2024" not in names


def test_credentials_never_asked_about_are_discarded():
    """The fake returns a credential absent from the input. Anything the model
    invents rather than was asked about must be dropped, or a hallucinated
    syllabus could corroborate a skill."""
    node = make_research_curriculum_node(FakeStructuredLLM(), Config())
    result = node(
        {
            "cv_extraction": {
                "skills": [{"name": "scikit-learn"}],
                "courses": [{"title": "Machine Learning"}],
            }
        }
    )
    names = {entry["credential_name"] for entry in result["curriculum"]}
    assert "Fabricated Course Nobody Asked About" not in names
    assert "Kubernetes" not in json.dumps(result["curriculum"])


def test_skipped_when_there_are_no_credentials():
    node = make_research_curriculum_node(FakeStructuredLLM(), Config())
    result = node({"cv_extraction": {"skills": [{"name": "Python"}]}})
    assert result["curriculum"] == []


def test_skipped_when_there_are_no_skills_to_corroborate():
    """Curriculum only ever corroborates claimed skills, so with no claims there
    is nothing to research and no call worth paying for."""
    fake = FakeStructuredLLM()
    node = make_research_curriculum_node(fake, Config())
    result = node({"cv_extraction": {"skills": [], "courses": [{"title": "Machine Learning"}]}})
    assert result["curriculum"] == []
    assert "CurriculumResearch" not in fake.calls


def test_evidence_context_fences_curriculum_as_non_document():
    """The judge must be able to tell syllabus background from CV text."""
    context = build_evidence_context(
        {"projects": [{"name": "ArabicNER", "technologies": ["PyTorch"]}]},
        None,
        [
            {
                "credential_name": "Machine Learning",
                "credential_kind": "course",
                "typical_skills": ["scikit-learn"],
                "typical_concepts": ["gradient descent"],
            }
        ],
    )
    assert "PROJECT: ArabicNER" in context
    assert "CURRICULUM OF COURSE" in context
    assert "not stated in the CV" in context
    assert "scikit-learn" in context


def test_skill_prompt_states_evidence_precedence():
    """Guards a rule that cost a real regression to find.

    When curriculum evidence was first added, the judge started preferring it over
    project evidence, and a candidate's JavaScript/PHP/MERN dropped from
    high/project to medium/course — their actual shipped web system outranked by a
    syllabus. Prompt behaviour cannot be asserted without a live model, so this at
    least fails if the precedence rule is edited away.
    """
    from agents.agent_a_cv_extraction.prompts import SKILL_JUDGE_PROMPT

    text = SKILL_JUDGE_PROMPT.template
    assert "EVIDENCE PRECEDENCE" in text
    assert "never lower" in text
    assert "caps quality at \"medium\"" in text


def test_curriculum_prompt_demands_components_not_umbrellas():
    """A course that returns only a collective noun for its field corroborates
    nothing, because candidates list individual tools, not the umbrella."""
    from agents.agent_a_cv_extraction.prompts import CURRICULUM_RESEARCH_PROMPT

    text = CURRICULUM_RESEARCH_PROMPT.template
    assert "NAME THE COMPONENTS, NOT THE UMBRELLA" in text


def test_prompts_are_not_tuned_to_one_candidate_or_field():
    """Guards against re-baking specifics from a real CV into the prompts.

    Naming actual credentials from one candidate's documents as examples makes
    the prompt fragile and quietly domain-locks the agent — a habit worth failing
    a test over rather than rediscovering later.
    """
    from agents.agent_a_cv_extraction.prompts import (
        CURRICULUM_RESEARCH_PROMPT,
        SKILL_JUDGE_PROMPT,
    )

    combined = CURRICULUM_RESEARCH_PROMPT.template + SKILL_JUDGE_PROMPT.template
    for specific in ("PyHack", "NextGen", "Infinity to Innovation", "Gheath", "UTAS"):
        assert specific not in combined, f"prompt is tuned to one CV: {specific!r}"


def test_curriculum_is_published_in_provenance(tmp_path):
    from langgraph.types import Command

    from agents.agent_a_cv_extraction.nodes.human_review import SKIP_SENTINEL

    app = build_graph(FakeStructuredLLM(), Config())
    config = {"configurable": {"thread_id": f"curric-{tmp_path.name}"}}
    state = app.invoke(
        {
            "cv_path": str(FIXTURE),
            "transcript_path": str(TRANSCRIPT),
            "run_id": "curric",
            "output_dir": str(tmp_path),
            "interactive": False,
            "review_rounds": 0,
        },
        config=config,
    )
    while state.get("__interrupt__"):
        state = app.invoke(Command(resume=SKIP_SENTINEL), config=config)

    profile = json.loads(Path(state["final_output_path"]).read_text(encoding="utf-8"))
    researched = profile["provenance"]["curriculum_researched"]

    assert researched, "curriculum evidence should be published for the consumer"
    assert {"credential_name", "typical_skills"} <= set(researched[0])
    # The consumer must never see an unrecognised credential presented as researched.
    assert all(entry["typical_skills"] or entry["typical_concepts"] for entry in researched)
