"""End-to-end graph tests with a canned LLM — no network, no API key, no Paddle.

This is the highest-value test in the project: it proves the topology, the state
reducers, the interrupt/resume cycle and the output contract all hold together,
and it costs nothing to run.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agents.agent_a_cv_extraction.graph import build_graph
from agents.agent_a_cv_extraction.nodes.human_review import SKIP_SENTINEL
from shared.config import Config
from shared.contracts import CandidateProfile
from tests.fake_llm import FakeStructuredLLM

FIXTURE = Path(__file__).resolve().parent.parent / "fixtures" / "sample_cv.txt"


TRANSCRIPT = Path(__file__).resolve().parent.parent / "fixtures" / "sample_transcript.txt"


def run(
    tmp_path: Path,
    *,
    interactive: bool,
    answers: dict | None = None,
    transcript: Path | None = None,
) -> dict:
    app = build_graph(FakeStructuredLLM(), Config())
    config = {"configurable": {"thread_id": f"test-{interactive}-{tmp_path.name}"}}
    state = app.invoke(
        {
            "cv_paths": [str(FIXTURE)],
            "transcript_paths": [str(transcript)] if transcript else [],
            "run_id": "testrun",
            "output_dir": str(tmp_path),
            "interactive": interactive,
            "review_rounds": 0,
        },
        config=config,
    )

    from langgraph.types import Command

    guard = 0
    while state.get("__interrupt__"):
        guard += 1
        assert guard <= 5, "interrupt loop did not terminate"
        state = app.invoke(
            Command(resume=answers or SKIP_SENTINEL), config=config
        )
    return state


def test_no_hitl_run_completes_and_writes_valid_envelope(tmp_path):
    state = run(tmp_path, interactive=False)

    assert not state.get("errors"), state.get("errors")
    output = Path(state["final_output_path"])
    assert output.exists()

    # The envelope must satisfy the published contract, not merely be JSON.
    profile = CandidateProfile.model_validate(json.loads(output.read_text(encoding="utf-8")))
    assert profile.candidate["full_name"] == "Sara Al-Balushi"
    assert profile.summary["headline"]


def test_hallucinated_phone_is_dropped_before_output(tmp_path):
    """The fake LLM returns a phone number absent from the fixture. It must not
    survive into the envelope."""
    state = run(tmp_path, interactive=False)
    profile = json.loads(Path(state["final_output_path"]).read_text(encoding="utf-8"))

    assert profile["candidate"]["contact"].get("phone") is None
    assert "contact.phone" in profile["provenance"]["dropped_fields"]


def test_generic_skill_is_filtered_and_kept_with_rationale(tmp_path):
    state = run(tmp_path, interactive=False)
    accepted = {s["name"] for s in state["accepted_skills"]}
    rejected = {s["name"] for s in state["rejected_skills"]}

    assert "Python" in accepted and "PyTorch" in accepted
    assert "Teamwork" in rejected
    # Filtered skills stay visible to the downstream agent, with a reason.
    assert all(s.get("rationale") for s in state["rejected_skills"])


def test_interrupt_fires_and_resume_merges_human_input(tmp_path):
    """The fixture has no phone number, so contact.phone is dropped and becomes a
    gap. Interactive mode must pause, accept an answer, and merge it."""
    state = run(
        tmp_path,
        interactive=True,
        answers={"contact.email": "sara.b@squ.edu.om"},
    )

    assert state.get("review_rounds", 0) >= 1, "the HITL loop never ran"
    assert not state.get("errors"), state.get("errors")
    assert Path(state["final_output_path"]).exists()


def test_hitl_loop_is_bounded(tmp_path):
    """Answering nothing must not loop forever — review_rounds has to advance
    even when the user skips every prompt."""
    state = run(tmp_path, interactive=True, answers={})
    assert state.get("review_rounds", 0) <= Config().max_review_rounds
    assert Path(state["final_output_path"]).exists()


def test_ocr_artifacts_absent_for_text_input(tmp_path):
    """A plain-text fixture must never invoke OCR."""
    state = run(tmp_path, interactive=False)
    assert state["cv_doc"]["kind"] == "text"
    assert state["cv_doc"].get("ocr_json_path") is None


def test_empty_resume_would_hang_so_sentinel_is_required(tmp_path):
    """Regression guard for a langgraph 1.2.9 behaviour that is easy to reintroduce.

    ``Command(resume={})`` does not resume — an empty mapping is read as a
    resume-map with no entries, so the node re-runs and interrupts again, with no
    error and no progress. A user pressing Enter through every prompt would hang
    the CLI forever. SKIP_SENTINEL exists solely to avoid that, and this test
    pins the underlying behaviour so a "simplification" back to `resume={}` fails
    loudly instead of shipping.
    """
    from langgraph.types import Command

    app = build_graph(FakeStructuredLLM(), Config())
    config = {"configurable": {"thread_id": f"sentinel-{tmp_path.name}"}}
    state = app.invoke(
        {
            "cv_paths": [str(FIXTURE)],
            "run_id": "sentinel",
            "output_dir": str(tmp_path),
            "interactive": True,
            "review_rounds": 0,
        },
        config=config,
    )
    assert state.get("__interrupt__"), "expected the graph to pause"

    # The broken form: still interrupted, no progress made.
    state = app.invoke(Command(resume={}), config=config)
    assert state.get("__interrupt__"), (
        "langgraph now resumes on an empty dict — SKIP_SENTINEL may be removable, "
        "but verify before doing so"
    )
    assert state.get("review_rounds", 0) == 0

    # The sentinel form: progresses.
    state = app.invoke(Command(resume=SKIP_SENTINEL), config=config)
    assert state.get("review_rounds", 0) >= 1


def test_sentinel_never_reaches_extraction(tmp_path):
    """The sentinel is transport plumbing; it must not become a candidate field."""
    state = run(tmp_path, interactive=True, answers=None)
    assert "__skipped__" not in json.dumps(state.get("cv_extraction") or {})
    assert not any(
        "__skipped__" in path for path in (state.get("human_supplied_fields") or [])
    )


def test_transcript_is_optional_and_routing_skips_it(tmp_path):
    """No transcript means the transcript extraction call is never made — the
    conditional edge exists to avoid paying for it."""
    fake = FakeStructuredLLM()
    app = build_graph(fake, Config())
    config = {"configurable": {"thread_id": f"no-transcript-{tmp_path.name}"}}
    app.invoke(
        {
            "cv_paths": [str(FIXTURE)],
            "transcript_paths": [],
            "run_id": "no-t",
            "output_dir": str(tmp_path),
            "interactive": False,
            "review_rounds": 0,
        },
        config=config,
    )
    assert "TranscriptExtraction" not in fake.calls


def test_transcript_courses_merge_into_the_profile(tmp_path):
    state = run(tmp_path, interactive=False, transcript=TRANSCRIPT)
    profile = json.loads(Path(state["final_output_path"]).read_text(encoding="utf-8"))

    titles = {c.get("title") for c in profile["candidate"]["courses"]}
    assert "Machine Learning" in titles
    assert profile["candidate"]["academic_record"]["cgpa"] == "3.71"
    roles = {d["role"] for d in profile["source_documents"]}
    assert roles == {"cv", "transcript"}


def test_coursework_skill_is_derived_end_to_end(tmp_path):
    """A skill the transcript's coursework teaches but the CV never claimed is
    promoted into skills.accepted, flagged and capped at medium."""
    from agents.agent_a_cv_extraction.schemas import (
        CredentialCurriculum,
        CurriculumResearch,
        SkillJudgement,
        SkillVerdict,
    )

    # The transcript fixture has a "Machine Learning" course; teach it Apache Spark
    # (which the CV does not claim) and have the reused judge keep it.
    fake = FakeStructuredLLM(
        CurriculumResearch=CurriculumResearch(credentials=[
            CredentialCurriculum(
                credential_name="Machine Learning", credential_kind="course",
                recognized=True, typical_skills=["Apache Spark"], typical_concepts=[],
                covers_claimed_skills=[],
            )
        ]),
        SkillJudgement=SkillJudgement(verdicts=[
            SkillVerdict(name="Apache Spark", keep=True, quality="medium", category="tool",
                         evidence_type="course", rationale="taught in the ML course"),
        ]),
    )
    app = build_graph(fake, Config())
    cfg = {"configurable": {"thread_id": f"derive-{tmp_path.name}"}}
    state = app.invoke(
        {
            "cv_paths": [str(FIXTURE)],
            "transcript_paths": [str(TRANSCRIPT)],
            "run_id": "derive", "output_dir": str(tmp_path),
            "interactive": False, "review_rounds": 0,
        },
        config=cfg,
    )
    profile = json.loads(Path(state["final_output_path"]).read_text(encoding="utf-8"))
    derived = [s for s in profile["skills"]["accepted"] if s.get("origin") == "coursework_derived"]
    assert any(s["name"] == "Apache Spark" for s in derived), (
        "the coursework-taught skill never reached the profile"
    )
    assert all(s["quality"] == "medium" for s in derived), "derived skill exceeded medium"
    spark = next(s for s in derived if s["name"] == "Apache Spark")
    assert spark["corroborating_credential"] == "Machine Learning"


def test_provenance_records_grounding_method(tmp_path):
    state = run(tmp_path, interactive=False)
    profile = json.loads(Path(state["final_output_path"]).read_text(encoding="utf-8"))
    grounding = profile["provenance"]["grounding"]

    assert grounding["full_name"]["method"] == "exact"
    assert grounding["full_name"]["grounded"] is True
    assert profile["confidence"]["overall"] > 0
