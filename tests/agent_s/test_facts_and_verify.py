"""The fact sheet and the verifier — offline, no database, no API.

These are the two pieces of Agent S that are code rather than prompt, which is
to say they are the two pieces that actually enforce anything.
"""

from __future__ import annotations

import pytest

from agents.agent_s_assistant.facts import (
    build_fact_sheet,
    deterministic_answer,
    verify_answer,
)
from agents.agent_s_assistant.graph import build_assistant_graph
from agents.agent_s_assistant.nodes import Deps
from agents.agent_s_assistant.schemas import AssistantReply
from shared.config import Config

JOBS = [{"title": "Junior Data Analyst", "employer": "Bank Muscat",
         "location": "Muscat", "why": "SQL covers their requirement for SQL"}]
COURSES = [{"title": "Power BI Essentials", "provider": "Coursera",
            "why": "covers Power BI"}]


def sheet(**over) -> str:
    kwargs = {"readiness": 42, "jobs": JOBS, "courses": COURSES,
              "gaps": ["Power BI"], "suggested_role": None,
              "matched_at": "2026-08-15T10:00:00+00:00"}
    kwargs.update(over)
    return build_fact_sheet(**kwargs)


# ---------------------------------------------------------------------------
# the fact sheet
# ---------------------------------------------------------------------------
def test_no_machine_timestamp_ever_reaches_the_sheet():
    """The bug, as a regression.

    A real user was told their results were produced on
    "2026-08-09T04:09:49.388371+00:00". The model echoed what it was handed, so
    this asserts on what it is handed.
    """
    text = sheet(matched_at="2026-08-09T04:09:49.388371+00:00")

    assert "T04:09" not in text and "388371" not in text
    assert "9 August 2026" in text


@pytest.mark.parametrize("delta_days, expected", [
    (0, "today"), (1, "yesterday"), (6, "6 days ago"), (29, "29 days ago"),
])
def test_how_long_ago_is_stated_because_staleness_is_a_real_question(delta_days, expected):
    """"Are my results out of date?" cannot be answered from a timestamp without
    arithmetic the model should not be doing."""
    from datetime import datetime, timedelta, timezone
    when = datetime.now(timezone.utc) - timedelta(days=delta_days, hours=1)

    assert expected in sheet(matched_at=when.isoformat())


def test_an_old_result_gets_a_date_without_a_misleading_day_count():
    """"412 days ago" is arithmetic, not information."""
    from datetime import datetime, timedelta, timezone
    when = datetime.now(timezone.utc) - timedelta(days=400)

    text = sheet(matched_at=when.isoformat())
    assert "days ago" not in text
    assert str(when.year) in text


def test_an_unparseable_date_survives_rather_than_crashing():
    """Better a strange string in one line than no fact sheet at all."""
    assert "sometime" in sheet(matched_at="sometime")


def test_a_missing_readiness_is_stated_not_omitted():
    """Null is not zero, and an omitted line invites the model to fill the gap.
    A line that states the absence gives it the true answer to quote instead."""
    text = sheet(readiness=None)

    assert "not measured yet" in text
    assert "0/100" not in text


def test_it_carries_the_evidence_chain_not_just_the_title():
    assert "SQL covers their requirement for SQL" in sheet()


def test_the_two_rankings_are_not_conflated():
    """The bug this file had for about ten minutes, caught on live data.

    The list is ordered by market-wide demand; the count is how many of THIS
    person's matched roles wanted the skill. Measured on a real profile, `API
    integration` had the highest count (5) and sat LAST. Labelling the order
    "most in demand first" invited exactly the wrong reading, so the sheet now
    says which is which and warns that position is not the count.
    """
    text = sheet()

    assert "whole job market" in text
    assert "do not call the first one the most needed" in text
    assert "most in demand first" not in text, "the misleading label came back"


def test_position_and_count_can_disagree_and_the_sheet_survives_it():
    """The real shape that exposed the bug: last in the list, highest count."""
    text = sheet(gaps=[{"skill": "communication skills", "jobs_missing_in": 3},
                       {"skill": "API integration", "jobs_missing_in": 5}])

    assert text.index("communication skills") < text.index("API integration")
    assert "5 of the roles they matched asked for it" in text


def test_a_measured_demand_count_is_published_but_the_internal_weight_is_not():
    text = sheet(gaps=[{"skill": "Power BI", "jobs_missing_in": 3,
                        "low_confidence": False}])

    assert "3 of the roles they matched asked for it" in text
    # priority_score is an internal weight on no meaningful scale, and an answer
    # naming it is rejected outright — so it must never be in the sheet either.
    assert "priority" not in text.lower()


def test_thin_demand_data_is_flagged_rather_than_stated_flatly():
    text = sheet(gaps=[{"skill": "welding", "jobs_missing_in": 1,
                        "low_confidence": True}])

    assert "thin data" in text


def test_a_gap_with_no_demand_count_still_appears():
    """`jobs_missing_in` is absent on an older envelope. The skill is still real
    and must be listed — just without a number nobody measured."""
    text = sheet(gaps=[{"skill": "Power BI"}, "TypeScript"])

    assert "Power BI" in text and "TypeScript" in text


def test_nothing_but_this_users_rows_appear():
    """The sheet is built from arguments the caller assembled per user. There is
    no code path here that could reach a second user, which is the property the
    API-level isolation test verifies end to end."""
    text = sheet()

    assert "Bank Muscat" in text
    assert "average" not in text.lower() and "other user" not in text.lower()


# ---------------------------------------------------------------------------
# the verifier
# ---------------------------------------------------------------------------
def test_a_number_absent_from_the_record_is_rejected():
    problem = verify_answer("You matched 47 jobs.", sheet())

    assert problem and "47" in problem


def test_a_number_present_in_the_record_is_allowed():
    assert verify_answer("Your readiness is 42 out of 100.", sheet()) is None


def test_comma_and_trailing_zero_forms_compare_equal():
    """'1,919' and '1919' are the same figure, and so are '4.70' and '4.7'.
    Without this a correct answer is rejected for formatting."""
    facts = "Enrolled: 1919 people at 4.7 stars"

    assert verify_answer("1,919 people, rated 4.70", facts) is None


@pytest.mark.parametrize("token", ["gap_score", "ESCO", "posting_id", "priority_score"])
def test_internal_vocabulary_is_rejected(token):
    """`gap_score` especially: it is scaled so 0.0 is the BEST result, so a user
    shown that number reads it backwards."""
    assert verify_answer(f"Your {token} is fine.", sheet()) is not None


def test_an_empty_answer_is_rejected():
    assert verify_answer("", sheet()) is not None


def test_vague_praise_still_passes_and_that_is_the_known_limit():
    """The record cannot adjudicate "you're in good shape", so this check does
    not pretend to. Pinned so the limit is known rather than assumed away."""
    assert verify_answer("You are in a strong position overall.", sheet()) is None


# ---------------------------------------------------------------------------
# the fallback
# ---------------------------------------------------------------------------
def test_the_fallback_is_a_real_sentence_not_an_empty_string():
    assert deterministic_answer(sheet(), has_results=True).strip()
    assert deterministic_answer(sheet(), has_results=False).strip()


def test_it_says_results_are_not_ready_when_they_are_not():
    assert "not ready yet" in deterministic_answer(sheet(), has_results=False)


# ---------------------------------------------------------------------------
# the graph
# ---------------------------------------------------------------------------
class FakeLLM:
    """Callable, because the graph composes `prompt | llm` and LCEL accepts a
    Runnable, a plain callable or a dict — an object with only `.invoke` is
    rejected before `.invoke` is ever consulted."""

    def __init__(self, reply): self.reply = reply
    def __call__(self, payload): return self.reply


def run(reply, *, question="how am I doing?") -> dict:
    graph = build_assistant_graph(Deps(config=Config(), llm=FakeLLM(reply)))
    return graph.invoke({"question": question, "fact_sheet": sheet(),
                         "history": [], "has_results": True})


def test_a_verified_answer_is_published_as_the_model_wrote_it():
    out = run(AssistantReply(answer="Your readiness is 42 out of 100."))

    assert out["answer"] == "Your readiness is 42 out of 100."
    assert out["answer_source"] == "model"


def test_a_rejected_answer_loses_its_rerun_proposal_too():
    """A model that just said something unsupported does not get to keep
    proposing that the user spend their one weekly credit."""
    out = run(AssistantReply(answer="You matched 47 jobs.", intent="propose_rerun",
                             rerun_reason="things change"))

    assert "47" not in out["answer"]
    assert out["answer_source"] == "template"
    assert out["proposed_rerun"] is False


def test_no_model_still_answers():
    graph = build_assistant_graph(Deps(config=Config(), llm=None))
    out = graph.invoke({"question": "hi", "fact_sheet": sheet(),
                        "history": [], "has_results": True})

    assert out["answer"].strip()
    assert out["answer_source"] == "template"


def test_the_graph_cannot_reach_a_database_or_start_a_run():
    """Structural, not behavioural: Agent S is handed everything it may know.

    The graph has no store, no runner and no config beyond thresholds, so
    "the model cannot act" is a fact about what is in scope here rather than a
    promise made in a prompt.
    """
    deps = Deps(config=Config(), llm=None)

    assert not hasattr(deps, "store")
    assert not hasattr(deps, "runner")
