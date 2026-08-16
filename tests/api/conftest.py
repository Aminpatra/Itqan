"""API tests: the real routes, a real database, a fake pipeline.

The pipeline is faked through the same `PipelineRunner` seam the production app
injects — no monkeypatching — so these tests exercise the actual routing, auth,
job lifecycle and mapping without OCR, OpenAI or a 90-second wait.

The database is real, because every route's behaviour is SQL: ownership scoping,
`ON CONFLICT` upserts and the `latest_completed_run` ordering cannot be checked
against a dictionary. Skips cleanly when `ITQAN_TEST_DATABASE_URL` is unset, the
same contract the agents' DB suites use.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient      # noqa: E402

from api.db import AppStore, apply_migrations   # noqa: E402
from api.jobs import PipelineRunner             # noqa: E402
from api.main import create_app                 # noqa: E402
from shared.config import Config                # noqa: E402

# `app_reset_throttle` is listed EXPLICITLY and must stay listed: it is the one
# table here with no foreign key to `app_users`, so `TRUNCATE ... CASCADE` does
# not reach it. Left out, its hourly counters accumulate across tests until the
# limit is exhausted, and every later test quietly receives no email — which is
# exactly how it failed the first time, as fifteen unrelated assertions.
_APP_TABLES = ("app_assistant_messages", "app_assistant_usage", "app_reset_throttle",
               "app_password_resets", "app_profiles", "app_onboarding_progress",
               "app_runs", "app_documents", "app_users")


def pytest_collection_modifyitems(config, items):  # noqa: ARG001
    """Skip everything in THIS directory when no test database is configured.

    The path check is load-bearing and its absence was a real bug: a
    `pytest_collection_modifyitems` in a subdirectory's conftest still receives
    every item in the session, so the unscoped version marked the entire suite
    skipped. `pytest tests/ -q` reported "754 skipped" and looked like a pass —
    the worst possible failure mode for a test suite. Scoped the same way
    `tests/agent_b/db/conftest.py` does it.
    """
    if os.getenv("ITQAN_TEST_DATABASE_URL"):
        return
    here = Path(__file__).parent
    skip = pytest.mark.skip(reason="ITQAN_TEST_DATABASE_URL is not set")
    for item in items:
        if here in Path(item.fspath).parents:
            item.add_marker(skip)


@pytest.fixture
def dsn() -> str:
    url = os.getenv("ITQAN_TEST_DATABASE_URL")
    if not url:
        pytest.skip("ITQAN_TEST_DATABASE_URL is not set")
    apply_migrations(url)
    return url


class FakeRunner(PipelineRunner):
    """Returns envelopes shaped like the real ones, and records the stage calls.

    `fail_at` makes a specific agent raise, which is how the failure paths are
    tested — the UI has a first-class recovery route for an unreadable document
    and it is only reachable if the error names the agent.
    """

    def __init__(self, *, fail_at: str | None = None) -> None:
        # A real Config, because the base class reaches for `self.config` — the
        # profile-restore path resolves `output_dir` through it. A fake that skips
        # the parent's __init__ diverges from the thing it stands in for.
        super().__init__(Config())
        self.fail_at = fail_at
        self.stages: list[str] = []
        self.calls: list[str] = []
        # The flags each agent was invoked with, which is how the preference tests
        # check that the user's answers actually reached the pipeline rather than
        # being written to a table and forgotten.
        self.flags: dict[str, list[str]] = {}

    def run_agent_a(self, *, cv_paths, transcript_paths, run_id, on_node) -> dict[str, Any]:
        self.calls.append("A")
        # Replays the REAL node names in the real order, so the tests drive the
        # actual checkpoint schedule rather than a made-up one. A rename in Agent
        # A's graph that the progress policy has not been told about shows up here
        # as a bar that never reaches the pause.
        for node in ("ingest", "extract_text", "llm_extract_cv",
                     *(("llm_extract_transcript",) if transcript_paths else ()),
                     "verify_grounding", "assess_gaps", "research_curriculum",
                     "judge_skills", "derive_coursework_skills", "summarize", "persist"):
            on_node(node)
        if self.fail_at == "A":
            raise RuntimeError("agent_a_unreadable_document")
        return {
            "candidate": {
                "full_name": "Maryam Al Balushi",
                "education": [{"end_date": "2025-06"}],
            },
            "skills": {"accepted": [
                {"name": "SQL", "quality": "high", "evidence_type": "course",
                 "corroborating_credential": "Database Systems II"},
                {"name": "Python", "quality": "high", "evidence_type": "project"},
            ], "rejected": []},
            "provenance": {"grounding": {
                "full_name": {"grounded": True, "score": 0.97, "method": "exact",
                              "evidence_quote": "Maryam Al Balushi"},
            }},
            "confidence": {"overall": 0.91},
        }

    def run_agent_c(self, *, run_id, flags=None, on_node=None) -> dict[str, Any]:
        self.calls.append("C")
        self.flags["C"] = list(flags or [])
        for node in ("build_query_embedding", "retrieve_postings", "map_candidate_skills",
                     "resolve_ambiguous_skills", "gap_analysis", "persist"):
            if on_node:
                on_node(node)
        if self.fail_at == "C":
            raise RuntimeError("agent_c_no_market_data")
        return {
            "used_fallback": False,
            "matched_jobs": [{
                "job_id": "jp_991", "job_title": "Junior Data Analyst",
                "gap_score": 0.42, "source": "el7far",
                "source_url": "https://example.test/jp_991",
                "posted_date": "2026-07-28", "location": "Muscat",
                "seniority_level": "junior",
                "skill_resolution": [
                    {"skill": "SQL", "verdict": "matched", "resolved_by": "esco",
                     "satisfied_by": "SQL", "best_similarity": None},
                    {"skill": "Power BI", "verdict": "missing", "resolved_by": "cosine",
                     "satisfied_by": None, "best_similarity": 0.31},
                ],
            }],
            "aggregate": {
                "missing_skill_details": [
                    {"skill": "Power BI", "esco_code": "uri:pbi", "priority_score": 3.21,
                     "jobs_missing_in": 3, "low_confidence": False},
                ],
                "most_common_missing_skills": ["Power BI"],
                "average_gap_score": 0.42,
                "jobs_scored": 1,
            },
        }

    def run_agent_e(self, *, run_id, flags=None, on_node=None) -> dict[str, Any]:
        self.calls.append("E")
        self.flags["E"] = list(flags or [])
        for node in ("load_missing_skills", "retrieve_candidate_courses",
                     "greedy_cover_assign", "attach_flags", "generate_rationale",
                     "persist"):
            if on_node:
                on_node(node)
        if self.fail_at == "E":
            raise RuntimeError("agent_e_no_courses")
        return {
            "generated_at": "2026-07-29T10:00:00+00:00",
            "recommendations": [
                {"skill": "Power BI", "no_course_found": False,
                 "supply": {"courses_available": 9, "thin": False},
                 "course": {
                     "course_id": "c_88", "title": "Power BI Essentials",
                     "provider": "Coursera", "url": "https://example.test/c88",
                     "covers_other_skills": ["data visualisation"],
                     "quality": {"rating": 4.7, "review_count": 900,
                                 "price": None, "level": "beginner"},
                     "rationale": "Covers Power BI.", "rationale_source": "template",
                 }},
                {"skill": "welding", "no_course_found": True, "course": None,
                 "supply": {"courses_available": 0, "thin": True}},
            ],
        }


@pytest.fixture
def runner() -> FakeRunner:
    return FakeRunner()


class FakeAssistantLLM:
    """Stands in for `structured(build_llm(...), AssistantReply)`.

    Returns whatever `reply` is set to, and records what it was shown — which is
    how the isolation tests check the FACT SHEET rather than only the prose. An
    answer that happens not to mention another user is luck; a prompt that never
    contained them is the guarantee.

    Callable rather than `.invoke`-only: the graph composes `prompt | llm`, and
    LCEL accepts a Runnable, a plain callable or a dict — an object exposing
    only `invoke` is rejected before `invoke` is ever consulted.
    """

    def __init__(self, reply: Any = None) -> None:
        from agents.agent_s_assistant.schemas import AssistantReply
        self.reply = reply or AssistantReply(answer="Here is what your results say.")
        self.prompts: list[str] = []

    def __call__(self, payload: Any) -> Any:
        self.prompts.append(str(payload))
        return self.reply


@pytest.fixture
def assistant_llm() -> "FakeAssistantLLM":
    return FakeAssistantLLM()


@pytest.fixture
def store(dsn: str) -> AppStore:
    s = AppStore(dsn)
    with s.connect().cursor() as cur:
        cur.execute(f"TRUNCATE {', '.join(_APP_TABLES)} CASCADE")
    yield s
    s.close()


@pytest.fixture
def client(dsn: str, store: AppStore, runner: FakeRunner,
           assistant_llm: "FakeAssistantLLM") -> TestClient:
    app = create_app(Config(database_url=dsn), store=store, runner=runner,
                     assistant_llm=assistant_llm, migrate=False)
    return TestClient(app)


@pytest.fixture
def signed_in(client: TestClient) -> TestClient:
    """A registered, signed-in client — the state every app route requires."""
    res = client.post("/api/placeholder/signup",
                      data={"email": "maryam@itqan.test", "password": "Str0ng!pass",
                            "name": "Maryam Al Balushi"})
    assert res.status_code == 200, res.text
    return client
