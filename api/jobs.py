"""The analysis worker: A, then a pause for the user, then C -> E.

The progress bar is driven by **actual phase completion**, never a timer. That was
an explicit requirement and it is also the only honest option: a clock-based bar
reaches 90% and stops while the user waits, and a stalled pipeline is
indistinguishable from a slow one. Here a stage advances when the work behind it
has finished, so a stuck stage reads as stuck.

    queued -> reading -> translating -> awaiting_confirmation -> matching -> done
                                                              \\-> failed

    reading                Agent A has ingested and read the documents (OCR)
    translating            Agent A has judged skills and derived coursework skills
    awaiting_confirmation  PHASE ONE IS DONE. The extracted details are stored and
                           the run waits for the user to correct them and answer
                           the preference questions
    matching               Agent C has produced the gap; Agent E is selecting
    done                   all three envelopes written
    failed                 the agent that failed is named in error_code

**Why the pause exists.** The first version ran all three agents in one pass, which
made the two things the user asked for impossible. The confirm screen could not
show the extracted details until Agent E had also finished — around three minutes
of skeleton for ninety seconds of relevant work — and the answers the user gave
during that wait could not influence anything, because Agent C had already run on
a profile that predated every one of them. Splitting the run at the confirmation
step fixes both with one structural change: the user's corrections and answers are
now *inputs* to the matching rather than a record of it.

`awaiting_confirmation` is different in kind from every other stage: the others
advance when work finishes, this one advances when a PERSON acts. That is why
`stale_runs` excludes it — someone taking ten minutes over a form is not a crashed
process.

Runs in a thread rather than a task queue. That is a deliberate scope choice for
one box: it survives the request finishing (which is the whole point of the async
transport) but not a process restart. A restart leaves a run 'reading' forever,
so `stale_runs` exists to find them, and moving to a real queue is a contained
change because the only contract is the `app_runs` row.
"""

from __future__ import annotations

import threading
import traceback
from pathlib import Path
from typing import Any, Callable, Optional

from api import progress as progress_policy
from shared.config import Config

# Which agent failed, in the vocabulary the UI branches on. The frontend has a
# first-class recovery path for an unreadable document (re-upload or manual
# entry), so naming the cause is what makes that path reachable.
ERROR_AGENT_A = "agent_a_unreadable_document"
ERROR_AGENT_C = "agent_c_no_market_data"
ERROR_AGENT_E = "agent_e_no_courses"
ERROR_UNKNOWN = "pipeline_failed"

# Where the run pauses. The number comes from `api/progress.py`, which owns how the
# bar is divided: Agent A takes the largest share because it IS the largest share
# (~70s of the ~90s measured end to end, and it carries the only unbounded step,
# OCR).
STAGE_AWAITING = "awaiting_confirmation"
PROGRESS_AWAITING = progress_policy.PHASE_ONE_SPAN[1]


# ---------------------------------------------------------------------------
# Preferences -> CLI flags
#
# The user answers four questions during the wait. Turning them into flags here
# rather than into agent state keeps `api/` an orchestrator: it drives the agent
# CLIs and never reaches into their internals, which is the same boundary
# `agents/pipeline.py` and `agents/status.py` observe.
#
# Every one of these RANKS, none EXCLUDES (user decision 2026-07-30). A preference
# that empties the result is worse than one that reorders it: a gap whose only
# course is paid still needs an answer, and "we found nothing" reads as a broken
# product rather than as respect for the answer.
# ---------------------------------------------------------------------------
def agent_c_flags(preferences: Optional[dict[str, Any]]) -> list[str]:
    """Retrieval preferences: which postings this candidate is compared against.

    `preferredRole` is the load-bearing one. Agent C embeds the candidate in the
    same essence shape Agent B embeds a posting in — a title line, then scalars,
    then skills — so the role lands in the *title* slot and genuinely changes which
    postings come back. `openToOtherRoles = 'no'` makes it replace the CV headline
    instead of joining it.

    `workArrangement` is a text bias only, and says so in the agent's calibration
    block: nothing in the corpus records whether a posting is remote, so claiming
    to filter on it would be a fabrication. Extracting it in Agent B is the real
    fix and is a separate piece of work.
    """
    prefs = preferences or {}
    flags: list[str] = []
    role = str(prefs.get("preferredRole") or "").strip()
    if role:
        flags += ["--preferred-role", role]
        # Only meaningful alongside a role: "not open to other roles" with no role
        # named is not a narrowing, it is an empty statement.
        if str(prefs.get("openToOtherRoles") or "").strip().lower() == "no":
            flags.append("--roles-only")
    arrangement = str(prefs.get("workArrangement") or "").strip()
    if arrangement:
        flags += ["--preferred-arrangement", arrangement]
    return flags


def agent_e_flags(preferences: Optional[dict[str, Any]]) -> list[str]:
    """Course preferences. `free` reorders the tiebreak; it never filters.

    Measured on the live corpus: 0 of 1,999 Coursera courses publish a price and
    only freeCodeCamp's 98 are flagged free, so a hard filter would cut the
    catalogue by 95% and turn most gaps into `no_course_found`.
    """
    prefs = preferences or {}
    if str(prefs.get("coursePricing") or "").strip().lower() == "free":
        return ["--prefer-free"]
    return []


class PipelineRunner:
    """Drives the three agents. Injected so tests can substitute a fake, the same
    way every agent takes a `Deps` — no monkeypatching, no network in tests."""

    def __init__(self, config: Config) -> None:
        self.config = config

    # Each step returns the envelope it produced, or raises. `on_node` is called
    # with each graph node's name as it completes, which is what makes the progress
    # bar move in eleven real steps instead of two.
    def run_agent_a(self, *, cv_paths: list[str], transcript_paths: list[str],
                    run_id: str, on_node: Callable[[str], None]) -> dict[str, Any]:
        from agents.agent_a_cv_extraction.cli import main as agent_a
        from shared.contracts import load_profile

        argv = ["--cv", *cv_paths, "--run-id", run_id, "--no-hitl"]
        if transcript_paths:
            argv += ["--transcript", *transcript_paths]
        if agent_a(argv, on_node=on_node) != 0:
            raise RuntimeError(ERROR_AGENT_A)

        profile_path = Path(self.config.output_dir) / run_id / "candidate_profile.json"
        if not profile_path.exists():
            raise RuntimeError(ERROR_AGENT_A)
        return load_profile(str(profile_path)).model_dump()

    def run_agent_c(self, *, run_id: str, flags: Optional[list[str]] = None,
                    on_node: Optional[Callable[[str], None]] = None) -> dict[str, Any]:
        import json

        from agents.agent_c_gap_analysis.cli import main as agent_c

        profile = Path(self.config.output_dir) / run_id / "candidate_profile.json"
        argv = ["--profile", str(profile), "--run-id", run_id, *(flags or [])]
        if agent_c(argv, on_node=on_node) != 0:
            raise RuntimeError(ERROR_AGENT_C)
        gap = Path(self.config.output_dir) / run_id / "skill_gap.json"
        if not gap.exists():
            raise RuntimeError(ERROR_AGENT_C)
        return json.loads(gap.read_text(encoding="utf-8"))

    def run_agent_e(self, *, run_id: str, flags: Optional[list[str]] = None,
                    on_node: Optional[Callable[[str], None]] = None) -> dict[str, Any]:
        import json

        from agents.agent_e_course_recommend.cli import main as agent_e

        gap = Path(self.config.output_dir) / run_id / "skill_gap.json"
        if agent_e(["--gap", str(gap), "--run-id", run_id, *(flags or [])],
                   on_node=on_node) != 0:
            raise RuntimeError(ERROR_AGENT_E)
        recs = Path(self.config.output_dir) / run_id / "course_recommendations.json"
        if not recs.exists():
            raise RuntimeError(ERROR_AGENT_E)
        return json.loads(recs.read_text(encoding="utf-8"))


def _record_failure(store: Any, job_id: str, exc: BaseException, *, phase: str) -> None:
    """Attribute a failure to the phase it happened in, and always log it.

    The first version keyed only off the exception MESSAGE — a `RuntimeError`
    whose text began "agent_" named the agent, anything else became
    `pipeline_failed`. That looked fine and was measurably wrong: a corrupt upload
    raises `fitz.FileDataError`, which **subclasses RuntimeError**, so a genuinely
    unreadable CV was reported as a generic pipeline failure. The UI's re-upload
    recovery path is gated on `agent_a_unreadable_document`, so the one error a
    user can actually fix was the one they were given no route out of.

    The phase is now the source of truth, because the caller knows it for certain
    and a string never did. And the traceback is printed either way: the old
    `except RuntimeError` branch printed nothing, which is how a broken PDF turned
    into a silent 'pipeline_failed' with no clue in the log.
    """
    traceback.print_exc()
    code = str(exc)
    store.fail_run(job_id, code if code.startswith("agent_") else phase)


def execute_phase_one(store: Any, runner: PipelineRunner, *, job_id: str, run_id: str,
                      cv_paths: list[str], transcript_paths: list[str]) -> None:
    """Agent A, then stop and wait for the user.

    Every failure is caught and recorded: a job that dies silently leaves the UI
    polling a stage that will never change, which is worse than a named error.
    """
    # Eleven checkpoints instead of two, weighted by how long each node actually
    # takes, so the bar advances at a roughly even rate in SECONDS rather than
    # flying through the cheap nodes and freezing on the OCR. The schedule drops
    # the transcript node when no transcript was uploaded — a node that will not
    # run must not hold a slice of the bar.
    checkpoints = progress_policy.agent_a_checkpoints(
        has_transcript=bool(transcript_paths))

    def report(node: str) -> None:
        if (mark := checkpoints.get(node)) is not None:
            store.set_progress(job_id, progress_policy.stage_for(node), mark)

    try:
        profile = runner.run_agent_a(
            cv_paths=cv_paths, transcript_paths=transcript_paths, run_id=run_id,
            on_node=report)
    except Exception as exc:                # noqa: BLE001 - a worker must not die silently
        # Anything thrown while reading the documents IS an Agent A failure,
        # whatever its type: that is what makes "try another file" reachable.
        _record_failure(store, job_id, exc, phase=ERROR_AGENT_A)
        return

    try:
        # Agent A is done: skills judged and coursework-derived skills added.
        store.set_stage(job_id, "translating", 0.55)
        # Profile first, stage second. A poll landing between the two would
        # otherwise see `awaiting_confirmation` with no result and render an empty
        # form over data that exists.
        store.attach_profile(job_id, profile)
        store.set_stage(job_id, STAGE_AWAITING, PROGRESS_AWAITING)
    except Exception as exc:                # noqa: BLE001
        _record_failure(store, job_id, exc, phase=ERROR_UNKNOWN)


def execute_phase_two(store: Any, runner: PipelineRunner, *, job_id: str, run_id: str,
                      preferences: Optional[dict[str, Any]] = None) -> None:
    """Agent C then Agent E, shaped by what the user answered.

    Called when the profile is confirmed, so the gap is computed against the role
    the candidate actually wants and the courses respect their pricing answer.
    """
    config = getattr(runner, "config", None)
    c_marks = progress_policy.agent_c_checkpoints(
        llm_matching=bool(getattr(config, "agent_c_llm_matching", True)))
    e_marks = progress_policy.agent_e_checkpoints()

    def report(marks: dict[str, float]) -> Callable[[str], None]:
        def on_node(node: str) -> None:
            if (mark := marks.get(node)) is not None:
                store.set_progress(job_id, "matching", mark)
        return on_node

    try:
        store.set_stage(job_id, "matching", progress_policy.AGENT_C_SPAN[0])
        gap = runner.run_agent_c(run_id=run_id, flags=agent_c_flags(preferences),
                                 on_node=report(c_marks))
    except Exception as exc:                # noqa: BLE001
        _record_failure(store, job_id, exc, phase=ERROR_AGENT_C)
        return

    try:
        recommendations = runner.run_agent_e(run_id=run_id,
                                             flags=agent_e_flags(preferences),
                                             on_node=report(e_marks))
    except Exception as exc:                # noqa: BLE001
        _record_failure(store, job_id, exc, phase=ERROR_AGENT_E)
        return

    try:
        # profile=None: phase one already stored it, and `finish_run` COALESCEs
        # rather than overwriting.
        store.finish_run(job_id, profile=None, skill_gap=gap,
                         recommendations=recommendations)
    except Exception as exc:                # noqa: BLE001
        _record_failure(store, job_id, exc, phase=ERROR_UNKNOWN)


def _spawn(target: Any, store: Any, runner: PipelineRunner,
           **kwargs: Any) -> threading.Thread:
    """Start a phase in the background and return immediately.

    Daemon so a shutdown is not blocked by an in-flight pipeline; the run is
    recoverable from its `app_runs` row either way.
    """
    thread = threading.Thread(target=target, args=(store, runner), kwargs=kwargs,
                             daemon=True, name=f"analysis-{kwargs.get('job_id')}")
    thread.start()
    return thread


def spawn(store: Any, runner: PipelineRunner, **kwargs: Any) -> threading.Thread:
    return _spawn(execute_phase_one, store, runner, **kwargs)


def spawn_phase_two(store: Any, runner: PipelineRunner, **kwargs: Any) -> threading.Thread:
    return _spawn(execute_phase_two, store, runner, **kwargs)


def stale_runs(store: Any, *, older_than_minutes: int = 30) -> list[dict[str, Any]]:
    """Runs stuck mid-flight, which is what a process restart leaves behind.

    Exposed rather than swept automatically: deciding a run is dead is a judgement
    about the deployment, and a 25-minute backfill-heavy run is not the same as a
    crashed one.

    `awaiting_confirmation` is excluded, and that exclusion is the whole reason
    this list stays useful. A paused run is waiting on a PERSON, so it is expected
    to sit there for as long as someone takes over a form — or forever, if they
    close the tab. Counting those as stuck would bury the real cases (a process
    killed mid-OCR) under every abandoned signup.
    """
    return store._all(                       # noqa: SLF001 - same package, one SQL home
        """
        SELECT job_id, user_id, run_id, stage, started_at
          FROM app_runs
         WHERE stage NOT IN ('done','failed','awaiting_confirmation')
           AND started_at < now() - make_interval(mins => %s)
         ORDER BY started_at
        """,
        (older_than_minutes,))
