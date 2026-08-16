"""The interactive session: how it ends, and what it must not do.

The tests that matter here are the two ways a person leaves. Ctrl+C at the
prompt is the DOCUMENTED way to stop this tool, so a traceback there would be
the tool shouting at someone for using it correctly — and `KeyboardInterrupt`
escaping `input()` is exactly how that happens by default.

Everything runs against fakes; no database, no model, no network.
"""

from __future__ import annotations

import builtins
from typing import Any

import pytest

from agents.agent_s_assistant import cli
from shared.config import Config


class FakeStore:
    """Only the methods the session touches."""

    def __init__(self) -> None:
        self.claims: list[tuple] = []
        self.used = 0

    def claim_quota(self, user_id, *, kind, limit, period_start):
        self.claims.append((user_id, kind, limit))
        if self.used >= limit:
            return None
        self.used += 1
        return self.used

    def quota_used(self, user_id, *, kind, period_start):
        return self.used


def quota_state(store, config, user_id, kind):
    limit = 10 if kind == "message" else 1
    return {"used": store.used, "limit": limit,
            "remaining": max(0, limit - store.used),
            "resetsAt": "2026-08-16T00:00:00+04:00"}


def day_start(config):
    from datetime import date
    return date(2026, 8, 15)


def session(inputs: list[Any], *, answers=None, answer=None, enforce_quota=False,
            store=None, **over):
    """Drive the loop with a scripted stdin. An element that is an exception
    class is RAISED instead of returned, which is how Ctrl+C is simulated.

    A caller-supplied `answer` is WRAPPED rather than substituted — recording
    every question the loop actually asks is the whole point of the harness, and
    an earlier version let a custom answer silently bypass it, which made a
    passing behaviour look like a failure.
    """
    replies = list(answers or [])
    seen: list[str] = []

    def fake_input(_prompt=""):
        if not inputs:
            raise EOFError
        nxt = inputs.pop(0)
        if isinstance(nxt, type) and issubclass(nxt, BaseException):
            raise nxt
        return nxt

    def record(question, history):
        seen.append(question)
        if answer is not None:
            return answer(question, history)
        return replies.pop(0) if replies else {"answer": "ok.", "answer_source": "model"}

    original, builtins.input = builtins.input, fake_input
    try:
        kwargs = dict(
            answer=record, style=cli._Style(False), store=store or FakeStore(),
            config=Config(), user={"full_name": "Maryam", "user_id": "u_1"},
            fact_sheet="Readiness score: 42/100", has_results=True,
            quota_state=quota_state, day_start=day_start,
            enforce_quota=enforce_quota, offline=False,
        )
        kwargs.update(over)
        code = cli._session(**kwargs)
    finally:
        builtins.input = original
    return code, seen


# ---------------------------------------------------------------------------
# how it ends — the point of this change
# ---------------------------------------------------------------------------
def test_ctrl_c_at_the_prompt_leaves_cleanly():
    """THE test. Ctrl+C is the documented way to stop this tool, so it must exit
    0 with a goodbye — not a traceback, which is what an uncaught
    KeyboardInterrupt out of `input()` produces."""
    code, _ = session([KeyboardInterrupt])

    assert code == 0


def test_ctrl_d_leaves_cleanly_too():
    """A piped or closed stdin ends the session the same way. Without this the
    loop spins on EOF forever."""
    code, _ = session([])

    assert code == 0


@pytest.mark.parametrize("word", ["/quit", "/exit", "/q", "/QUIT"])
def test_the_quit_commands_all_work(word):
    code, asked = session([word, "never reached"])

    assert code == 0
    assert asked == [], "input after /quit was still sent to the model"


def test_it_keeps_going_until_told_to_stop():
    """The actual request: run until *I* stop it. Three questions, one session."""
    code, asked = session(["first", "second", "third", KeyboardInterrupt])

    assert code == 0
    assert asked == ["first", "second", "third"]


def test_ctrl_c_during_an_answer_cancels_that_answer_not_the_session():
    """Losing a whole session because one question was slow is a poor trade."""
    def answer(question, history):
        if question == "slow":
            raise KeyboardInterrupt
        return {"answer": "ok.", "answer_source": "model"}

    code, asked = session(["slow", "after", KeyboardInterrupt], answer=answer)

    assert code == 0
    assert "after" in asked, "the session ended instead of returning to the prompt"


# ---------------------------------------------------------------------------
# what makes it usable
# ---------------------------------------------------------------------------
def test_follow_ups_can_see_the_turns_before_them():
    """Without history, "what about the second one?" is unanswerable — which is
    most of what makes a session better than repeated one-shot calls."""
    captured: list[list] = []

    def answer(question, history):
        captured.append(list(history))
        return {"answer": f"re: {question}", "answer_source": "model"}

    session(["one", "two", KeyboardInterrupt], answer=answer)

    assert captured[0] == []
    assert [t["content"] for t in captured[1]] == ["one", "re: one"]


def test_clear_forgets_the_conversation():
    captured: list[list] = []

    def answer(question, history):
        captured.append(list(history))
        return {"answer": "ok.", "answer_source": "model"}

    session(["one", "/clear", "two", KeyboardInterrupt], answer=answer)

    assert captured[-1] == [], "/clear left history behind"


def test_blank_input_is_ignored_rather_than_asked():
    _, asked = session(["", "   ", "real", KeyboardInterrupt])

    assert asked == ["real"]


def test_commands_are_never_sent_to_the_model():
    _, asked = session(["/help", "/facts", "/quota", "/nonsense", KeyboardInterrupt])

    assert asked == [], "a slash command was answered as a question"


def test_an_unknown_command_does_not_end_the_session():
    _, asked = session(["/nope", "still here", KeyboardInterrupt])

    assert asked == ["still here"]


# ---------------------------------------------------------------------------
# quota: unmetered by default, faithful on request
# ---------------------------------------------------------------------------
def test_a_session_spends_nothing_by_default():
    """The CLI is a developer tool. Debugging must not consume a real person's
    ten daily messages."""
    store = FakeStore()

    session(["one", "two", KeyboardInterrupt], store=store)

    assert store.claims == [], "the default session claimed quota"


def test_enforce_quota_meters_the_session_exactly_as_the_api_does():
    store = FakeStore()

    session(["one", "two", KeyboardInterrupt], store=store, enforce_quota=True)

    assert [c[1] for c in store.claims] == ["message", "message"]


def test_hitting_the_limit_under_enforcement_stops_answering_but_not_the_session():
    store = FakeStore()
    store.used = 10                      # already at the cap

    code, asked = session(["one", "two", KeyboardInterrupt],
                          store=store, enforce_quota=True)

    assert code == 0
    assert asked == [], "a question was answered past the limit"


# ---------------------------------------------------------------------------
# the CLI still cannot act
# ---------------------------------------------------------------------------
class RerunStore(FakeStore):
    """Adds the reads and writes `/rerun` touches."""

    def __init__(self, *, awaiting=None, docs=None, latest=None) -> None:
        super().__init__()
        self._awaiting, self._docs, self._latest = awaiting, docs or [], latest
        self.spawned: list[str] = []

    def latest_completed_run(self, user_id):
        return self._latest

    def awaiting_run(self, user_id):
        return self._awaiting

    def all_documents(self, user_id):
        return self._docs

    def profile(self, user_id):
        return {"payload": {"preferences": {}}}

    def create_run(self, *, user_id, run_id, document_ids):
        self.spawned.append(run_id)
        return "job_1"

    def refund_quota(self, user_id, *, kind, period_start):
        self.used = max(0, self.used - 1)


CV = {"kind": "cv", "document_id": "d1", "stored_path": "/tmp/cv.pdf"}
DONE_RUN = {"run_id": "run_old", "profile": {}}


def _rerun_session(inputs, *, store, monkeypatch, spawned=None):
    """Drive /rerun with the real code path, stubbing only the thread spawn."""
    calls = spawned if spawned is not None else []
    monkeypatch.setattr("api.jobs.spawn",
                        lambda *a, **k: calls.append("full"))
    monkeypatch.setattr("api.jobs.spawn_phase_two",
                        lambda *a, **k: calls.append("match"))
    return session(inputs, store=store, runner=object()), calls


def test_slash_rerun_starts_a_match_rerun_after_an_explicit_yes(monkeypatch):
    """A command the user TYPED is the person acting — the same category as a
    button they clicked. That is the propose/dispose split working, not an
    exception to it."""
    store = RerunStore(latest=DONE_RUN)

    (_, _), calls = _rerun_session(["/rerun", "yes", KeyboardInterrupt],
                                   store=store, monkeypatch=monkeypatch)

    assert calls == ["match"]
    assert store.used == 1, "the credit was not claimed"


def test_slash_rerun_full_runs_agent_a(monkeypatch):
    store = RerunStore(latest=DONE_RUN, docs=[CV])

    (_, _), calls = _rerun_session(["/rerun full", "yes", KeyboardInterrupt],
                                   store=store, monkeypatch=monkeypatch)

    assert calls == ["full"]


def test_anything_but_yes_starts_nothing_and_costs_nothing(monkeypatch):
    store = RerunStore(latest=DONE_RUN)

    (_, _), calls = _rerun_session(["/rerun", "y", KeyboardInterrupt],
                                   store=store, monkeypatch=monkeypatch)

    assert calls == []
    assert store.used == 0, "a credit was spent without an explicit yes"


def test_a_full_rerun_is_refused_without_a_cv(monkeypatch):
    store = RerunStore(latest=DONE_RUN, docs=[])

    (_, _), calls = _rerun_session(["/rerun full", "yes", KeyboardInterrupt],
                                   store=store, monkeypatch=monkeypatch)

    assert calls == []
    assert store.used == 0, "a credit was spent on a run that could not start"


def test_an_abandoned_paused_run_does_not_block_a_full_rerun(monkeypatch):
    """Found live: `awaiting_confirmation` runs never expire, so abandoned
    onboarding attempts accumulate. Refusing on their existence locked out any
    account that had ever left one behind — the dev account had four."""
    store = RerunStore(latest=DONE_RUN, docs=[CV], awaiting={"job_id": "old"})

    (_, _), calls = _rerun_session(["/rerun full", "yes", KeyboardInterrupt],
                                   store=store, monkeypatch=monkeypatch)

    assert calls == ["full"]


def test_the_full_rerun_says_it_is_waiting_not_finished(monkeypatch, capsys):
    """"Started" and "finished" are very different things here, and the
    difference is the entire reason the pause exists."""
    store = RerunStore(latest=DONE_RUN, docs=[CV])

    _rerun_session(["/rerun full", "yes", KeyboardInterrupt],
                   store=store, monkeypatch=monkeypatch)

    # Collapsed, because `_say` wraps to the terminal width and the sentence
    # under test spans a line break.
    out = " ".join(capsys.readouterr().out.split())
    assert "STOPS and waits for you to confirm" in out
    assert "matching will not run until you do" in out
    assert "finished" not in out


def test_asking_in_prose_still_starts_nothing(monkeypatch):
    """The line that must not move. A model may PROPOSE a rerun; only a typed
    command starts one, so a persuasive turn or an injected string cannot spend
    somebody's single weekly credit."""
    store = RerunStore(latest=DONE_RUN, docs=[CV])

    (_, _), calls = _rerun_session(
        ["do it", "run it now", "yes please rerun agent A", KeyboardInterrupt],
        store=store, monkeypatch=monkeypatch)

    assert calls == []
    assert store.used == 0


def test_a_proposed_rerun_is_shown_but_never_started(capsys):
    """The session is a friendlier surface, not a more powerful one."""
    code, _ = session(
        ["any new jobs?", KeyboardInterrupt],
        answers=[{"answer": "Maybe.", "answer_source": "model",
                  "proposed_rerun": True, "rerun_reason": "your match is old"}])

    out = capsys.readouterr().out
    assert code == 0
    assert "not started" in out
    assert "your match is old" in out


def test_a_deterministic_answer_says_so(capsys):
    """A fallback nobody notices is a fallback that has quietly become the norm."""
    session(["hi", KeyboardInterrupt],
            answers=[{"answer": "I could not answer that.",
                      "answer_source": "template",
                      "warnings": ["model call failed: TimeoutError"]}])

    assert "deterministic answer" in capsys.readouterr().out


def test_output_wraps_rather_than_running_off_the_screen(capsys):
    session(["hi", KeyboardInterrupt],
            answers=[{"answer": "word " * 200, "answer_source": "model"}])

    lines = capsys.readouterr().out.splitlines()
    assert max(len(line) for line in lines) <= 120
