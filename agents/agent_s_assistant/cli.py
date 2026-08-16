"""Talk to Agent S from the terminal.

    python main.py agent-s --email you@example.com          # a session; Ctrl+C to leave
    python main.py agent-s --email you@example.com --ask "how am I doing?"

Opens a conversation and keeps it open until you stop it. Follow-ups work,
because the session remembers the turns before them.

**A developer and operator tool, not the product.** The product surface is
``POST /api/assistant/messages``, and three differences matter:

* it **does not spend quota** by default. The counters are claimed in the API
  layer, so a session here is unmetered. That is correct for a tool you can only
  run if you already hold the database credentials — anyone able to run it could
  query the tables directly — but it means what you see is not what a real user
  gets. The banner says so, ``/quota`` shows their true remaining allowance, and
  ``--enforce-quota`` makes a session behave exactly as production does;
* it **cannot start a rerun**. It prints the suggestion and stops. Agent S
  proposes and a person disposes, and a CLI flag that spent someone's single
  weekly credit would quietly undo that;
* it **does not write to their conversation history**. Turns live in memory for
  the session only, so testing never injects rows into a real user's transcript.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import textwrap
import uuid
from typing import Any, Optional

from shared.config import Config
from shared.display import arabize

from .facts import build_fact_sheet
from .graph import build_assistant_graph
from .nodes import Deps

BANNER_WIDTH = 72


# ---------------------------------------------------------------------------
# presentation
# ---------------------------------------------------------------------------
def _colour_enabled() -> bool:
    """Colour only for a real terminal. Piped into a file or a pipeline, escape
    codes are noise that breaks grep."""
    return sys.stdout.isatty()


class _Style:
    def __init__(self, enabled: bool) -> None:
        self.on = enabled

    def _wrap(self, code: str, text: str) -> str:
        return f"\033[{code}m{text}\033[0m" if self.on else text

    def dim(self, t: str) -> str:    return self._wrap("2", t)
    def bold(self, t: str) -> str:   return self._wrap("1", t)
    def cyan(self, t: str) -> str:   return self._wrap("36", t)
    def yellow(self, t: str) -> str: return self._wrap("33", t)


def _width() -> int:
    # Clamped: a maximised terminal produces 200-character lines that are
    # genuinely harder to read than 80.
    return max(48, min(shutil.get_terminal_size((80, 24)).columns - 2, 88))


def _say(text: str, *, indent: str = "  ") -> None:
    """Print wrapped, Arabic-shaped output.

    `arabize` is display-only and never touches anything stored — see its module
    docstring. Without it an Arabic answer renders as disconnected letters in
    reverse on the Windows console and in VS Code's terminal.
    """
    width = _width() - len(indent)
    for para in (text or "").split("\n"):
        if not para.strip():
            print()
            continue
        for line in textwrap.wrap(para, width=width) or [""]:
            print(f"{indent}{arabize(line)}")


# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python main.py agent-s",
        description="Talk to Agent S about one account's own results. "
                    "Reads only; starts nothing.",
        epilog="With no --ask, this opens a session and stays open until Ctrl+C.",
    )
    who = parser.add_mutually_exclusive_group(required=True)
    who.add_argument("--email", help="The account to answer for")
    who.add_argument("--user-id", help="The account to answer for, by id")

    parser.add_argument(
        "--ask", help="Ask one question and exit, instead of opening a session")
    parser.add_argument(
        "--facts", action="store_true",
        help="Print the fact sheet the model would be given, and exit. This is the "
             "whole of what it can see, so it is the fastest way to tell a bad "
             "answer from a thin record")
    parser.add_argument(
        "--no-llm", action="store_true",
        help="Skip the model and take the deterministic path, as an outage would")
    parser.add_argument(
        "--enforce-quota", action="store_true",
        help="Claim real quota per message, exactly as the API does. Off by "
             "default so debugging is free; on, this session is metered and a "
             "real user's allowance is consumed")
    return parser


HELP = """\
Ask anything about this account's results — readiness, which jobs matched and
why, what skills are missing, which courses were suggested.

  /rerun         re-match against today's job corpus (Agents C and E)
  /rerun full    re-read the documents too (Agent A), then stop for confirmation
  /facts         show the record Agent S is answering from
  /quota         this account's real remaining allowance
  /clear         forget this session's conversation
  /help          this message
  /quit          leave (Ctrl+C works too)

A rerun spends a real weekly credit and asks you to type 'yes' first. Note that
asking Agent S to run one in prose will not work, and that is deliberate: the
model may suggest a rerun, but only a command you type starts one."""


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    config = Config()
    style = _Style(_colour_enabled())

    # Imported here rather than at module scope: `api` is the orchestrator and
    # this is an agent, so the dependency is deliberate and worth being able to
    # see. Agents A-E do the same with their stores.
    from api import mapping
    from api.assistant import day_start, gaps_from, quota_state
    from api.db import AppStore

    store = AppStore(config.require_database_url())
    try:
        user = (store.user_by_email(args.email) if args.email
                else store.user_by_id(args.user_id))
        if user is None:
            print(f"No account matches {args.email or args.user_id!r}.")
            return 1

        fact_sheet, has_results = _facts_for(store, mapping, user["user_id"], gaps_from)

        if args.facts and not args.ask:
            print(fact_sheet)
            return 0

        llm = None
        if not args.no_llm:
            from agents.agent_s_assistant.schemas import AssistantReply
            from shared.llm import build_llm, structured
            llm = structured(build_llm(config), AssistantReply)
        graph = build_assistant_graph(Deps(config=config, llm=llm))

        def answer(question: str, history: list[dict[str, Any]]) -> dict[str, Any]:
            return graph.invoke({
                "question": question, "fact_sheet": fact_sheet,
                "history": history, "has_results": has_results,
            })

        # ---- one-shot ---------------------------------------------------
        if args.ask:
            if args.facts:
                print(fact_sheet)
                print("\n" + "-" * _width() + "\n")
            _render(answer(args.ask, []), style)
            return 0

        # ---- a session --------------------------------------------------
        # The runner is built lazily and only when a rerun is actually started:
        # constructing it imports the agent graphs, and a session that only ever
        # asks questions should not pay for machinery it never uses.
        def runner() -> Any:
            from api.jobs import PipelineRunner
            return PipelineRunner(config)

        return _session(
            answer=answer, style=style, store=store, config=config, user=user,
            fact_sheet=fact_sheet, has_results=has_results,
            quota_state=quota_state, day_start=day_start,
            enforce_quota=args.enforce_quota, offline=args.no_llm,
            runner=runner,
        )
    finally:
        store.close()


def _session(*, answer, style: _Style, store: Any, config: Config, user: dict,
             fact_sheet: str, has_results: bool, quota_state, day_start,
             enforce_quota: bool, offline: bool, runner=None) -> int:
    """The conversation loop. Ends on Ctrl+C, Ctrl+D, or /quit — never on a
    traceback: an interrupted session should look like leaving a room, not like
    a crash."""
    name = user.get("full_name") or user.get("email")
    rule = style.dim("─" * _width())

    print()
    print(f" {style.bold('Agent S')} {style.dim('· itqan')}")
    print(rule)
    print(f" Answering for {style.cyan(str(name))}")
    if not has_results:
        print(style.yellow(" No completed match yet — there is little to discuss "
                           "until one finishes."))
    if offline:
        print(style.yellow(" Model off (--no-llm): answers are the deterministic "
                           "fallback."))
    # Said plainly and every time. A developer who forgets this session is
    # unmetered will draw the wrong conclusion about what a real user experiences.
    print(style.dim(" Metered like production." if enforce_quota
                    else " Unmetered — a real user gets 10 messages a day. "
                         "/quota shows theirs."))
    print(style.dim(" /help for commands · Ctrl+C to leave"))
    print(rule)

    # In memory only, deliberately: a developer poking at Agent S must not write
    # turns into a real person's stored conversation.
    history: list[dict[str, Any]] = []
    turns = 0

    while True:
        try:
            raw = input(f"\n{style.cyan('you ›')} ").strip()
        except (KeyboardInterrupt, EOFError):
            # Ctrl+C at the prompt is how people leave. A traceback here would be
            # the tool shouting at someone for using it correctly.
            print(f"\n{style.dim(f'Goodbye. {turns} question(s) this session.')}\n")
            return 0

        if not raw:
            continue

        low = raw.lower()
        if low in ("/quit", "/exit", "/q"):
            print(f"{style.dim(f'Goodbye. {turns} question(s) this session.')}\n")
            return 0
        if low == "/help":
            _say(HELP, indent=" ")
            continue
        if low == "/facts":
            print()
            _say(fact_sheet, indent=" ")
            continue
        if low == "/clear":
            history.clear()
            print(style.dim("  (conversation forgotten)"))
            continue
        if low == "/quota":
            _print_quota(store, config, user["user_id"], quota_state, style)
            continue
        if low.startswith("/rerun"):
            _rerun(store, config, user, style,
                   full=low.split()[1:2] == ["full"],
                   quota_state=quota_state,
                   runner=runner() if callable(runner) else runner)
            continue
        if low.startswith("/"):
            print(style.yellow(f"  Unknown command {raw.split()[0]!r} — /help lists them."))
            continue

        if enforce_quota:
            used = store.claim_quota(user["user_id"], kind="message",
                                     limit=config.assistant_daily_messages,
                                     period_start=day_start(config))
            if used is None:
                limit = config.assistant_daily_messages
                print(style.yellow(
                    f"  Daily limit reached ({limit}/{limit}). This is what a real "
                    f"user would see. Run without --enforce-quota to keep testing."))
                continue

        try:
            print(style.dim("  thinking…"), end="\r", flush=True)
            result = answer(raw, list(history))
            print(" " * 20, end="\r")          # erase the indicator in place
        except KeyboardInterrupt:
            # Ctrl+C DURING a call abandons that answer and returns to the prompt.
            # Losing a session because one question was slow would be a poor trade.
            print(" " * 20, end="\r")
            print(style.dim("  (cancelled)"))
            continue

        turns += 1
        text = result.get("answer") or ""
        history.append({"role": "user", "content": raw})
        history.append({"role": "assistant", "content": text})
        print()
        _render(result, style)

    # unreachable


def _rerun(store: Any, config: Config, user: dict, style: _Style, *, full: bool,
           quota_state, runner: Any) -> None:
    """Start a rerun, because a PERSON typed a command.

    This is the propose/dispose split working as intended rather than an
    exception to it. What must never happen is the MODEL deciding a rerun should
    happen and it happening — a persuasive turn, or an injected string, spending
    someone's single weekly credit. A slash command the user typed is the same
    category as a button they clicked: it is the person, not the model.

    Asking Agent S to "do it" in prose still does nothing, and that stays true.
    """
    from api.assistant import week_start
    from api.jobs import spawn, spawn_phase_two

    quota = quota_state(store, config, user["user_id"], "rerun")
    if quota["remaining"] <= 0:
        print(style.yellow(f"  No rerun credits left. They reset "
                           f"{quota['resetsAt'][:16].replace('T', ' ')}."))
        return

    latest = store.latest_completed_run(user["user_id"])
    docs = store.all_documents(user["user_id"]) if full else []

    # Refused BEFORE the credit is claimed, every time.
    if full:
        # Deliberately NOT blocked by an existing paused run — see the note in
        # `api/assistant.py`. Abandoned onboarding attempts never expire, so that
        # check locked out any account that had ever left one behind.
        if not any(d["kind"] == "cv" for d in docs):
            print(style.yellow("  No CV on this account — Agent A cannot run without one."))
            return
    elif latest is None:
        print(style.yellow("  No completed run to re-match from yet."))
        return

    what = ("re-read the documents with Agent A, then STOP and wait for you to "
            "confirm the details in the app before matching runs"
            if full else
            "re-match against today's job corpus (Agents C and E), running to the end")
    _say(f"This will spend {quota['remaining']} of {quota['limit']} weekly credits and {what}.",
         indent="  ")
    try:
        if input(f"  {style.yellow('type yes to continue ›')} ").strip().lower() != "yes":
            print(style.dim("  (nothing started)"))
            return
    except (KeyboardInterrupt, EOFError):
        print(style.dim("\n  (nothing started)"))
        return

    used = store.claim_quota(user["user_id"], kind="rerun",
                             limit=config.assistant_weekly_reruns,
                             period_start=week_start(config))
    if used is None:
        print(style.yellow("  No rerun credits left."))
        return

    try:
        if full:
            run_id = f"run_{uuid.uuid4().hex[:12]}"
            cvs = [d for d in docs if d["kind"] == "cv"][:1]
            transcripts = [d for d in docs if d["kind"] == "transcript"][:1]
            job_id = store.create_run(
                user_id=user["user_id"], run_id=run_id,
                document_ids=[d["document_id"] for d in cvs + transcripts])
            spawn(store, runner, job_id=job_id, run_id=run_id,
                  cv_paths=[d["stored_path"] for d in cvs],
                  transcript_paths=[d["stored_path"] for d in transcripts])
        else:
            profile = store.profile(user["user_id"]) or {}
            preferences = (profile.get("payload") or {}).get("preferences") or {}
            job_id = store.create_run(user_id=user["user_id"], document_ids=[],
                                      run_id=latest["run_id"])
            spawn_phase_two(store, runner, job_id=job_id, run_id=latest["run_id"],
                            preferences=preferences, profile=latest.get("profile"))
    except Exception as exc:                      # noqa: BLE001
        store.refund_quota(user["user_id"], kind="rerun",
                           period_start=week_start(config))
        print(style.yellow(f"  Could not start it: {type(exc).__name__}: {exc}"))
        print(style.dim("  (credit refunded)"))
        return

    print(style.dim(f"  started · job {job_id}"))
    if full:
        # Said plainly, because "started" and "finished" are very different
        # things here and the difference is the whole reason the pause exists.
        _say("Agent A is re-reading the documents now. When it finishes, the run "
             "STOPS and waits for you to confirm the details in the app — matching "
             "will not run until you do.", indent="  ")
    else:
        _say("Matching is running now and will finish on its own. Ask me again in "
             "a minute and I will be answering from the new results.", indent="  ")
    print(style.dim("  (this session's facts are from the previous run until it lands)"))


def _render(result: dict[str, Any], style: _Style) -> None:
    _say(result.get("answer") or "")

    if result.get("answer_source") == "template":
        # Announced, because a fallback nobody notices is a fallback that has
        # quietly become the norm — the same reason the API records answer_source.
        for warning in result.get("warnings") or []:
            print(style.dim(f"  · deterministic answer: {warning}"))

    if result.get("proposed_rerun"):
        print()
        _say(style.yellow(f"Agent S would offer to re-match you: "
                          f"{result.get('rerun_reason')}"))
        print(style.dim("  · not started. A rerun costs 1 of 1 weekly credits and "
                        "is only ever spent by the user, in the app."))


def _print_quota(store: Any, config: Config, user_id: str, quota_state,
                 style: _Style) -> None:
    messages = quota_state(store, config, user_id, "message")
    reruns = quota_state(store, config, user_id, "rerun")
    print(style.dim(
        f"  messages {messages['remaining']}/{messages['limit']} left "
        f"(resets {messages['resetsAt'][:16].replace('T', ' ')})"))
    print(style.dim(
        f"  reruns   {reruns['remaining']}/{reruns['limit']} left "
        f"(resets {reruns['resetsAt'][:16].replace('T', ' ')})"))


def _facts_for(store: Any, mapping: Any, user_id: str,
               gaps_from: Any = None) -> tuple[str, bool]:
    """The same assembly `api/assistant._facts` does.

    Deliberately duplicated rather than imported: that function closes over a
    FastAPI app, and reaching into it from a CLI would couple this command to the
    web layer's construction. The shape being identical is what matters, and
    `build_fact_sheet` is the single place that decides it.
    """
    row = store.latest_completed_run(user_id)
    if row is None:
        return (build_fact_sheet(readiness=None, jobs=[], courses=[], gaps=[],
                                 suggested_role=None, matched_at=None), False)

    gap = row["skill_gap"] or {}
    recs = row["recommendations"] or {}
    board = mapping.dashboard(row["profile"] or {}, gap, recs)
    finished = row.get("finished_at")
    return (
        build_fact_sheet(
            readiness=board.get("readiness"),
            jobs=mapping.job_matches(gap, limit=10),
            courses=mapping.courses(recs)[:10],
            gaps=gaps_from(gap) if gaps_from else list(
                (gap.get("aggregate", {}) or {}).get("most_common_missing_skills", []))[:10],
            suggested_role=mapping.suggested_role(gap),
            matched_at=finished.isoformat() if hasattr(finished, "isoformat") else finished,
        ),
        True,
    )
