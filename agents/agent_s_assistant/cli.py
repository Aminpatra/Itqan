"""Ask Agent S a question from the terminal.

    python main.py agent-s --email you@example.com --ask "how am I doing?"

**A developer and operator tool, not the product.** The product surface is
``POST /api/assistant/messages``, and the difference matters:

* this **does not spend quota**. It reads the same rows and builds the same fact
  sheet, but the daily and weekly counters are claimed in the API layer, so
  running this is free. That is correct for a tool you can only run if you
  already have the database credentials, and it is the reason this command is
  NOT a way around the limits — anyone who can run it could query the tables
  directly anyway;
* this **cannot start a rerun**. It prints the model's suggestion if there is
  one and stops there. Agent S proposes and a person disposes, and a CLI flag
  that spent someone's single weekly credit would quietly undo that.

What it is genuinely for: seeing the fact sheet a real account produces
(``--facts``), and checking an answer against real data without a browser.
"""

from __future__ import annotations

import argparse
from typing import Any, Optional

from shared.config import Config

from .facts import build_fact_sheet
from .graph import build_assistant_graph
from .nodes import Deps


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python main.py agent-s",
        description="Ask about a user's own results. Reads only; spends no quota.",
    )
    who = parser.add_mutually_exclusive_group(required=True)
    who.add_argument("--email", help="The account to answer for")
    who.add_argument("--user-id", help="The account to answer for, by id")

    parser.add_argument("--ask", help="The question. Omit with --facts to just see the record")
    parser.add_argument(
        "--facts", action="store_true",
        help="Print the fact sheet the model would be given, and nothing else. This is "
             "the whole of what it can see, so it is the fastest way to tell a bad "
             "answer from a thin record")
    parser.add_argument(
        "--no-llm", action="store_true",
        help="Skip the model and take the deterministic path, as an outage would")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.ask and not args.facts:
        print("error: give --ask, or --facts to see the record")
        return 2

    config = Config()

    # Imported here rather than at module scope: `api` is the orchestrator and
    # this is an agent, so the dependency is deliberate and worth being able to
    # see. Agents A-E do the same with their stores.
    from api import mapping
    from api.assistant import gaps_from
    from api.db import AppStore

    store = AppStore(config.require_database_url())
    try:
        user = (store.user_by_email(args.email) if args.email
                else store.user_by_id(args.user_id))
        if user is None:
            print(f"no such account: {args.email or args.user_id}")
            return 1

        fact_sheet, has_results = _facts_for(store, mapping, user["user_id"], gaps_from)

        if args.facts:
            print(fact_sheet)
            if not args.ask:
                return 0
            print("\n" + "-" * 70 + "\n")

        llm = None
        if not args.no_llm:
            from agents.agent_s_assistant.schemas import AssistantReply
            from shared.llm import build_llm, structured
            llm = structured(build_llm(config), AssistantReply)

        result = build_assistant_graph(Deps(config=config, llm=llm)).invoke({
            "question": args.ask,
            "fact_sheet": fact_sheet,
            "history": [],
            "has_results": has_results,
        })

        print(result.get("answer") or "")
        if result.get("answer_source") == "template":
            # Said out loud, because a fallback nobody notices is a fallback that
            # has quietly become the norm.
            for warning in result.get("warnings") or []:
                print(f"  [deterministic answer: {warning}]")
        if result.get("proposed_rerun"):
            print(f"\n  [Agent S would offer a rerun: {result.get('rerun_reason')}]")
            print("  [not started — a rerun is only ever spent by the user, "
                  "through POST /api/assistant/rerun]")
        return 0
    finally:
        store.close()


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
            gaps=gaps_from(gap),
            suggested_role=mapping.suggested_role(gap),
            matched_at=finished.isoformat() if hasattr(finished, "isoformat") else finished,
        ),
        True,
    )
