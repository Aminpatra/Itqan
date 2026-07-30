"""Agent E command-line frontend.

Reads Agent C's ``skill_gap.json`` and Agent D's course tables; writes
``course_recommendations.json``. Needs the database (to retrieve courses) and an
OpenAI key (the rationale is the single LLM step). ``--no-rationale`` runs the
whole deterministic pipeline and writes empty rationales — useful for inspecting
selection without spending tokens or needing a key.
"""

from __future__ import annotations

import argparse
import sys
from typing import Callable, Optional

from shared.config import Config
from shared.graph_progress import run_reporting


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent-e",
        description=(
            "Recommend one course per missing skill: deterministic coverage-first "
            "selection over Agent D's courses, then a short grounded rationale each."
        ),
    )
    parser.add_argument("--gap", required=True,
                        help="Path to skill_gap.json (Agent C's output)")
    parser.add_argument("--user-id", default=None,
                        help="Identifier written to the output (default: the gap file's user_id)")
    parser.add_argument("--no-rationale", "--no-llm", dest="no_rationale", action="store_true",
                        help="Skip the LLM step. Rationales are still written, built "
                        "deterministically from the same facts, so the agent is fully "
                        "offline and reproducible (no API key needed)")
    parser.add_argument("--prefer-free", action="store_true",
                        help="The candidate asked for free courses. Moves free-ness to "
                        "the front of the tie-break; it never EXCLUDES a paid course, "
                        "because a gap whose only course is paid still needs an answer")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--run-id", default=None)
    return parser


def main(argv: list[str] | None = None, *,
         on_node: Optional[Callable[[str], None]] = None) -> int:
    """`on_node` reports each finished graph node; see Agent A's CLI."""
    args = build_parser().parse_args(argv)
    config = Config()

    print("\n  Itqan | Agent E - course recommendation\n")

    try:
        config.require_database_url()
        if not args.no_rationale:
            config.require_api_key()        # the rationale step only
    except RuntimeError as exc:
        print(f"  {exc}\n", file=sys.stderr)
        return 2

    from .graph import build_recommend_graph
    from .nodes import Deps

    llm = None
    if not args.no_rationale:
        from shared.llm import build_llm
        llm = build_llm(config)

    deps = Deps(config=config, llm=llm)
    graph = build_recommend_graph(deps)

    try:
        state = run_reporting(graph, {
            "gap_path": args.gap,
            "user_id": args.user_id,
            "prefer_free": args.prefer_free,
            "output_dir": args.output_dir,
            "run_id": args.run_id,
        }, on_node=on_node)
    except (FileNotFoundError, ValueError) as exc:
        print(f"  {exc}\n", file=sys.stderr)
        return 2

    _print_result(state)
    return 0


def _print_result(state: dict) -> None:
    from shared.display import arabize

    recs = state.get("recommendations", [])
    found = [r for r in recs if not r["no_course_found"]]
    missing = [r for r in recs if r["no_course_found"]]
    print(f"  {len(found)} course(s) recommended, {len(missing)} skill(s) with no course found\n")
    for r in found:
        c = r["course"]
        extra = f"  (+{len(c['covers_other_skills'])} more)" if c["covers_other_skills"] else ""
        sel = r.get("selection") or {}
        # A pick nothing distinguished is marked, not dressed up as a ranking.
        mark = f"  ~{sel.get('equivalent_candidates', 0)} equal" \
            if sel.get("basis") == "arbitrary" else ""
        print(f"    [{r['priority_bucket']:>8}] {arabize(r['skill'][:30]):<30} -> "
              f"{arabize((c['title'] or '')[:40])}{extra}{mark}")
    for r in missing:
        print(f"    [{r['priority_bucket']:>8}] {arabize(r['skill'][:30]):<30} -> (no course found)")

    cal = state.get("run_calibration") or {}
    arbitrary = cal.get("recommendations_by_arbitrary_pick") or []
    if arbitrary:
        print(f"\n  {len(arbitrary)} of {len(found)} pick(s) had nothing to choose between "
              f"candidates on — no rating, price or date. Those are representative,")
        print(f"  not ranked: {', '.join(arbitrary[:5])}")
    sources = cal.get("rationale_sources") or {}
    if sources.get("template"):
        print(f"  {sources['template']} rationale(s) came from the deterministic template.")

    for warning in state.get("warnings", []):
        print(f"  ! {warning}")
    print(f"\n  output -> {state.get('output_path')}\n")


if __name__ == "__main__":
    raise SystemExit(main())
