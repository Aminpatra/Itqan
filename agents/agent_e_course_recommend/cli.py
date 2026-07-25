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

from shared.config import Config


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
    parser.add_argument("--no-rationale", action="store_true",
                        help="Skip the LLM step: run selection only, leave rationales empty "
                        "(no API key needed)")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--run-id", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
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
        state = graph.invoke({
            "gap_path": args.gap,
            "user_id": args.user_id,
            "output_dir": args.output_dir,
            "run_id": args.run_id,
        })
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
        extra = f"  (+{len(c['covers_other_skills'])} more skill(s))" if c["covers_other_skills"] else ""
        print(f"    [{r['priority_bucket']:>8}] {arabize(r['skill'][:34]):<34} -> "
              f"{arabize((c['title'] or '')[:44])}{extra}")
    for r in missing:
        print(f"    [{r['priority_bucket']:>8}] {arabize(r['skill'][:34]):<34} -> (no course found)")
    for warning in state.get("warnings", []):
        print(f"  ! {warning}")
    print(f"\n  output -> {state.get('output_path')}\n")


if __name__ == "__main__":
    raise SystemExit(main())
