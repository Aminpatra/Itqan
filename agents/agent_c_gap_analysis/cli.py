"""Agent C command-line frontend.

Reads a candidate profile (Agent A's output) and Agent B's tables; writes
``skill_gap.json``. Needs the database and an OpenAI key for embeddings — but no
chat model: this agent makes zero LLM calls by design.
"""

from __future__ import annotations

import argparse
import sys

from shared.config import Config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent-c",
        description=(
            "Candidate-to-job skill gap analysis: match a candidate profile against "
            "live postings, fall back to sector demand stats when retrieval is thin."
        ),
    )
    parser.add_argument("--profile",
                        help="Path to candidate_profile.json (Agent A's output). "
                        "Required unless --watch is given")
    parser.add_argument("--watch", action="store_true",
                        help="Run continuously: whenever Agent A writes a new "
                        "candidate_profile.json under the output dir, analyse it "
                        "automatically and write skill_gap.json beside it")
    parser.add_argument("--poll-seconds", type=float, default=5.0,
                        help="Watch-mode scan interval (default: 5)")
    parser.add_argument("--backfill", action="store_true",
                        help="In watch mode, also process profiles that already "
                        "existed at startup (default: new arrivals only)")
    parser.add_argument("--top-k", type=int, default=15,
                        help="Postings to retrieve (default: 15)")
    parser.add_argument("--sector", default=None,
                        help="ISCO-08 major group '0'-'9' for the stats fallback; "
                        "overrides the modal-sector inference")
    parser.add_argument("--user-id", default=None,
                        help="Identifier written to the output (default: the profile's run_id)")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--run-id", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = Config()

    print("\n  Itqan | Agent C - skill gap analysis\n")

    if args.sector is not None and args.sector not in set("0123456789"):
        print(f"  --sector must be a single digit 0-9, got {args.sector!r}", file=sys.stderr)
        return 2

    try:
        config.require_api_key()        # embeddings only — no chat model
        config.require_database_url()
    except RuntimeError as exc:
        print(f"  {exc}\n", file=sys.stderr)
        return 2

    from shared.embeddings import build_embedder

    from .graph import build_gap_graph
    from .nodes import Deps

    # Built ONCE — in watch mode the same embedder and compiled graph serve
    # every arriving profile.
    deps = Deps(config=config, embedder=build_embedder(config))
    graph = build_gap_graph(deps)

    def analyse(profile_path: str, *, run_id=None, output_dir=None) -> dict:
        state = graph.invoke({
            "profile_path": profile_path,
            "user_id": args.user_id,
            "top_k": args.top_k,
            "sector_override": args.sector,
            "output_dir": output_dir if output_dir is not None else args.output_dir,
            "run_id": run_id if run_id is not None else args.run_id,
        })
        _print_result(state, config)
        return state

    if args.watch:
        from pathlib import Path

        from .watcher import Watcher

        root = Path(args.output_dir or config.output_dir)

        def process(profile: Path) -> None:
            # The gap output lands IN THE PROFILE'S OWN run directory, so its
            # presence marks that run processed and everything about one
            # candidate lives in one folder.
            analyse(str(profile), run_id=profile.parent.name, output_dir=str(root))

        print(f"  Agent C in watch mode — run Agent A in another terminal and the")
        print(f"  gap analysis will follow its profile automatically.")
        Watcher(
            root=root,
            process=process,
            poll_seconds=args.poll_seconds,
            backfill=args.backfill,
        ).run_forever()
        return 0

    if not args.profile:
        print("  --profile is required (or use --watch).", file=sys.stderr)
        return 2

    try:
        analyse(args.profile)
    except (FileNotFoundError, ValueError) as exc:
        print(f"  {exc}\n", file=sys.stderr)
        return 2
    return 0


def _print_result(state: dict, config: Config) -> None:
    # Display-only shaping: dumb terminals cannot connect Arabic letters. The
    # stored JSON keeps pure logical-order Arabic; only these prints reshape.
    from shared.display import arabize

    sims = sorted(
        (round(p.similarity, 3) for p in state.get("postings", [])), reverse=True
    )
    print(f"  retrieved  {len(sims)} postings; similarities: {sims}")
    print(f"  usable     {len(state.get('usable_postings', []))} at threshold "
          f"{config.agent_c_match_threshold} "
          f"{'-> FALLBACK to sector stats' if state.get('used_fallback') else '-> direct matching'}")
    if state.get("used_fallback"):
        print(f"  sector     {state.get('inferred_sector') or 'UNKNOWN (fallback skipped)'}")
    for job in state.get("matched_jobs", []):
        print(f"    gap {job['gap_score']:.2f}  {arabize(job['job_title'][:56])}")
    fallback = state.get("fallback_sector_gap")
    if fallback:
        print(f"    sector gap {fallback['gap_score']:.2f}: "
              f"{len(fallback['matched_skills'])} matched, "
              f"{len(fallback['missing_skills'])} missing, "
              f"{len(fallback['possible_match_skills'])} possible")
    for warning in state.get("warnings", []):
        print(f"  ! {warning}")
    print(f"\n  output -> {state.get('output_path')}\n")


if __name__ == "__main__":
    raise SystemExit(main())
