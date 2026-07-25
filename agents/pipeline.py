"""The end-to-end pipeline: CV in, course recommendations out, one command.

    python main.py pipeline --cv cv.pdf --transcript t.pdf

Runs Agent A (extraction + verification, interactive HITL included), then Agent
C on the profile it produced (which queries Agent B's live demand tables), then
Agent E on the gap it produced (which reads Agent D's course supply). One run
directory holds everything: the profile, the OCR artifacts, skill_gap.json, and
course_recommendations.json. ``--no-recommend`` stops after the gap (the old A->C
behaviour).

This module is an ORCHESTRATOR, not an agent: it lives beside the agent
packages and drives them through their public CLIs, exactly as ``main.py``
does. The agents themselves remain fully decoupled — none imports another, and
each still runs standalone. The join is a shared ``run_id``: the pipeline mints
one and hands the same one to every stage, so each writes into the same folder.

How the four connect, concretely:

    Agent A ──candidate_profile.json──▶ Agent C ──skill_gap.json──▶ Agent E
       (this pipeline / --watch)          ▲                            ▲
                          job_postings + skill_demand_stats   courses / supply
                                   (Agent B, live)             (Agent D, live)
"""

from __future__ import annotations

import argparse
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from shared.config import Config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pipeline",
        description="Run Agent A on a CV, then Agent C on the resulting profile "
        "(reads Agent B's tables), then Agent E on the gap (reads Agent D's courses).",
    )
    # ---- Agent A side ----
    parser.add_argument("--cv", nargs="+", required=True,
                        help="CV file(s), PDF or image, in page order")
    parser.add_argument("--transcript", nargs="+", default=None)
    parser.add_argument("--no-hitl", action="store_true",
                        help="Never prompt; gaps are recorded instead")
    parser.add_argument("--fake-llm", action="store_true",
                        help="Offline extraction (Agent A only — Agent C always "
                        "uses real embeddings)")
    parser.add_argument("--ocr-lang", default=None)
    parser.add_argument("--model", default=None)
    # ---- Agent C side ----
    parser.add_argument("--sector", default=None,
                        help="ISCO-08 sector for the stats fallback")
    parser.add_argument("--top-k", type=int, default=15)
    parser.add_argument("--user-id", default=None)
    # ---- Agent E side ----
    parser.add_argument("--no-recommend", action="store_true",
                        help="Stop after the skill gap; do not run Agent E "
                        "(the original A -> C behaviour)")
    parser.add_argument("--no-rationale", action="store_true",
                        help="Run Agent E selection only, without the LLM rationale step")
    return parser


def _agent_a_argv(args: argparse.Namespace, run_id: str) -> list[str]:
    argv = ["--cv", *args.cv, "--run-id", run_id]
    if args.transcript:
        argv += ["--transcript", *args.transcript]
    if args.no_hitl:
        argv.append("--no-hitl")
    if args.fake_llm:
        argv.append("--fake-llm")
    if args.ocr_lang:
        argv += ["--ocr-lang", args.ocr_lang]
    if args.model:
        argv += ["--model", args.model]
    return argv


def _agent_c_argv(args: argparse.Namespace, run_id: str, profile: Path) -> list[str]:
    argv = ["--profile", str(profile), "--run-id", run_id, "--top-k", str(args.top_k)]
    if args.sector:
        argv += ["--sector", args.sector]
    if args.user_id:
        argv += ["--user-id", args.user_id]
    return argv


def _agent_e_argv(args: argparse.Namespace, run_id: str, gap: Path) -> list[str]:
    # No --output-dir: Agent E defaults to config.output_dir / run_id, the same
    # folder Agent A and C wrote into.
    argv = ["--gap", str(gap), "--run-id", run_id]
    if args.user_id:
        argv += ["--user-id", args.user_id]
    if args.no_rationale:
        argv.append("--no-rationale")
    return argv


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = Config()

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:8]

    flow = "A -> C" if args.no_recommend else "A -> C -> E"
    print(f"\n  Itqan | pipeline ({flow})  run {run_id}")
    print("  " + "=" * 66)

    # ---- stage 1: Agent A -------------------------------------------------
    from agents.agent_a_cv_extraction.cli import main as agent_a_main

    code = agent_a_main(_agent_a_argv(args, run_id))
    if code != 0:
        print(f"\n  Agent A exited {code}; stopping before the gap analysis.",
              file=sys.stderr)
        return code

    profile = Path(config.output_dir) / run_id / "candidate_profile.json"
    if not profile.exists():
        print(f"\n  Agent A reported success but {profile} is missing; stopping.",
              file=sys.stderr)
        return 2

    # ---- stage 2: Agent C (reads Agent B's live tables) -------------------
    print("  " + "=" * 66)
    from agents.agent_c_gap_analysis.cli import main as agent_c_main

    code = agent_c_main(_agent_c_argv(args, run_id, profile))
    if code != 0:
        print(f"\n  Agent C exited {code}; stopping before recommendations.",
              file=sys.stderr)
        return code
    if args.no_recommend:
        return 0

    gap = Path(config.output_dir) / run_id / "skill_gap.json"
    if not gap.exists():
        print(f"\n  Agent C reported success but {gap} is missing; stopping.",
              file=sys.stderr)
        return 2

    # ---- stage 3: Agent E (reads Agent D's course supply) ----------------
    print("  " + "=" * 66)
    from agents.agent_e_course_recommend.cli import main as agent_e_main

    return agent_e_main(_agent_e_argv(args, run_id, gap))


if __name__ == "__main__":
    raise SystemExit(main())
