"""Command-line frontend.

All console I/O lives here rather than in the graph. The interrupt node replays
on resume, so a ``print`` inside it would fire twice; keeping rendering out here
also means a Streamlit or web frontend could drive the same graph unchanged.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from langgraph.types import Command

from shared.artifacts import new_run_id
from shared.config import PROJECT_ROOT, Config

from .graph import build_graph
from .nodes.human_review import SKIP_SENTINEL
from .state import Gap

BAR = "-" * 68


def _render_gap(index: int, total: int, gap: Gap) -> None:
    print(f"\n  [{index}/{total}] {gap['field_path']}")
    print(f"      why: {gap.get('reason', '')}")
    if gap.get("current_value"):
        print(f"      current: {gap['current_value']!r}")
    if gap.get("ocr_confidence") is not None:
        print(f"      OCR confidence: {gap['ocr_confidence']:.2f}")


def collect_answers(payload: dict[str, Any]) -> dict[str, str]:
    """Show the gaps and collect what the user chooses to fill in."""
    gaps: list[Gap] = payload.get("gaps", [])
    print(f"\n{BAR}")
    print(f"  MISSING OR UNCERTAIN - round {payload.get('round', 1)}")
    print(f"  {len(gaps)} item(s) need your input. Press Enter to skip any of them.")
    print(BAR)

    answers: dict[str, str] = {}
    for i, gap in enumerate(gaps, start=1):
        _render_gap(i, len(gaps), gap)
        try:
            value = input(f"      {gap.get('prompt_hint', 'value')}\n      > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  (input closed - skipping the rest)")
            break
        if value:
            answers[gap["field_path"]] = value
    return answers


def print_report(state: dict[str, Any]) -> None:
    print(f"\n{BAR}")
    print("  RESULT")
    print(BAR)

    summary = state.get("summary") or {}
    if summary.get("headline"):
        print(f"\n  {summary['headline']}")
    if summary.get("profile"):
        print(f"\n  {summary['profile']}")

    accepted = state.get("accepted_skills") or []
    rejected = state.get("rejected_skills") or []
    if accepted:
        print(f"\n  Skills kept ({len(accepted)}):")
        for skill in accepted:
            print(
                f"    + {skill['name']:<28} {skill.get('quality', '?'):<7}"
                f" {skill.get('evidence_type', '?')}"
            )
    if rejected:
        print(f"\n  Skills filtered out ({len(rejected)}):")
        for skill in rejected:
            print(f"    - {skill['name']:<28} {skill.get('rationale', '')}")

    if summary.get("gaps_or_unknowns"):
        print("\n  Gaps and unknowns:")
        for item in summary["gaps_or_unknowns"]:
            print(f"    - {item}")

    for note in state.get("human_validation_notes") or []:
        print(f"\n  [manual entry] {note}")
    for warning in state.get("warnings") or []:
        print(f"\n  [warning] {warning}")
    for error in state.get("errors") or []:
        print(f"\n  [error] {error}")

    if state.get("final_output_path"):
        print(f"\n  Profile written to:\n    {state['final_output_path']}")
    print()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent-a",
        description="Extract a structured, verified candidate profile from a CV and transcript.",
    )
    parser.add_argument(
        "--cv",
        required=True,
        nargs="+",
        metavar="PATH",
        help=(
            "CV file(s), PDF or image. Pass several for a multi-page document "
            "photographed a page at a time — they are read in the order given, "
            "so list them in page order."
        ),
    )
    parser.add_argument(
        "--transcript",
        nargs="+",
        metavar="PATH",
        help="Transcript file(s), PDF or image. Several are joined in the order given.",
    )
    parser.add_argument("--output-dir", help="Where run artifacts go (default: ./output)")
    parser.add_argument("--model", help="OpenAI chat model (default: gpt-5.4-mini)")
    parser.add_argument(
        "--ocr-lang",
        help=(
            "PaddleOCR language for scanned documents (default: en). "
            "Use e.g. ar, fr, es, ch, japan, korean, devanagari for CVs not in English "
            "— the default will misread them."
        ),
    )
    parser.add_argument(
        "--no-hitl",
        action="store_true",
        help="Never prompt; record gaps and continue. Use for scripted runs.",
    )
    parser.add_argument("--fake-llm", action="store_true", help="Offline canned LLM, for testing")
    parser.add_argument("--run-id", help="Reuse a specific run id (advanced)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    cv_paths = [Path(p).expanduser() for p in args.cv]
    transcript_paths = [Path(p).expanduser() for p in (args.transcript or [])]

    # Check every file up front. Failing on page three after OCR-ing pages one
    # and two wastes real time on a scanned document.
    missing = [p for p in cv_paths + transcript_paths if not p.exists()]
    if missing:
        for path in missing:
            print(f"error: file not found: {path}", file=sys.stderr)
        return 2

    config = Config()
    if args.model:
        config.model = args.model
    if args.ocr_lang:
        config.ocr_lang = args.ocr_lang
    output_dir = Path(args.output_dir).expanduser() if args.output_dir else config.output_dir

    if args.fake_llm:
        from tests.fake_llm import FakeStructuredLLM

        llm: Any = FakeStructuredLLM()
    else:
        from shared.llm import build_llm

        try:
            llm = build_llm(config)
        except RuntimeError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2

    app = build_graph(llm, config)

    # A fresh thread_id per run. Reusing one resumes the previous run's checkpoint
    # instead of starting clean — the most common way to get baffling results here.
    run_id = args.run_id or new_run_id()
    graph_config = {"configurable": {"thread_id": run_id}}

    initial = {
        "cv_paths": [str(p) for p in cv_paths],
        "transcript_paths": [str(p) for p in transcript_paths],
        "run_id": run_id,
        "output_dir": str(output_dir),
        "interactive": not args.no_hitl,
        "review_rounds": 0,
    }

    def _describe(paths: list[Path]) -> str:
        names = ", ".join(p.name for p in paths)
        return f"{names} ({len(paths)} files)" if len(paths) > 1 else names

    print("\n  Itqan | Agent A - CV extraction")
    print(f"  run {run_id}")
    print(f"  cv: {_describe(cv_paths)}")
    if transcript_paths:
        print(f"  transcript: {_describe(transcript_paths)}")

    state = app.invoke(initial, config=graph_config)

    while state.get("__interrupt__"):
        payload = state["__interrupt__"][0].value
        answers = collect_answers(payload)
        # Never resume with an empty dict — see SKIP_SENTINEL in nodes/human_review.
        state = app.invoke(
            Command(resume=answers or SKIP_SENTINEL), config=graph_config
        )

    print_report(state)
    return 1 if state.get("errors") else 0


if __name__ == "__main__":
    sys.path.insert(0, str(PROJECT_ROOT))
    raise SystemExit(main())
