"""Itqan entry point.

    python main.py agent-a --cv resume.pdf --transcript transcript.pdf

Agents register as subcommands. Adding Agent B means adding one entry to
``AGENTS`` below and a ``main(argv)`` in that agent's ``cli`` module.
"""

from __future__ import annotations

import sys
import warnings

# Before any third-party import: langgraph pulls in `requests`, which warns about
# an urllib3/chardet version skew in this environment on every single run. The
# filter has to be installed here — by the time shared.config runs, requests is
# already imported and the warning has already fired.
warnings.filterwarnings("ignore", message=r".*urllib3.*chardet.*")

# CVs contain non-ASCII names, and the Windows console defaults to cp1252 — which
# raises UnicodeEncodeError mid-report rather than degrading. Ask for UTF-8, and
# fall back to replacement characters if the stream will not take it.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

AGENTS = {
    "agent-a": (
        "agents.agent_a_cv_extraction.cli",
        "Extract a verified candidate profile from a CV and transcript",
    ),
    "agent-b": (
        "agents.agent_b_job_ingest.cli",
        "Ingest public job postings into the job-demand tables (12h cycle)",
    ),
    "agent-c": (
        "agents.agent_c_gap_analysis.cli",
        "Compute a candidate's skill gap against live postings and demand stats",
    ),
    "agent-d": (
        "agents.agent_d_course_ingest.cli",
        "Ingest online courses into the course-supply tables (3-day cycle)",
    ),
    "agent-e": (
        "agents.agent_e_course_recommend.cli",
        "Recommend one course per missing skill from a candidate's skill gap",
    ),
    "agent-s": (
        "agents.agent_s_assistant.cli",
        "Ask about a user's own results (reads what A/C/E produced; spends no quota)",
    ),
    "status": (
        "agents.status",
        "One read-only view of the whole system: corpus, aggregation freshness, "
        "ESCO coverage, source health",
    ),
    "pipeline": (
        "agents.pipeline",
        "CV in, course recommendations out: Agent A -> C -> E in one run "
        "(C reads Agent B's tables, E reads Agent D's courses)",
    ),
}


def usage() -> int:
    print("usage: python main.py <agent> [options]\n")
    print("agents:")
    for name, (_, description) in AGENTS.items():
        print(f"  {name:<12} {description}")
    print("\nrun `python main.py <agent> --help` for that agent's options")
    return 2


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in {"-h", "--help", "help"}:
        return usage()

    name = argv[0]
    if name not in AGENTS:
        print(f"error: unknown agent {name!r}\n")
        return usage()

    module_path = AGENTS[name][0]
    module = __import__(module_path, fromlist=["main"])
    return module.main(argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
