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
    "pipeline": (
        "agents.pipeline",
        "CV in, skill gap out: Agent A then Agent C in one run (C reads Agent B's tables)",
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
