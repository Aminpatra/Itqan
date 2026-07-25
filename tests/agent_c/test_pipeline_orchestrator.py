"""The A->C->E pipeline orchestrator: argv plumbing and the shared run_id.

The orchestrator's whole correctness is that every stage receives the SAME
run_id, so the profile, the gap analysis, and the recommendations land in one
directory — everything else is flag pass-through.
"""

from __future__ import annotations

from pathlib import Path

from agents.pipeline import _agent_a_argv, _agent_c_argv, _agent_e_argv, build_parser


def parse(argv):
    return build_parser().parse_args(argv)


def test_both_stages_share_the_run_id():
    args = parse(["--cv", "cv.pdf"])
    a = _agent_a_argv(args, "RUN123")
    c = _agent_c_argv(args, "RUN123", Path("output/RUN123/candidate_profile.json"))

    assert a[a.index("--run-id") + 1] == "RUN123"
    assert c[c.index("--run-id") + 1] == "RUN123"
    assert c[c.index("--profile") + 1].endswith("candidate_profile.json")


def test_agent_a_flags_pass_through_and_agent_c_flags_do_not_leak_into_a():
    args = parse(["--cv", "p1.png", "p2.png", "--transcript", "t.pdf",
                  "--no-hitl", "--fake-llm", "--ocr-lang", "ar",
                  "--sector", "2", "--top-k", "9", "--user-id", "u1"])

    a = _agent_a_argv(args, "R")
    assert a[:3] == ["--cv", "p1.png", "p2.png"]
    assert "--transcript" in a and "--no-hitl" in a and "--fake-llm" in a
    assert a[a.index("--ocr-lang") + 1] == "ar"
    assert "--sector" not in a and "--top-k" not in a, "Agent C flags leaked into Agent A"

    c = _agent_c_argv(args, "R", Path("p.json"))
    assert c[c.index("--sector") + 1] == "2"
    assert c[c.index("--top-k") + 1] == "9"
    assert c[c.index("--user-id") + 1] == "u1"
    assert "--fake-llm" not in c, (
        "Agent C has no fake mode — it always uses real embeddings — so the "
        "flag must not be forwarded"
    )


def test_minimal_invocation_builds_minimal_argvs():
    args = parse(["--cv", "cv.pdf"])
    a = _agent_a_argv(args, "R")
    c = _agent_c_argv(args, "R", Path("p.json"))
    e = _agent_e_argv(args, "R", Path("output/R/skill_gap.json"))

    assert "--transcript" not in a and "--no-hitl" not in a
    assert "--sector" not in c and "--user-id" not in c
    assert "--user-id" not in e and "--no-rationale" not in e


def test_agent_e_stage_shares_the_run_id_and_reads_the_gap():
    args = parse(["--cv", "cv.pdf", "--user-id", "u1"])
    e = _agent_e_argv(args, "RUN123", Path("output/RUN123/skill_gap.json"))

    assert e[e.index("--run-id") + 1] == "RUN123"
    assert e[e.index("--gap") + 1].endswith("skill_gap.json")
    assert e[e.index("--user-id") + 1] == "u1"
    # Agent A/C flags must not leak into Agent E
    assert "--top-k" not in e and "--sector" not in e and "--cv" not in e


def test_no_rationale_flag_forwards_only_to_agent_e():
    args = parse(["--cv", "cv.pdf", "--no-rationale"])
    a = _agent_a_argv(args, "R")
    c = _agent_c_argv(args, "R", Path("p.json"))
    e = _agent_e_argv(args, "R", Path("g.json"))

    assert "--no-rationale" in e
    assert "--no-rationale" not in a and "--no-rationale" not in c
