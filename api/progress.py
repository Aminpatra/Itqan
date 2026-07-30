"""How the progress bar is divided between the agents, and between their nodes.

This is a **policy** file and it lives in `api/` rather than in any agent for that
reason: an agent should not have to know that it occupies 73% of somebody's
progress bar. The agents report which node just finished (via
`shared.graph_progress`); the mapping from a node name to a number belongs here.

Two rules shaped every number below.

**The bar never advances on a clock.** Every value here is reached because a node
actually finished. That was an explicit requirement and it is also the only honest
option: a timer-driven bar reaches 90% and stops, and a hung run becomes
indistinguishable from a slow one. Smoothness therefore comes from having MANY
real checkpoints — 11 in Agent A rather than 2 — not from interpolating between
few.

**The weights are durations, not counts.** Equal steps per node would make the bar
lurch: `extract_text` can be 70 seconds of OCR while `assess_gaps` is a
millisecond of arithmetic, so giving them the same slice produces a bar that flies
through the cheap nodes and freezes on the expensive one. The weights below are
relative time costs, so the bar advances at a roughly even rate in seconds.
"""

from __future__ import annotations

# Agent A owns the largest share because it IS the largest share. Measured end to
# end on this machine: Agent A 16s on a text-layer CV and ~72s on a scanned one,
# against ~19s for Agent C and Agent E together. It also carries the only
# unbounded step (OCR), so under-allocating it is what produced a bar that sat at
# 15% for over a minute.
#
# It stops at 0.75 rather than 1.0 because the run PAUSES there for the user to
# confirm; the remaining quarter is the matching their confirmation starts.
PHASE_ONE_SPAN = (0.02, 0.75)
AGENT_C_SPAN = (0.75, 0.92)
AGENT_E_SPAN = (0.92, 1.00)

# Relative cost of each node on Agent A's no-HITL path (the path the web app
# takes), MEASURED — a real CV plus transcript, 2026-07-30, wall-clock seconds
# between checkpoints:
#
#   ingest + extract_text      3s   (text-layer CV; ~70s when OCR is needed)
#   llm_extract_cv             7s
#   transcript + grounding     6s
#   research_curriculum       21s
#   judge_skills               8s
#   derive_coursework_skills  48s   <- the long pole, and it was weighted at 3
#   summarize + persist        5s
#
# The first pass guessed, and the guess was wrong in a way only a live run shows:
# `derive_coursework_skills` held one position for forty-eight seconds on five
# percent of the bar, while `extract_text` — three seconds on this document — had
# nineteen. Weights are durations, so they have to come from measurement.
#
# `extract_text` still carries a large share despite measuring 3s here, because it
# is bimodal: a scanned CV puts ~70s of OCR in this one node. It is the hedge
# against the slowest case rather than a claim about the fastest.
AGENT_A_WEIGHTS: dict[str, float] = {
    "ingest": 1,                       # open the files
    "extract_text": 8,                 # OCR — 3s to ~70s, by far the most variable
    "llm_extract_cv": 7,               # one large structured call
    "llm_extract_transcript": 4,       # skipped when no transcript was uploaded
    "verify_grounding": 4,             # grounding, plus adjudication calls
    "assess_gaps": 1,
    "research_curriculum": 12,         # one call per credential, so several
    "judge_skills": 6,
    "derive_coursework_skills": 14,    # a call per candidate coursework skill
    "summarize": 4,
    "persist": 1,
}

AGENT_C_WEIGHTS: dict[str, float] = {
    "build_query_embedding": 2,         # one embedding call
    "retrieve_postings": 3,            # pgvector search over the corpus
    "map_candidate_skills": 3,         # ESCO tiers, embeddings for the unmapped
    "resolve_ambiguous_skills": 4,     # the fenced LLM tier; absent when disabled
    "gap_analysis": 2,
    "persist": 1,
}

AGENT_E_WEIGHTS: dict[str, float] = {
    "load_missing_skills": 1,
    "retrieve_candidate_courses": 3,
    "greedy_cover_assign": 1,
    "attach_flags": 1,
    "generate_rationale": 6,           # one LLM call per recommendation
    "persist": 1,
}

# Which stage word each Agent A node belongs to, in the frontend's existing
# vocabulary. The stage is the label; the weights above are the bar.
_READING_NODES = {"ingest", "extract_text"}


def _checkpoints(weights: dict[str, float],
                 span: tuple[float, float]) -> dict[str, float]:
    """Cumulative fraction after each node, scaled into `span`.

    A node's checkpoint is the progress once it has FINISHED, so the last node in
    a phase lands exactly on the top of the span and nothing has to be nudged
    there afterwards.
    """
    lo, hi = span
    total = sum(weights.values()) or 1.0
    out: dict[str, float] = {}
    running = 0.0
    for node, weight in weights.items():
        running += weight
        out[node] = round(lo + (hi - lo) * (running / total), 4)
    return out


def agent_a_checkpoints(*, has_transcript: bool) -> dict[str, float]:
    """Agent A's schedule for THIS run.

    A node that will not run must not be given a slice of the bar, or its absence
    reads as a stall followed by a lurch: with no transcript uploaded,
    `llm_extract_transcript` never fires and the bar would step 28% -> 44% in one
    go. Dropping it and re-spreading the remaining weights keeps the steps even,
    and is not a fudge — the work genuinely is not being done.
    """
    weights = dict(AGENT_A_WEIGHTS)
    if not has_transcript:
        weights.pop("llm_extract_transcript", None)
    return _checkpoints(weights, PHASE_ONE_SPAN)


def agent_c_checkpoints(*, llm_matching: bool) -> dict[str, float]:
    """Same reasoning: the fenced LLM tier is skipped when it is switched off."""
    weights = dict(AGENT_C_WEIGHTS)
    if not llm_matching:
        weights.pop("resolve_ambiguous_skills", None)
    return _checkpoints(weights, AGENT_C_SPAN)


def agent_e_checkpoints(*, rationale: bool = True) -> dict[str, float]:
    """`generate_rationale` still runs with no model — it writes the deterministic
    template — but it costs almost nothing, so its slice shrinks to match."""
    weights = dict(AGENT_E_WEIGHTS)
    if not rationale:
        weights["generate_rationale"] = 1
    return _checkpoints(weights, AGENT_E_SPAN)


def stage_for(node: str) -> str:
    """The stage word to display while a phase-one node runs."""
    return "reading" if node in _READING_NODES else "translating"
