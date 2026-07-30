"""How the progress bar is divided, and the properties it must never violate.

The bar is the only thing a user has to judge whether the system is working, so
the invariants here are about trust rather than arithmetic: it must not go
backwards, it must not reach the end before the work does, and it must not claim
a position because a clock moved.
"""

from __future__ import annotations

import statistics

import pytest

from api import progress as p


def test_agent_a_owns_the_largest_share():
    """It is the longest phase by a wide margin — measured 16s on a text-layer CV
    and ~72s on a scanned one, against ~19s for Agents C and E together — and it
    carries the only unbounded step. Under-allocating it is what left the bar at
    15% for over a minute."""
    a_share = p.PHASE_ONE_SPAN[1] - p.PHASE_ONE_SPAN[0]
    c_share = p.AGENT_C_SPAN[1] - p.AGENT_C_SPAN[0]
    e_share = p.AGENT_E_SPAN[1] - p.AGENT_E_SPAN[0]
    assert a_share > c_share + e_share
    assert a_share == pytest.approx(0.73)


def test_the_spans_tile_the_bar_without_gap_or_overlap():
    """A gap is a jump the user sees; an overlap is the bar going backwards."""
    assert p.PHASE_ONE_SPAN[1] == p.AGENT_C_SPAN[0]
    assert p.AGENT_C_SPAN[1] == p.AGENT_E_SPAN[0]
    assert p.AGENT_E_SPAN[1] == 1.0


@pytest.mark.parametrize("marks", [
    p.agent_a_checkpoints(has_transcript=True),
    p.agent_a_checkpoints(has_transcript=False),
    p.agent_c_checkpoints(llm_matching=True),
    p.agent_c_checkpoints(llm_matching=False),
    p.agent_e_checkpoints(),
])
def test_every_schedule_is_strictly_increasing(marks):
    values = list(marks.values())
    assert values == sorted(values)
    assert len(set(values)) == len(values), "two nodes cannot share a checkpoint"


def test_a_phase_ends_exactly_on_its_span():
    """The last node lands on the top of the span, so nothing has to be nudged
    there afterwards — a nudge is a jump with no work behind it."""
    for marks, span in ((p.agent_a_checkpoints(has_transcript=True), p.PHASE_ONE_SPAN),
                        (p.agent_c_checkpoints(llm_matching=True), p.AGENT_C_SPAN),
                        (p.agent_e_checkpoints(), p.AGENT_E_SPAN)):
        assert list(marks.values())[-1] == pytest.approx(span[1], abs=1e-4)


def test_no_single_step_is_a_big_jump():
    """The complaint that started this: 0.15 -> 0.55 -> 0.80. Eleven weighted
    checkpoints keep every step small, and the largest is the OCR one — which is
    genuinely the longest piece of work, so it has earned it."""
    for has_transcript in (True, False):
        marks = p.agent_a_checkpoints(has_transcript=has_transcript)
        steps = [b - a for a, b in zip([p.PHASE_ONE_SPAN[0], *marks.values()],
                                       marks.values())]
        # Two properties, and the median is the one the user feels: most steps are
        # small, and the worst case is bounded. Two nodes ARE genuinely long — the
        # OCR pass and the coursework-skill judging, measured at 48s — so a ceiling
        # tight enough to exclude them would mean lying about their weight.
        assert len(steps) >= 10, "smoothness comes from many real checkpoints"
        assert max(steps) < 0.18, f"largest step {max(steps):.3f} is too coarse"
        assert statistics.median(steps) < 0.08, "the typical step must be small"


def test_a_skipped_node_does_not_leave_a_hole():
    """With no transcript uploaded `llm_extract_transcript` never fires, so holding
    a slice of the bar for it would read as a stall followed by a lurch."""
    with_t = p.agent_a_checkpoints(has_transcript=True)
    without = p.agent_a_checkpoints(has_transcript=False)
    assert "llm_extract_transcript" in with_t
    assert "llm_extract_transcript" not in without
    # Still finishes exactly at the top of the span, spread over fewer nodes.
    assert list(without.values())[-1] == pytest.approx(p.PHASE_ONE_SPAN[1], abs=1e-4)


def test_weights_are_durations_not_counts():
    """Equal slices per node would make the bar fly through the arithmetic and
    freeze on the OCR. `extract_text` must therefore outweigh `assess_gaps` by a
    lot."""
    assert p.AGENT_A_WEIGHTS["extract_text"] >= 5 * p.AGENT_A_WEIGHTS["assess_gaps"]


def test_reading_becomes_translating_when_the_documents_are_read():
    assert p.stage_for("ingest") == "reading"
    assert p.stage_for("extract_text") == "reading"
    assert p.stage_for("llm_extract_cv") == "translating"
    assert p.stage_for("judge_skills") == "translating"


def test_an_unknown_node_moves_nothing():
    """A node this policy has never heard of — a new one, or the HITL nodes the web
    app does not use — must not put the bar at an invented position. The previous
    checkpoint stands, which is the truth: nothing known has completed."""
    marks = p.agent_a_checkpoints(has_transcript=True)
    assert marks.get("human_review") is None
    assert marks.get("some_future_node") is None
