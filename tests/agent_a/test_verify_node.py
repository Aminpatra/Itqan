"""The verification node itself — adjudication, spans, and the transcript.

This path had no coverage at all: the shared fake LLM always returned an empty
GroundingReport, so the fabricated-quote backstop, the fail-closed handler and the
pruning boundary were never executed by any test, despite being the code that
decides what reaches the published profile.
"""

from __future__ import annotations

from typing import Any

from agents.agent_a_cv_extraction.nodes.verify import TRANSCRIPT_PREFIX, make_verify_node
from agents.agent_a_cv_extraction.schemas import FieldVerdict, GroundingReport
from shared.config import Config

CV_TEXT = (
    "Sara Al-Balushi is a graduate of Sultan Qaboos University. "
    "She built an inventory tool. Contact: sara@x.om"
)


class VerdictLLM:
    """Returns a canned GroundingReport, so the adjudication branch actually runs."""

    def __init__(self, verdicts: list[FieldVerdict]):
        self.verdicts = verdicts
        self.calls = 0

    def with_structured_output(self, schema: type, **kw: Any):
        from langchain_core.runnables import RunnableLambda

        def _invoke(_payload: Any):
            self.calls += 1
            return GroundingReport(verdicts=self.verdicts)

        return RunnableLambda(_invoke)


class ExplodingLLM:
    def with_structured_output(self, schema: type, **kw: Any):
        from langchain_core.runnables import RunnableLambda

        def _boom(_payload: Any):
            raise RuntimeError("adjudicator unavailable")

        return RunnableLambda(_boom)


def _state(extraction: dict, transcript: dict | None = None) -> dict:
    return {
        "cv_doc": {"text": CV_TEXT, "role": "cv"},
        "transcript_doc": None,
        "cv_extraction": extraction,
        "transcript_extraction": transcript or {},
    }


def _run(llm, extraction, transcript=None, **cfg):
    node = make_verify_node(llm, Config(**cfg))
    return node(_state(extraction, transcript))


# An OCR typo in a real institution name scores 0.92 — exactly the default accept
# threshold. Raising the bar in the adjudication tests puts it in the band so the
# adjudicator actually runs, rather than relying on a value that happens to sit
# between two thresholds a config change could move.
BAND = {"grounded_threshold": 0.95}


def typo() -> dict:
    """A fresh extraction each time — verify prunes IN PLACE, so a shared literal
    would leak one test's pruning into the next."""
    return {"education": [{"institution": "Sultan Qaboos Univrsity"}]}


# ---------------------------------------------------------------------------
# the backstop that stops the model self-certifying
# ---------------------------------------------------------------------------
def _verdict(quote: str) -> VerdictLLM:
    return VerdictLLM([
        FieldVerdict(field_path="education[0].institution",
                     value="Sultan Qaboos Univrsity", grounded=True, evidence_quote=quote)
    ])


def test_a_verdict_whose_quote_is_fabricated_is_refused():
    """'grounded, and here is my evidence' is only worth as much as the evidence."""
    out = _run(_verdict("She completed a PhD at Imperial College London"), typo(), **BAND)
    assert out["cv_extraction"]["education"][0]["institution"] is None
    assert "education[0].institution" in out["dropped_fields"]
    assert any("could not quote" in w for w in out["warnings"])


def test_a_real_quote_that_does_not_support_the_value_is_refused():
    """The quote is genuinely in the CV, but says nothing about this value — the
    self-certification route that checking only 'does the quote exist' left open."""
    out = _run(_verdict("She built an inventory tool"), typo(), **BAND)
    assert out["cv_extraction"]["education"][0]["institution"] is None
    assert any("could not quote" in w for w in out["warnings"])


def test_a_verdict_with_a_supporting_quote_is_accepted():
    out = _run(_verdict("a graduate of Sultan Qaboos University"), typo(), **BAND)
    assert out["cv_extraction"]["education"][0]["institution"] == "Sultan Qaboos Univrsity"
    assert "education[0].institution" not in out.get("dropped_fields", [])


def test_adjudicator_failure_fails_closed():
    out = _run(ExplodingLLM(), typo(), **BAND)
    assert out["cv_extraction"]["education"][0]["institution"] is None
    assert any("adjudication failed" in w for w in out["warnings"])


# ---------------------------------------------------------------------------
# source_span — demanded by the prompt, previously never checked
# ---------------------------------------------------------------------------
def test_a_paraphrased_span_never_costs_a_grounded_skill():
    """THE regression this file exists to prevent. An earlier version dropped any
    skill whose span failed to verify; on a real PDF CV that deleted all 24 of the
    candidate's skills — every one of which had matched the document verbatim
    (score 1.0). A model's inability to quote exactly says nothing about whether
    the candidate has the skill."""
    extraction = {"skills": [
        {"name": "inventory", "source_span": "she built a small inventory system"},
    ]}
    out = _run(VerdictLLM([]), extraction)
    assert out["cv_extraction"]["skills"][0]["name"] == "inventory"
    assert "skills[0].name" not in out.get("dropped_fields", [])


def test_an_unverified_span_is_recorded_as_a_flag():
    """Still worth knowing — it just annotates instead of condemning."""
    extraction = {"skills": [
        {"name": "inventory", "source_span": "she built a small inventory system"},
    ]}
    out = _run(VerdictLLM([]), extraction)
    assert out["grounding_report"]["skills[0].name"]["span_verified"] is False
    assert any("could not be matched verbatim" in w for w in out["warnings"])


def test_a_verbatim_span_verifies():
    extraction = {"skills": [
        {"name": "inventory", "source_span": "She built an inventory tool"},
    ]}
    out = _run(VerdictLLM([]), extraction)
    assert out["grounding_report"]["skills[0].name"]["span_verified"] is True


def test_a_skill_whose_name_is_absent_is_still_dropped():
    """The name grounding remains the real check, unchanged."""
    extraction = {"skills": [{"name": "Kubernetes", "source_span": None}]}
    out = _run(VerdictLLM([]), extraction)
    assert out["cv_extraction"]["skills"][0]["name"] is None


# ---------------------------------------------------------------------------
# the transcript is published, so it is verified
# ---------------------------------------------------------------------------
def test_transcript_fields_are_grounded_and_namespaced():
    """CGPA reached the envelope with no provenance whatsoever."""
    out = _run(
        VerdictLLM([]),
        {"candidate": {"full_name": "Sara Al-Balushi"}},
        transcript={"institution": "Sultan Qaboos University"},
    )
    key = f"{TRANSCRIPT_PREFIX}institution"
    assert key in out["grounding_report"]
    assert out["grounding_report"][key]["grounded"] is True


def test_an_unsupported_transcript_field_is_pruned_from_the_transcript():
    out = _run(
        VerdictLLM([]),
        {"candidate": {"full_name": "Sara Al-Balushi"}},
        transcript={"institution": "University Of Somewhere Else Entirely"},
    )
    assert out["transcript_extraction"]["institution"] is None
