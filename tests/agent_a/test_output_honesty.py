"""What the envelope claims about itself must be true.

Covers the four places Agent A was quietly overstating or discarding: the
confidence number, the gap list, a human's typed answer, and the evidence quote.
"""

from __future__ import annotations

from typing import Any

from agents.agent_a_cv_extraction.nodes.gaps import make_assess_gaps_node, route_after_ingest
from agents.agent_a_cv_extraction.nodes.judge_skills import _clean_evidence_quote
from agents.agent_a_cv_extraction.nodes.persist import make_persist_node
from agents.agent_a_cv_extraction.nodes.validate_human import make_validate_human_node
from agents.agent_a_cv_extraction.schemas import HumanInputValidation, ValidatedField
from shared.config import Config
from shared.contracts import CandidateProfile


def _persist(state_extra: dict[str, Any], tmp_path) -> dict:
    import json
    from pathlib import Path

    base = {
        "run_id": "t", "output_dir": str(tmp_path),
        "cv_doc": {"role": "cv", "parts": [], "text": "x"},
        "cv_extraction": {}, "grounding_report": {},
    }
    base.update(state_extra)
    out = make_persist_node(Config())(base)
    return json.loads(Path(out["final_output_path"]).read_text(encoding="utf-8"))


def _report(**entries) -> dict:
    return {
        path: {"grounded": g, "score": s, "method": m, "value": "v"}
        for path, (g, s, m) in entries.items()
    }


# ---------------------------------------------------------------------------
# confidence
# ---------------------------------------------------------------------------
def test_catching_a_hallucination_no_longer_lowers_confidence(tmp_path):
    """The old formula multiplied by the survival rate, so a profile got LESS
    confident as its unsupported fields were removed."""
    report = _report(
        **{"a": (True, 1.0, "exact"), "b": (True, 1.0, "exact"), "c": (False, 0.1, "dropped")}
    )
    profile = _persist(
        {"grounding_report": report, "dropped_fields": ["c"]}, tmp_path
    )
    conf = profile["confidence"]
    assert conf["overall"] == 1.0, "shipped fields are all perfectly grounded"
    assert conf["extraction_precision"] < 1.0, "but the model did propose a bad field"
    assert conf["fields"] == {"verified": 2, "human": 0, "dropped": 1}


def test_unmeasured_confidence_is_null_not_zero(tmp_path):
    """0.0 conflated 'we could not measure' with 'measured and worthless'."""
    profile = _persist({"grounding_report": {}}, tmp_path)
    assert profile["confidence"]["overall"] is None
    assert profile["confidence"]["extraction_precision"] is None


def test_per_section_ignores_dropped_fields_like_overall_does(tmp_path):
    report = _report(**{
        "contact.email": (True, 1.0, "exact"),
        "contact.phone": (False, 0.2, "dropped"),
    })
    profile = _persist({"grounding_report": report, "dropped_fields": ["contact.phone"]}, tmp_path)
    assert profile["confidence"]["per_section"]["contact"] == 1.0


# ---------------------------------------------------------------------------
# gaps are published
# ---------------------------------------------------------------------------
def test_unresolved_gaps_reach_the_envelope(tmp_path):
    gaps = [{"field_path": "contact.email", "reason": "not found", "ocr_confidence": 0.4}]
    profile = _persist({"unresolved_gaps": gaps}, tmp_path)
    published = profile["provenance"]["unresolved_gaps"]
    assert published == [
        {"field_path": "contact.email", "reason": "not found", "ocr_confidence": 0.4}
    ]
    CandidateProfile.model_validate(profile)   # still a valid envelope


def test_gaps_beyond_the_prompt_cap_are_still_recorded():
    """They were truncated before being stored, so gaps 9+ existed nowhere — while
    the warning told the user they were 'recorded in the output'."""
    extraction = {"candidate": {}, "education": [], "skills": []}
    node = make_assess_gaps_node(Config())
    out = node({"cv_extraction": extraction, "cv_doc": {"text": "x", "blocks": []},
                "grounding_report": {}, "review_rounds": 0})
    assert len(out["gaps"]) <= 8
    assert len(out["unresolved_gaps"]) >= len(out["gaps"])
    for warning in out.get("warnings", []):
        assert "remain recorded in the output" not in warning


# ---------------------------------------------------------------------------
# a fatal document short-circuits instead of burning LLM calls
# ---------------------------------------------------------------------------
def test_an_unreadable_cv_routes_straight_to_persist():
    assert route_after_ingest({"errors": ["No text could be extracted from the CV."]}) == "persist"
    assert route_after_ingest({"cv_doc": {"text": "   "}}) == "persist"


def test_a_readable_cv_continues_to_extraction():
    assert route_after_ingest({"cv_doc": {"text": "Sara Al-Balushi, engineer"}}) == "llm_extract_cv"


# ---------------------------------------------------------------------------
# human answers are never silently destroyed
# ---------------------------------------------------------------------------
class _Validator:
    def __init__(self, fields, boom=False):
        self.fields, self.boom = fields, boom

    def with_structured_output(self, schema, **kw):
        from langchain_core.runnables import RunnableLambda

        def _invoke(_):
            if self.boom:
                raise RuntimeError("validator unavailable")
            return HumanInputValidation(fields=self.fields)

        return RunnableLambda(_invoke)


def _validate(validator) -> dict:
    node = make_validate_human_node(validator, Config())
    return node({
        "human_input": {"contact.email": "sara@x.om", "candidate.full_name": "Sara Al-Balushi"},
        "gaps": [{"field_path": "contact.email", "reason": "r"},
                 {"field_path": "candidate.full_name", "reason": "r"}],
        "cv_extraction": {"candidate": {}}, "review_rounds": 0,
    })


def test_an_answer_the_validator_omits_is_kept_not_lost():
    """The exact failure judge_skills already guards against, unguarded here."""
    out = _validate(_Validator([
        ValidatedField(field_path="contact.email", normalized_value="sara@x.om", accepted=True)
    ]))
    assert "candidate.full_name" in out["human_supplied_fields"]
    assert any("not returned by the validator" in w for w in out["warnings"])


def test_a_validator_outage_does_not_discard_a_round_of_typing():
    out = _validate(_Validator([], boom=True))
    assert set(out["human_supplied_fields"]) == {"contact.email", "candidate.full_name"}
    assert any("accepting it as typed" in w for w in out["warnings"])


# ---------------------------------------------------------------------------
# evidence_quote is a quote, or nothing
# ---------------------------------------------------------------------------
def test_a_paraphrased_evidence_quote_is_nulled():
    source = "Built an inventory tool in Python for the ward."
    cleaned = _clean_evidence_quote({"evidence_quote": "used in a named project"}, source)
    assert cleaned["evidence_quote"] is None


def test_a_real_evidence_quote_survives():
    source = "Built an inventory tool in Python for the ward."
    cleaned = _clean_evidence_quote({"evidence_quote": "Built an inventory tool"}, source)
    assert cleaned["evidence_quote"] == "Built an inventory tool"
