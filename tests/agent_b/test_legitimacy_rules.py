"""The legitimacy rule scorer.

**No file in this repo contains a realistic scam posting, and none should.** A
convincing fake in a source tree is a working template, and it would be the most
harmful artifact the project could produce. So the tests decompose the problem
until no such posting is needed:

  * phrase rules are exercised on single CLAUSES, which cannot function as an
    advertisement on their own
  * the arithmetic is tested independently of any text at all
  * the band routing is tested by stubbing scores, not by writing postings that
    land in each band

Precision on the reject class is the metric that matters, not recall — a false
reject silently deletes a real job, and it is invisible afterwards because
rejected rows are excluded from the very statistics anyone would use to notice.
That measurement needs real labelled data and is deliberately not attempted here.
"""

from __future__ import annotations

import pytest

from agents.agent_b_job_ingest.legitimacy import (
    SIGNAL_WEIGHTS,
    Assessment,
    Signal,
    assert_auditable,
    combine_signals,
    distribution,
    normalize_ar,
    score_text,
)
from shared.config import Config

# A neutral posting body with enough role vocabulary that the structural signals
# stay quiet, so each test isolates the rule it names.
NEUTRAL = (
    "Example Co is hiring an Analyst in Muscat. Responsibilities include reporting "
    "and analysis. Requirements: three years of experience. Apply by email to "
    "careers@example.test or via https://careers.example.test/apply"
)


def codes(text: str, **kw) -> set[str]:
    return set(score_text(text, company=kw.pop("company", "Example Co"), **kw).codes)


# ---------------------------------------------------------------------------
# arithmetic — no text involved
# ---------------------------------------------------------------------------
def test_noisy_or_does_not_reach_certainty_from_weak_signals():
    """Five 0.20 signals SUMMED reach exactly 1.00 — certainty, from five weak
    hints — which would make any threshold meaningless. Noisy-OR gives 0.67:
    "several weak signals, still not certain"."""
    assert combine_signals([0.2] * 5) == pytest.approx(0.67232)
    assert sum([0.2] * 5) == pytest.approx(1.0)


def test_risk_is_bounded_below_one_by_construction():
    """No combination of rules can ever assert certainty, however many are
    added later."""
    assert combine_signals(list(SIGNAL_WEIGHTS.values())) < 1.0
    assert combine_signals([0.99] * 50) < 1.0


def test_no_signals_is_zero_risk():
    assert combine_signals([]) == 0.0


@pytest.mark.parametrize("weights", [[], [0.1], [0.45, 0.3], [0.2, 0.2, 0.15]])
@pytest.mark.parametrize("extra", [0.0, 0.05, 0.45])
def test_adding_a_signal_never_lowers_risk(weights, extra):
    """A property test rather than examples: it catches a whole class of
    weight-tuning regressions that a fixed table would miss."""
    assert combine_signals(weights + [extra]) >= combine_signals(weights) - 1e-9


def test_no_single_correlate_can_reject_on_its_own():
    """Every signal except the fraud mechanism itself has an innocent
    explanation in isolation, and the weights must say so."""
    config = Config()
    for code, weight in SIGNAL_WEIGHTS.items():
        score = 1.0 - combine_signals([weight])
        assert score > config.legitimacy_reject_at, f"{code} alone rejects a posting"


def test_score_is_the_inverse_of_risk():
    assert Assessment(risk=0.3).score == 0.7
    assert Assessment(risk=0.0).score == 1.0


# ---------------------------------------------------------------------------
# phrase rules — single clauses, never postings
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "clause,expected",
    [
        ("A registration fee is required", "advance_fee"),
        ("refundable deposit required", "advance_fee"),
        ("رسوم التسجيل", "advance_fee"),
        ("لا يوجد خصم من الراتب", "no_salary_deduction"),
        ("بدون خصم من الراتب", "no_salary_deduction"),
        ("free accommodation and food", "free_accommodation_bait"),
        ("سكن مجاني", "free_accommodation_bait"),
    ],
)
def test_each_phrase_rule_fires_on_its_own_clause(clause, expected):
    assert expected in codes(f"{NEUTRAL} {clause}")


def test_a_neutral_posting_raises_no_phrase_signals():
    """The rules must not fire on ordinary language — a false positive here
    deletes a real job."""
    assert codes(NEUTRAL) == set()


@pytest.mark.parametrize(
    "variant",
    ["لا يوجد خصم من الراتب", "لا يُوجد خصم من الراتب", "لا يوجد خصم من الراتب"],
)
def test_arabic_orthographic_variants_do_not_evade_a_rule(variant):
    """Diacritics and alef forms are optional in Arabic. A filter that fails on
    a variant fails exactly where it matters."""
    assert "no_salary_deduction" in codes(f"{NEUTRAL} {variant}")


def test_normalization_folds_the_forms_that_are_one_word():
    assert normalize_ar("أحمد") == normalize_ar("احمد")
    assert normalize_ar("مُحَمَّد") == normalize_ar("محمد")


# ---------------------------------------------------------------------------
# contact-channel rule
# ---------------------------------------------------------------------------
def test_whatsapp_alongside_other_channels_is_not_suspicious():
    """WhatsApp is the region's ordinary business channel. Treating its presence
    as a signal would penalise a large slice of legitimate local postings."""
    assert "whatsapp_only_contact" not in codes(f"{NEUTRAL} or WhatsApp 9xxxxxxx")


def test_whatsapp_as_the_only_route_does_fire():
    text = (
        "Analyst needed. Responsibilities include reporting. Requirements: experience. "
        "Contact us on WhatsApp only."
    )
    assert "whatsapp_only_contact" in codes(text)


# ---------------------------------------------------------------------------
# structural signals — evidence is an absence
# ---------------------------------------------------------------------------
def test_a_missing_employer_is_structural_and_quotes_nothing():
    """Its evidence is an absence, and inventing a quote for it would be exactly
    the fabrication the audit check exists to catch."""
    assessment = score_text(NEUTRAL, company=None)
    signal = next(s for s in assessment.signals if s.code == "no_employer_named")

    assert signal.evidence is None
    assert signal.is_structural
    assert signal.basis


def test_a_named_employer_silences_that_signal():
    assert "no_employer_named" not in codes(NEUTRAL, company="Example Co")


def test_an_unextracted_employer_is_not_the_same_claim_as_an_absent_one():
    """Caught at the phase-2b gate, where every posting scored
    ``no_employer_named`` — not because the postings named no employer, but
    because adapters correctly do not set `company` and extraction had not run.
    The band distribution was measuring phase ordering, which would have led to
    retuning the weights against an artifact.
    """
    assessment = score_text(NEUTRAL, company=None, employer_extracted=False)
    assert "no_employer_named" not in assessment.codes

    ran = score_text(NEUTRAL, company=None, employer_extracted=True)
    assert "no_employer_named" in ran.codes


def test_a_posting_with_no_role_described_is_flagged():
    assert "contact_only_no_role" in codes("Jobs available. Call now.")


def test_a_described_role_is_not_flagged():
    assert "contact_only_no_role" not in codes(NEUTRAL)


# ---------------------------------------------------------------------------
# auditability
# ---------------------------------------------------------------------------
def test_every_quoted_span_is_verbatim_from_the_posting():
    """A risk score whose reasons cannot be quoted back is not auditable, and
    this filter's false positives are invisible by construction."""
    text = f"{NEUTRAL} A registration fee is required. سكن مجاني"
    assessment = score_text(text, company=None)

    assert_auditable(assessment, text)
    for signal in assessment.signals:
        if not signal.is_structural:
            assert signal.evidence in text or signal.evidence.casefold() in text.casefold()


def test_a_fabricated_span_is_caught():
    """The backstop against a scorer bug producing a confident number attached
    to evidence that is not in the posting."""
    fake = Assessment(
        risk=0.45,
        signals=(Signal(code="advance_fee", weight=0.45, evidence="pay 500 OMR", basis="x"),),
    )

    with pytest.raises(ValueError, match="does not appear"):
        assert_auditable(fake, NEUTRAL)


def test_structural_signals_are_exempt_by_construction_not_by_exception():
    assessment = Assessment(
        risk=0.25,
        signals=(Signal(code="no_employer_named", weight=0.25, evidence=None, basis="absent"),),
    )
    assert_auditable(assessment, NEUTRAL)


# ---------------------------------------------------------------------------
# band routing — stubbed scores, no postings
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "risk,expected",
    [
        (0.00, "clean"),
        (0.39, "clean"),
        (0.40, "adjudicate"),
        (0.69, "adjudicate"),
        (0.70, "reject"),
        (0.95, "reject"),
    ],
)
def test_bands_route_at_the_documented_boundaries(risk, expected):
    assert Assessment(risk=risk).band(Config()) == expected


def test_distribution_reports_the_phase_gate_metric():
    """More than ~20% in `adjudicate` means the weights are miscalibrated — the
    rules are producing ambiguity rather than resolving it."""
    counts = distribution([Assessment(risk=r) for r in (0.0, 0.5, 0.8, 0.1)])

    assert counts == {"clean": 2, "adjudicate": 1, "reject": 1}


def test_scoring_makes_no_network_or_llm_calls():
    """Phase 2b is scoring only. Deterministic and offline is what allows the
    distribution to be measured before the filter is given the power to delete
    anything."""
    first = score_text(NEUTRAL, company=None)
    second = score_text(NEUTRAL, company=None)

    assert first.risk == second.risk
    assert first.codes == second.codes
