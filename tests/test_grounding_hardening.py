"""The specific ways grounding could be fooled, and the data it used to destroy.

Every case here was measured against the real code before it was fixed, so each
test pins a defect that actually shipped rather than a hypothetical one. The
adversarial half matters most: grounding is the only thing standing between a
model's invention and the published profile.
"""

from __future__ import annotations

from shared.grounding import (
    contains_token_span,
    ground_extraction,
    ground_value,
    normalize,
    verify_quote,
)
from shared.text_norm import fold_digits, is_rtl, normalize_ar

CV = (
    "Sara worked with JavaScript at Bank Muscat. Grade obtained: (C). "
    "GPA 3.10. Email: sara@x.om Skills: SQL, Python."
)
NS = normalize(CV)


# ---------------------------------------------------------------------------
# fabrications that used to score a perfect 1.0 "exact"
# ---------------------------------------------------------------------------
def test_a_substring_of_a_real_word_is_not_grounded():
    """The largest hole in the old check: `needle in haystack`. The CV says only
    'JavaScript', so 'Java' and 'Script' are inventions — both scored 1.0."""
    for fabricated in ("Java", "Script"):
        score, method = ground_value(fabricated, NS)
        assert not (score == 1.0 and method == "exact"), (
            f"{fabricated!r} was accepted as an exact match off a longer word"
        )


def test_a_fabricated_single_letter_grade_is_not_grounded():
    """A falsified grade 'A' matched any text containing the letter a."""
    score, method = ground_value("A", NS)
    assert method == "too_short" and score == 0.0


def test_a_truncated_number_does_not_match_a_longer_one():
    """'3.1' must not ground off a real GPA of '3.10' — digits are exactly where
    a silent mismatch is most costly."""
    score, method = ground_value("3.1", NS)
    assert not (score == 1.0 and method == "exact")


# ---------------------------------------------------------------------------
# ...while every genuine value still grounds
# ---------------------------------------------------------------------------
def test_genuine_values_still_ground_exactly():
    """The boundary rule must not cost real data. These include a value ending a
    sentence ('Bank Muscat.'), an email and a GPA — all of which keep punctuation
    that whitespace-only boundaries would have broken on."""
    for real in ("JavaScript", "Bank Muscat", "SQL", "Python", "C", "3.10", "sara@x.om"):
        assert ground_value(real, NS) == (1.0, "exact"), f"{real!r} regressed"


def test_a_multi_word_value_matches_inside_a_longer_phrase():
    assert contains_token_span("bank muscat", normalize("Bank Muscat Head Office"))


# ---------------------------------------------------------------------------
# short values: adjudicated, never silently deleted
# ---------------------------------------------------------------------------
def test_a_short_value_with_an_ocr_typo_is_adjudicated_not_dropped():
    """'SQI' (OCR misreading SQL) scored a flat 0.0 and was deleted unseen."""
    result = ground_extraction({"skills": [{"name": "SQI"}]}, CV)
    path = "skills[0].name"
    assert result["report"][path]["method"] == "too_short"
    assert [i["field_path"] for i in result["needs_llm"]] == [path]
    assert result["dropped"] == [], "a short value must not be deleted without review"


# ---------------------------------------------------------------------------
# Arabic — the data this project actually serves
# ---------------------------------------------------------------------------
def test_the_same_arabic_name_grounds_regardless_of_hamza_form():
    """أحمد and احمد are the same name. This scored 0.6 — below the adjudication
    floor — so a real candidate's real name was deleted as a hallucination."""
    source = normalize("الاسم: أحمد الحارثي")
    assert ground_value("احمد", source) == (1.0, "exact")


def test_arabic_indic_digits_compare_equal_to_ascii():
    source = normalize("المعدل ٣.٧١")
    assert ground_value("3.71", source) == (1.0, "exact")
    assert fold_digits("۹۸۷") == "987"


def test_taa_marbuta_and_tatweel_variants_fold_together():
    assert normalize_ar("جامعة") == normalize_ar("جامعه")
    assert normalize_ar("مـــحمد") == normalize_ar("محمد")


def test_latin_text_is_untouched_by_the_arabic_folding():
    assert normalize_ar("Sultan Qaboos University 3.71") == "Sultan Qaboos University 3.71"


def test_rtl_detection():
    assert is_rtl("ar") and is_rtl("AR") and is_rtl("fa")
    assert not is_rtl("en") and not is_rtl(None)


# ---------------------------------------------------------------------------
# verify_quote: real AND supporting
# ---------------------------------------------------------------------------
def test_a_real_quote_that_does_not_support_the_value_is_rejected():
    """The self-certification hole: an adjudicator could certify a fabricated
    employer by citing any true sentence in the document."""
    assert verify_quote("Grade obtained: (C)", CV, "Bank Of Fabrication") is False


def test_a_real_quote_that_supports_the_value_is_accepted():
    assert verify_quote("at Bank Muscat", CV, "Bank Muscat") is True


def test_a_fabricated_quote_is_rejected():
    assert verify_quote("Head of Engineering at Google", CV, "Google") is False


def test_the_two_argument_form_keeps_its_old_meaning_for_agent_b():
    """Agent B's legitimacy adjudicator calls this without a value; that call
    must keep checking only that the quote is real."""
    assert verify_quote("Grade obtained: (C)", CV) is True
