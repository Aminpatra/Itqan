"""Grounding tests — offline, no LLM, no network.

These are the tests that matter most: grounding is the layer that actually stops
hallucinations reaching the output, so it gets adversarial input rather than
happy-path input.
"""

from __future__ import annotations

from agents.agent_a_cv_extraction.grounding import (
    compact_nulls,
    ground_extraction,
    normalize,
    prune_field,
    verify_quote,
)

SOURCE = """Sara Al-Balushi
Email: sara.b@squ.edu.om | Muscat, Oman
BSc Computer Science, Sultan Qaboos University, 2020-2024, GPA 3.71
Skills: Python, PyTorch, SQL
Project: ArabicNER - NER for Arabic news using PyTorch
"""


def test_exact_values_are_grounded():
    result = ground_extraction({"full_name": "Sara Al-Balushi"}, SOURCE)
    entry = result["report"]["full_name"]
    assert entry["grounded"] is True
    assert entry["score"] == 1.0
    assert entry["method"] == "exact"


def test_invented_phone_number_is_dropped():
    result = ground_extraction({"contact": {"phone": "+968 9123 4567"}}, SOURCE)
    assert "contact.phone" in result["dropped"]
    assert result["report"]["contact.phone"]["grounded"] is False


def test_expanded_degree_is_not_grounded():
    """The source says "BSc Computer Science". Expanding it is a hallucination,
    even though every word is individually plausible."""
    result = ground_extraction(
        {"education": [{"degree": "Bachelor of Science in Computer Science"}]}, SOURCE
    )
    assert result["report"]["education[0].degree"]["grounded"] is False


def test_skill_absent_from_source_is_dropped():
    result = ground_extraction({"skills": [{"name": "Kubernetes"}]}, SOURCE)
    assert "skills[0].name" in result["dropped"]


def test_ocr_noise_still_scores_in_the_adjudication_band():
    """A realistic OCR corruption should not be dropped outright — it should be
    escalated to the adjudicator rather than silently discarded."""
    result = ground_extraction(
        {"education": [{"institution": "Sultan Qab0os Universlty"}]}, SOURCE
    )
    entry = result["report"]["education[0].institution"]
    assert 0.75 <= entry["score"] < 0.92, entry
    assert result["needs_llm"], "corrupted-but-real text should reach the adjudicator"


def test_human_supplied_fields_bypass_grounding():
    result = ground_extraction(
        {"contact": {"phone": "+968 9123 4567"}},
        SOURCE,
        skip_fields={"contact.phone"},
    )
    entry = result["report"]["contact.phone"]
    assert entry["grounded"] is True
    assert entry["method"] == "human"


def test_category_labels_are_not_grounded():
    """`category` is a model-assigned label, not a quotation. Grounding it would
    drop every skill whose category word is absent from the CV."""
    result = ground_extraction(
        {"skills": [{"name": "Python", "category": "technical"}]}, SOURCE
    )
    assert "skills[0].category" not in result["report"]


def test_fabricated_evidence_quote_is_rejected():
    assert verify_quote("Sultan Qaboos University", SOURCE) is True
    assert verify_quote("Harvard University", SOURCE) is False
    assert verify_quote(None, SOURCE) is False


def test_prune_field_nulls_nested_values():
    data = {"education": [{"degree": "Invented"}], "contact": {"phone": "123"}}
    assert prune_field(data, "education[0].degree") is True
    assert prune_field(data, "contact.phone") is True
    assert data["education"][0]["degree"] is None
    assert data["contact"]["phone"] is None


def test_prune_field_survives_bad_paths():
    assert prune_field({}, "nope.nothing[4].here") is False


def test_pruning_a_list_item_then_compacting_leaves_clean_output():
    """Regression: pruning nulls list entries in place, so a hallucinated item in
    a string list left ``["Python", None]`` behind. That crashed every consumer
    that joined the list, and would have published a bare null inside a string
    array in the envelope."""
    data = {"projects": [{"name": "X", "technologies": ["Python", "Kubernetes"]}]}
    prune_field(data, "projects[0].technologies[1]")
    assert data["projects"][0]["technologies"] == ["Python", None]

    cleaned = compact_nulls(data)
    assert cleaned["projects"][0]["technologies"] == ["Python"]
    # The consumers' actual failure mode:
    ", ".join(cleaned["projects"][0]["technologies"])


def test_compact_nulls_drops_empty_entries_but_keeps_scalars():
    assert compact_nulls({"a": [None, "x", "", {}]}) == {"a": ["x"]}
    assert compact_nulls({"gpa": None}) == {"gpa": None}  # scalars keep their null


def test_normalize_keeps_email_and_gpa_punctuation():
    assert "sara.b@squ.edu.om" in normalize("Email: Sara.B@SQU.edu.om")
    assert "3.71" in normalize("GPA: 3.71")
