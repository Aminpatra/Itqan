"""The whole-token matcher shared by the gap side and the course side.

Promoted from Agent C when `shared.course_market` needed the same relation. The
tests that matter are the ones about what must NOT match: 84% of course skills
never reach an ESCO concept, so this rule carries most of Agent E's retrieval,
and a rule that is too loose silently recommends the wrong course.
"""

from __future__ import annotations

import pytest

from shared.skill_match import contains_tokens, covers_skill, tokens


def test_tokens_keep_the_characters_that_are_part_of_skill_names():
    assert tokens("C++") == ["c++"]
    assert tokens("Node.js / React") == ["node.js", "react"]
    assert tokens("C#") == ["c#"]


def test_whole_token_matching_never_matches_inside_a_word():
    """The Agent A grounding bug in another costume: 'Java' must not match inside
    'JavaScript'."""
    assert contains_tokens("javascript", "java") is False
    assert contains_tokens("java programming", "java") is True


@pytest.mark.parametrize("taught,required", [
    ("python programming", "python"),          # the course is more specific
    ("python", "python programming"),          # the gap is more specific
    ("data analytics engineering", "data analytics"),
    ("Machine Learning", "machine learning"),  # case only
])
def test_a_course_covers_a_gap_it_contains(taught, required):
    assert covers_skill(taught, required) is True


@pytest.mark.parametrize("taught,required", [
    ("javascript", "java"),
    ("data analysis", "data"),                 # single short generic token
    ("python", "ruby"),
    ("c++", "c#"),
    ("machine learning", ""),
])
def test_unrelated_or_too_generic_pairs_do_not_match(taught, required):
    assert covers_skill(taught, required) is False


def test_a_bare_field_word_only_matches_exactly():
    """Without the guard, a gap in 'data' pulls in every data-anything course —
    exactly the loose-match harm Agent C's audit warned about for unmapped
    skills. It still matches its own exact self."""
    assert covers_skill("data science", "data") is False
    assert covers_skill("data", "data") is True


def test_length_is_not_what_makes_a_word_generic():
    """'java' and 'data' are both four characters and only one names a skill, so
    the guard is an explicit list rather than a length threshold."""
    assert covers_skill("java programming", "java") is True
    assert covers_skill("data analysis", "data") is False


def test_containment_is_measured_against_real_pairs_not_invented_ones():
    """Every pair below was produced by running the matcher over the real
    2026-07-28 corpus (4,064 taught skills) against a real Agent C gap file,
    rather than chosen to make the rule look good.

    The last group is the reason containment only ever runs when a skill has NO
    exact match: these are not wrong, but they are looser than the exact hit that
    already existed, and Agent E scores candidates on rating — so a loosely
    related course can outrank the exact one.
    """
    # real gains, where exact matching found nothing
    assert covers_skill("problem solving", "problem-solving skills") is True
    assert covers_skill("ci/cd pipelines", "ci/cd") is True

    # real matches that are looser than an exact hit — allowed by the rule,
    # which is why the caller withholds widening when an exact match exists
    assert covers_skill("environmental project management", "project management") is True
    assert covers_skill("project management office", "project management") is True
