"""Durations: what a provider actually said, and what must never be inferred.

`test_a_range_is_never_collapsed_to_a_midpoint` is the one that matters. The
reason course durations were left unstored the first time was that reducing
"2-4 hours a week for 4 weeks" to a single number means choosing a point nobody
stated — the same rule that keeps `gap_score` null instead of 0.0 and an
unpriced course null instead of free.

Every string in the parametrised cases below is a real shape from the live
Coursera catalogue, not an invented one.
"""

from __future__ import annotations

import pytest

from agents.agent_d_course_ingest.duration import parse_iso8601, parse_workload


# ---------------------------------------------------------------------------
# the real shapes
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("text, expected", [
    ("1 hour 30 minutes", (1.5, 1.5)),
    ("2 heures", (2.0, 2.0)),                       # French, and it is in the data
    ("Approx. 12 hours", (12.0, 12.0)),
    ("45 minutes", (0.75, 0.75)),
    ("4 weeks of study, 2-4 hours a week", (8.0, 16.0)),
    ("6 weeks of study, 3 hours a week", (18.0, 18.0)),
    ("2-4 hours", (2.0, 4.0)),
    ("90-120 minutes", (1.5, 2.0)),
])
def test_stated_durations_are_read_as_stated(text, expected):
    assert parse_workload(text) == expected


@pytest.mark.parametrize("text", ["", "   ", None, "Self-paced", "varies",
                                  "at your own pace", "4 weeks"])
def test_nothing_stated_is_nothing_returned(text):
    """Unparseable is (None, None), never 0.0 — the caller keeps the raw text so
    a person still sees whatever the provider wrote.

    "4 weeks" alone is deliberately here: elapsed calendar time is not study
    effort, and converting it would be a fabrication.
    """
    assert parse_workload(text) == (None, None)


# ---------------------------------------------------------------------------
# the rule
# ---------------------------------------------------------------------------
def test_a_range_is_never_collapsed_to_a_midpoint():
    """THE test. 8..16 must stay 8..16; publishing 12 would be a figure no
    provider ever stated."""
    low, high = parse_workload("4 weeks of study, 2-4 hours a week")

    assert (low, high) == (8.0, 16.0)
    assert low != high, "a range was averaged into a single number"


def test_equal_ends_mean_one_stated_figure_not_an_average():
    """'2 heures' is a point, and both ends being 2.0 says so. It is NOT the
    midpoint of anything."""
    assert parse_workload("2 heures") == (2.0, 2.0)


def test_the_range_is_ordered_so_a_card_never_reads_backwards():
    """A backwards range is the author's typo. Ordering it keeps a real duration;
    storing it as written would put '16 to 8 hours' in front of someone — and the
    migration's CHECK would reject the row outright."""
    low, high = parse_workload("4-2 hours")

    assert (low, high) == (2.0, 4.0)


def test_a_weekly_rate_in_MINUTES_is_converted_not_read_as_a_total():
    """The same bug in another unit, also found live.

    "30 min/week" matched none of the hour-words, fell through to the
    single-figure rule, and was read as 30 minutes for the whole course. Over 6
    weeks it is 3 hours — and reading the 30 as hours instead would claim 180.
    """
    assert parse_workload("6 weeks, 30 min/week") == (3.0, 3.0)


@pytest.mark.parametrize("text", ["3-5 hours/week", "2 hours a week",
                                  "4-6 hours per week", "30 min/week"])
def test_a_weekly_rate_with_no_week_count_is_not_a_total(text):
    """Found on live Coursera data, and it was wrong in the dangerous direction.

    "3-5 hours/week" was being read as "3-5 hours" — one week's effort presented
    as the whole course, understating it by however many weeks it runs. Without
    a week count the total is unknowable, so nothing is claimed and the
    provider's words are shown instead.
    """
    assert parse_workload(text) == (None, None)


def test_the_same_rate_WITH_a_week_count_is_a_total():
    """The other half: given both halves, the multiplication is real."""
    assert parse_workload("2 weeks of study, 5-7 hours/week") == (10.0, 14.0)


def test_a_hyphen_inside_words_is_not_a_range():
    """'self-paced' must not read as a duration."""
    assert parse_workload("self-paced course") == (None, None)


def test_the_weekly_shape_wins_over_the_bare_week_count():
    """The string contains both '4 weeks' and '2-4 hours a week'. Reading the
    first would give 4 hours; the total is 8-16."""
    assert parse_workload("4 weeks of study, 2-4 hours a week") == (8.0, 16.0)


# ---------------------------------------------------------------------------
# ISO-8601, which edX publishes in its JSON-LD
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("text, expected", [
    ("PT6H", (6.0, 6.0)),
    ("PT6H30M", (6.5, 6.5)),
    ("PT90M", (1.5, 1.5)),
    ("P1D", (24.0, 24.0)),
])
def test_iso_durations_are_read(text, expected):
    assert parse_iso8601(text) == expected


def test_a_week_long_iso_duration_is_refused():
    """`P4W` is elapsed calendar time, not study effort. Converting it would
    claim 672 hours of work — wrong by two orders of magnitude, and stated with
    total confidence."""
    assert parse_iso8601("P4W") == (None, None)


@pytest.mark.parametrize("text", ["", None, "P", "PT", "not a duration"])
def test_malformed_iso_is_nothing(text):
    assert parse_iso8601(text) == (None, None)
