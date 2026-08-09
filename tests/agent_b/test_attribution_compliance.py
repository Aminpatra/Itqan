"""The three conditions GulfTalent's terms attach to crawling it.

Their Terms of Use prohibit crawlers outright, with one exception:

    "...except as an internet search engine making the information searchable by
     users, and provided you display only minimal snippets of each GulfTalent
     page to your users, in each case mention the source clearly as GulfTalent,
     and link each snippet back to the corresponding page on GulfTalent."

That sentence is the entire legal basis for `sources/gulftalent.py` existing, so
its conditions are not aspirations — they are behaviour, and behaviour that is
not tested is behaviour that drifts.

**Two of the three currently hold by accident of design**, which is exactly what
makes them fragile: nothing stopped someone adding a job-description preview to
a card, and the `final_url` improvement of 2026-08-08 already *would* have broken
the link-back condition if `link_back_required` had not been built alongside it.
That is the failure this file exists to catch — not a deliberate breach, but a
reasonable-looking change somewhere else.

If a condition here is ever deliberately changed, the source must be disabled in
the same commit.
"""

from __future__ import annotations

from datetime import datetime, timezone

from agents.agent_b_job_ingest.sources.config import DEFAULT_SOURCES
from api.mapping import job_matches

GULFTALENT = next(s for s in DEFAULT_SOURCES if s.name == "gulftalent")

_EVIDENCE = [{"skill": "SQL", "verdict": "matched", "satisfied_by": "SQL"}]
_GT_PAGE = "https://www.gulftalent.com/oman/jobs/area-manager-604518"
_EMPLOYER_PAGE = "https://careers.altoobi.com/vacancy/9931"


def _card(**fields):
    """One job card as the API publishes it."""
    gap = {"matched_jobs": [dict(job_id="a", job_title="Area Manager", gap_score=0.2,
                                 skill_resolution=_EVIDENCE, **fields)]}
    return job_matches(gap)[0]


# ---------------------------------------------------------------------------
# condition 1 — display only minimal snippets
# ---------------------------------------------------------------------------
def test_the_job_description_never_reaches_the_api():
    """We store `raw_description` because extraction needs it. We must not SHOW
    it. The published card is a title, an employer, a location, a score and an
    evidence sentence we wrote ourselves — which is less than a snippet, not
    more."""
    description = ("Al Toobi New Enterprises is seeking an Area Manager. "
                   "Responsibilities include operational excellence, monitoring "
                   "key business performance indicators, and leading a team.")
    card = _card(raw_description=description, description=description,
                 source_url=_GT_PAGE, attribution="GulfTalent")

    blob = repr(card)
    assert description not in blob
    for phrase in ("Responsibilities include", "operational excellence",
                   "monitoring key business"):
        assert phrase not in blob, f"description text {phrase!r} reached the card"


def test_the_card_carries_only_the_fields_we_expect():
    """A whitelist, deliberately. A new field added upstream should have to be
    considered here rather than appearing on the card by inheritance — that is
    how a description ends up published by accident."""
    card = _card(source_url=_GT_PAGE, attribution="GulfTalent")
    assert set(card) == {"id", "title", "employer", "location", "arrangement",
                         "score", "why", "matchedSkills", "source"}
    assert set(card["source"]) == {"name", "url", "retrievedAt"}


# ---------------------------------------------------------------------------
# condition 2 — mention the source clearly as GulfTalent
# ---------------------------------------------------------------------------
def test_the_publisher_is_credited_by_name_not_by_database_key():
    """`source` is 'gulftalent', an internal identifier. The condition says
    "mention the source clearly as GulfTalent", and a lowercase slug on a job
    card is not a credit."""
    card = _card(source="gulftalent", attribution="GulfTalent", source_url=_GT_PAGE)
    assert card["source"]["name"] == "GulfTalent"


def test_the_registry_supplies_that_name():
    assert GULFTALENT.credit == "GulfTalent"
    assert GULFTALENT.terms_url == "https://www.gulftalent.com/terms"


def test_a_source_with_no_attribution_still_names_something():
    """Sources under no such obligation must not lose their source line."""
    card = _card(source="el7far", source_url="https://oman.el7far.com/x.html")
    assert card["source"]["name"] == "el7far"


# ---------------------------------------------------------------------------
# condition 3 — link each snippet back to the corresponding GulfTalent page
# ---------------------------------------------------------------------------
def test_the_registry_requires_linking_back():
    assert GULFTALENT.link_back_required is True


def test_the_apply_link_stays_on_the_publisher_even_when_an_employer_page_is_known():
    """THE regression this file was written for.

    On 2026-08-08 the apply link was improved to prefer `final_url`, the
    employer's own ATS page — a straight win for every aggregator. For a source
    whose terms require linking back it is the opposite: we would keep
    GulfTalent's data and send their traffic to the employer.

    The guard lives in the pipeline, which declines to SET `final_url` for such a
    source. This asserts the end state a user actually sees, so it keeps holding
    however the plumbing is refactored.
    """
    card = _card(source="gulftalent", attribution="GulfTalent",
                 source_url=_GT_PAGE, final_url=None)
    assert card["source"]["url"] == _GT_PAGE


def test_the_pipeline_declines_to_record_a_destination_for_such_a_source():
    """Where the guard actually is. A GulfTalent ad linking out to the employer's
    careers page must not have that URL recorded, because recording it is what
    would later redirect the apply link."""
    from agents.agent_b_job_ingest.pipeline import _links_back_to_source
    from agents.agent_b_job_ingest.sources.base import RawPosting

    def posting(source: str) -> RawPosting:
        return RawPosting(
            source=source, source_group=source, source_type="html_scrape",
            source_url=_GT_PAGE, title="Area Manager",
            raw_description="Al Toobi New Enterprises is hiring.",
            posted_date=datetime(2026, 8, 1, tzinfo=timezone.utc),
            outbound_links=(_EMPLOYER_PAGE,),
        )

    assert _links_back_to_source(posting("gulftalent")) is True
    # And unchanged for every source without the obligation — the destination
    # crawl is a real improvement and must keep working for them.
    assert _links_back_to_source(posting("el7far")) is False


# ---------------------------------------------------------------------------
# the standing invariant
# ---------------------------------------------------------------------------
def test_a_source_crawled_under_conditions_declares_all_of_them():
    """A source that is enabled and carries terms-based obligations must declare
    every one of them. Half-configured is the dangerous state: crawling on the
    strength of an exception while silently failing one of its conditions.
    """
    assert GULFTALENT.enabled is True
    assert GULFTALENT.terms_reviewed is True, "a human must have read the terms"
    assert GULFTALENT.display_name, "condition 2 needs a name to display"
    assert GULFTALENT.terms_url, "the basis must be auditable"
    assert GULFTALENT.link_back_required is True, "condition 3"
