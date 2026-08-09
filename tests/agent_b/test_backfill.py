"""The operator backfills, and the limits that make them safe to run live.

`--reslice-roundups` rewrites a text field on 245 real rows and
`--backfill-destinations` sets columns on rows a user's dashboard already reads.
Neither is a cycle: there is no `--dry-run`-by-default, no staging table, and no
undo. What makes them safe is that each can only NARROW — replace a description
with a verified substring of itself, or fill a column that was NULL — and the
tests here are mostly about proving they cannot do anything else.

The one that matters most is `test_a_backfill_can_never_remove_a_posting`. A
repair command with a delete in it is a different and far more dangerous tool
than the one that was asked for.
"""

from __future__ import annotations

from typing import Any

import pytest

from agents.agent_b_job_ingest.backfill import backfill_destinations, reslice_roundups
from agents.agent_b_job_ingest.hashing import child_source_url, posting_id
from agents.agent_b_job_ingest.schemas import JobExtraction, JobExtractionBatch

POST_URL = "https://oman.el7far.com/2026/08/majees.html"
ACCOUNTS = ("Accounts Payable In-Charge. Process supplier invoices, run monthly payment "
            "cycles and reconcile vendor statements. Five years' experience required.")
WELDER = ("Welding Supervisor. Supervise eight welders on an EPC site, sign off weld "
          "quality and maintain the inspection log. Trade certification essential.")
BODY = f"Majees Technical Services — 38 Job Opportunities\n\n{ACCOUNTS}\n\n{WELDER}\n\nApply: hr@majees.com"


def child(title: str, index: int) -> str:
    return posting_id("el7far", child_source_url(POST_URL, title, index))


ACCOUNTS_ID = child("Accounts Payable In-Charge", 0)
WELDER_ID = child("Welding Supervisor", 1)


class FakeStore:
    """Just the four methods the backfills call."""

    def __init__(self, posts: list[dict[str, Any]]):
        self.posts = posts
        self.narrowed: dict[str, str] = {}
        self.updated: dict[str, dict[str, Any]] = {}
        self.deleted: list[str] = []

    def clustered_posts(self, *, sources=None, limit=None):
        return self.posts

    def narrow_descriptions(self, updates):
        self.narrowed.update(updates)
        return len(updates)

    def rows_without_destination(self, *, sources=None, limit=None,
                                 single_vacancy_only=False):
        return self.posts

    def update_posting(self, posting_id, values):
        self.updated[posting_id] = values


def clustered_post(**over) -> dict[str, Any]:
    post = {
        "source_post_url": POST_URL, "source": "el7far", "source_type": "blogger_feed",
        "children_rows": [{"posting_id": ACCOUNTS_ID, "title": "Accounts Payable In-Charge"},
                          {"posting_id": WELDER_ID, "title": "Welding Supervisor"}],
        "children": 2, "distinct_bodies": 1, "body": BODY,
        "posting_ids": [ACCOUNTS_ID, WELDER_ID], "needs_reslice": True,
    }
    post.update(over)
    return post


class Extractor:
    def __init__(self, jobs): self.jobs = jobs
    def invoke(self, payload): return JobExtractionBatch(jobs=self.jobs)


class Exploding:
    def invoke(self, payload): raise ValueError("model said something unparseable")


GOOD = [JobExtraction(title="Accounts Payable In-Charge", vacancy_text=ACCOUNTS),
        JobExtraction(title="Welding Supervisor", vacancy_text=WELDER)]


# ---------------------------------------------------------------------------
# re-slicing
# ---------------------------------------------------------------------------
def test_each_vacancy_is_narrowed_to_its_own_text():
    store = FakeStore([clustered_post()])
    report = reslice_roundups(store=store, extractor=Extractor(GOOD))

    assert store.narrowed == {ACCOUNTS_ID: ACCOUNTS, WELDER_ID: WELDER}
    assert report.rows_updated == 2


def test_a_backfill_can_never_remove_a_posting():
    """THE guarantee. This command was asked for as "remove the old rows"; what
    it does instead is narrow a field. It has no delete path at all, and a
    vacancy the re-extraction fails to reproduce is LEFT ALONE rather than
    tidied away."""
    store = FakeStore([clustered_post()])
    # The model returns only one of the two vacancies this time.
    reslice_roundups(store=store, extractor=Extractor([GOOD[0]]))

    assert store.deleted == []
    assert WELDER_ID not in store.narrowed, "the unmatched vacancy is untouched, not removed"
    assert ACCOUNTS_ID in store.narrowed


def test_a_vacancy_whose_title_no_longer_matches_is_skipped():
    """Matching is by role title, falling back to the minted id. A vacancy the
    model renames matches neither — and must NOT mint a row, because a repair
    command that creates postings is not a repair command."""
    store = FakeStore([clustered_post()])
    renamed = [JobExtraction(title="Senior Accounts Payable Lead", vacancy_text=ACCOUNTS)]
    report = reslice_roundups(store=store, extractor=Extractor(renamed))

    assert store.narrowed == {}
    assert report.rows_updated == 0


def test_an_unverifiable_span_leaves_the_row_and_is_counted():
    store = FakeStore([clustered_post()])
    invented = [JobExtraction(title="Accounts Payable In-Charge",
                              vacancy_text="Prose the article never contained anywhere at all.")]
    report = reslice_roundups(store=store, extractor=Extractor(invented))

    assert store.narrowed == {}
    assert report.unverified == 1


def test_an_already_narrowed_post_is_skipped_without_calling_the_model():
    """Idempotent by observation, not by a flag: children with distinct bodies
    have already been sliced. Re-running the command must not spend 40 more
    LLM calls to discover it has nothing to do."""
    store = FakeStore([clustered_post(distinct_bodies=2, needs_reslice=False)])
    report = reslice_roundups(store=store, extractor=Exploding())

    assert store.narrowed == {}
    assert report.rows_unchanged == 2


def test_one_bad_post_costs_one_post():
    """The lesson a 352-posting census taught at full price."""
    good = clustered_post()
    bad = clustered_post(source_post_url="https://oman.el7far.com/2026/08/other.html",
                         body="POISON body")

    class Fussy:
        def invoke(self, payload):
            if "POISON" in str(payload):
                raise ValueError("model said something unparseable")
            return JobExtractionBatch(jobs=GOOD)

    store = FakeStore([good, bad])
    report = reslice_roundups(store=store, extractor=Fussy())

    assert report.rows_updated == 2          # the healthy post still landed
    assert len(report.failures) == 1


def test_dry_run_writes_nothing_but_still_reports():
    store = FakeStore([clustered_post()])
    report = reslice_roundups(store=store, extractor=Extractor(GOOD), dry_run=True)

    assert store.narrowed == {}
    assert report.rows_updated == 2, "a dry run must still say what it would do"


# ---------------------------------------------------------------------------
# destination backfill
# ---------------------------------------------------------------------------
EMPLOYER = "https://careers.majees.com/jobs/accounts-payable-88213"


def stored_row(**over) -> dict[str, Any]:
    row = {"posting_id": ACCOUNTS_ID, "source": "el7far", "source_type": "blogger_feed",
           "source_url": "https://oman.el7far.com/2026/08/accounts.html",
           "title": "Accounts Payable In-Charge"}
    row.update(over)
    return row


class Fetcher:
    def __init__(self, text): self.text = text
    def fetch(self, url): return self.text


DESTINATION = ("Accounts Payable In-Charge at Majees. Full-time, remote. "
               "Requirements: accounts payable, reconciliation, ERP systems.")


def test_a_resolved_destination_is_recorded_with_what_it_states():
    store = FakeStore([stored_row()])
    job = JobExtraction(required_skills=["accounts payable", "reconciliation"],
                        work_arrangement="remote", employment_type="full_time")
    report = backfill_destinations(
        store=store, extractor=Extractor([job]), root_fetcher=Fetcher(DESTINATION),
        article_fetch=lambda url: (EMPLOYER,))

    written = store.updated[ACCOUNTS_ID]
    assert written["final_url"] == EMPLOYER
    assert written["required_skills"] == ["accounts payable", "reconciliation"]
    assert written["work_arrangement"] == "remote"
    assert report.rows_updated == 1


def test_a_posting_with_no_followable_link_is_left_alone():
    """Measured on el7far: 86% of single postings are like this — they describe
    the job themselves and give an email address."""
    store = FakeStore([stored_row()])
    report = backfill_destinations(
        store=store, extractor=Extractor([JobExtraction()]),
        root_fetcher=Fetcher(DESTINATION),
        article_fetch=lambda url: ("https://www.facebook.com/groups/omanjobs",))

    # No destination fields written — but WHY is recorded, because a prune that
    # deletes rows for lacking a destination has to be auditable afterwards.
    assert store.updated[ACCOUNTS_ID] == {"destination_status": "hub"}
    assert report.rows_unchanged == 1


def test_a_hub_page_is_not_treated_as_a_destination():
    """Several vacancies on the page means a listing, and nobody applies on a
    listing."""
    store = FakeStore([stored_row()])
    report = backfill_destinations(
        store=store, extractor=Extractor([JobExtraction(), JobExtraction()]),
        root_fetcher=Fetcher(DESTINATION), article_fetch=lambda url: (EMPLOYER,))

    assert store.updated[ACCOUNTS_ID] == {"destination_status": "hub"}
    assert report.rows_unchanged == 1


def test_facts_the_destination_does_not_state_are_not_written():
    """`verify_stated_facts` still applies. A backfilled row must be held to the
    same standard as one enriched during a cycle — otherwise the repair path
    becomes the way fabrications get in."""
    store = FakeStore([stored_row()])
    job = JobExtraction(required_skills=["accounts payable"],
                        work_arrangement="onsite",     # the page never says so
                        salary_min=900)                # nor this
    backfill_destinations(
        store=store, extractor=Extractor([job]), root_fetcher=Fetcher(DESTINATION),
        article_fetch=lambda url: (EMPLOYER,))

    written = store.updated[ACCOUNTS_ID]
    assert "work_arrangement" not in written
    assert "salary_min" not in written
    assert written["required_skills"] == ["accounts payable"]


def test_an_unreachable_destination_changes_nothing():
    store = FakeStore([stored_row()])
    report = backfill_destinations(
        store=store, extractor=Extractor([JobExtraction()]),
        root_fetcher=Fetcher(None), article_fetch=lambda url: (EMPLOYER,))

    assert store.updated[ACCOUNTS_ID] == {"destination_status": "unreachable"}
    assert report.rows_unchanged == 1


def test_the_store_refuses_a_column_the_backfill_did_not_declare():
    """`update_posting` takes column names from an allowlist. A backfill that
    could name a column could name `status` or `duplicate_of`."""
    from agents.agent_b_job_ingest.db.store import JobStore

    store = JobStore("postgresql://unused/unused")
    with pytest.raises(ValueError, match="unknown columns"):
        store.update_posting("abc", {"status": "rejected"})


def test_matching_survives_the_model_reordering_the_vacancies():
    """Ids are `sha(post_url#slug-INDEX)`, so order-based matching breaks the
    moment the model lists the roles differently — finding E3 of the Agent B
    audit. Title matching is what makes a repair possible at all."""
    store = FakeStore([clustered_post()])
    reversed_order = [GOOD[1], GOOD[0]]
    reslice_roundups(store=store, extractor=Extractor(reversed_order))

    assert store.narrowed == {ACCOUNTS_ID: ACCOUNTS, WELDER_ID: WELDER}


def test_two_rows_with_the_same_title_match_nothing():
    """Writing one vacancy's text onto another's row is worse than leaving both
    alone, so an ambiguous title is declined rather than guessed."""
    twin = clustered_post(children_rows=[
        {"posting_id": ACCOUNTS_ID, "title": "Accounts Payable In-Charge"},
        {"posting_id": WELDER_ID, "title": "Accounts Payable In-Charge"}])
    store = FakeStore([twin])
    reslice_roundups(store=store, extractor=Extractor([GOOD[0]]))

    assert store.narrowed == {}


def test_no_link_and_a_hub_link_are_recorded_as_different_facts():
    """Both mean "no destination", but they say different things about the
    source — and the prune's audit trail needs both. Measured on 30 el7far
    postings: 16 had no link at all, 11 pointed at a hub or LinkedIn."""
    bare = FakeStore([stored_row()])
    backfill_destinations(store=bare, extractor=Extractor([JobExtraction()]),
                          root_fetcher=Fetcher(None), article_fetch=lambda url: ())
    assert bare.updated[ACCOUNTS_ID] == {"destination_status": "no_link"}

    hub = FakeStore([stored_row()])
    backfill_destinations(store=hub, extractor=Extractor([JobExtraction()]),
                          root_fetcher=Fetcher(None),
                          article_fetch=lambda url: ("https://acme.test/careers",))
    assert hub.updated[ACCOUNTS_ID] == {"destination_status": "hub"}


def test_a_resolved_row_records_that_too():
    """Not only failures. A row that reached a vacancy page says so, which is
    what lets the prune target measured failures rather than 'final_url IS
    NULL' — an untraced row must never be mistaken for a traced one."""
    store = FakeStore([stored_row()])
    job = JobExtraction(required_skills=["accounts payable"])
    backfill_destinations(store=store, extractor=Extractor([job]),
                          root_fetcher=Fetcher(DESTINATION),
                          article_fetch=lambda url: (EMPLOYER,))

    assert store.updated[ACCOUNTS_ID]["destination_status"] == "resolved"


def test_a_dry_run_reports_the_breakdown_without_writing_it():
    store = FakeStore([stored_row()])
    report = backfill_destinations(store=store, extractor=Extractor([JobExtraction()]),
                                   root_fetcher=Fetcher(None),
                                   article_fetch=lambda url: (), dry_run=True)

    assert store.updated == {}
    assert report.outcomes == {"no_link": 1}


def test_a_roundup_child_never_inherits_its_posts_link():
    """MEASURED, after briefly getting this wrong: 49 of 55 surviving rows
    shared one destination, and a row titled "Business Development Manager"
    pointed at `.../jobs/senior-lowcode-developer-381`.

    A roundup article carries ONE outbound link, belonging to the whole
    roundup. Handing it to each child asserts that 19 different roles are all
    advertised at the same URL — a false claim about where to apply, which is
    worse than recording no destination at all.
    """
    store = FakeStore([stored_row(is_split=True)])
    report = backfill_destinations(
        store=store, extractor=Extractor([JobExtraction(required_skills=["x"])]),
        root_fetcher=Fetcher(DESTINATION), article_fetch=lambda url: (EMPLOYER,))

    assert store.updated[ACCOUNTS_ID] == {"destination_status": "hub"}
    assert "final_url" not in store.updated[ACCOUNTS_ID]
    assert report.rows_updated == 0


def test_a_single_vacancy_posting_still_gets_its_destination():
    """The exemption is for split children only — a normal posting's link is
    genuinely its own."""
    store = FakeStore([stored_row(is_split=False)])
    backfill_destinations(
        store=store, extractor=Extractor([JobExtraction(required_skills=["x"])]),
        root_fetcher=Fetcher(DESTINATION), article_fetch=lambda url: (EMPLOYER,))

    assert store.updated[ACCOUNTS_ID]["final_url"] == EMPLOYER
