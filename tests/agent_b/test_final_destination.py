"""Where the job actually is, and what the employer's page says about it.

Two behaviours, one crawl. `root_fetch` has always followed the aggregator's
outbound link to the employer's ATS and harvested skills from it — and then
thrown the URL away. Now it records it, which buys two things:

* an apply link that points at the employer instead of the aggregator;
* an exact-match duplicate signal, which is stronger than the embedding near-dup
  it supplements — two aggregators writing up one vacancy in different words are
  the case similarity is worst at, and a shared destination settles it.

Plus the fields an employer page states plainly and a roundup summary never
does. THE RULE THAT NEEDS THE TESTS IS THE HONESTY ONE: every one of them stays
NULL unless the page says so. This project has had to fix that class of bug
three times already (gap_score null not 0.0, price null not 0, birth date never
inferred), so it is pinned here rather than trusted to a prompt.
"""

from __future__ import annotations

from datetime import datetime, timezone

from agents.agent_b_job_ingest.schemas import (
    JobExtraction,
    JobExtractionBatch,
    LegitimacyVerdict,
)
from agents.agent_b_job_ingest.sources.base import RawPosting
from tests.agent_b.fake_store import FakeStore
from tests.agent_b.test_root_enrich import FakeRootFetcher, make_pipeline
from tests.fake_llm import FakeStructuredLLM

EMPLOYER_URL = "https://jobs.eni.com/en/sites/CX_1004/job/33730"
HUB_URL = "https://rihal.om/careers"
ROOT_TEXT = "Well Engineer at Eni, Muscat. Remote. Requirements: well planning, drilling."


def posting(*, source="el7far", url=None, links=(), title="Eni – Well Engineer"):
    return RawPosting(
        source=source,
        source_group=f"{source}_network",
        source_type="blogger_feed",
        source_url=url or f"https://{source}.example/2026/07/eni-well-engineer.html",
        title=title,
        raw_description="Eni is hiring a Well Engineer. Apply on the company site.",
        posted_date=datetime(2026, 7, 20, tzinfo=timezone.utc),
        outbound_links=links,
    )


class RootLLM(FakeStructuredLLM):
    """Thin for the aggregator body, rich for the destination page.

    `extra` is what the destination page is made to state — the whole point of
    the parametrised tests below is what happens when it states NOTHING.
    """

    def __init__(self, *, jobs_on_root=1, **extra):
        super().__init__()
        self.jobs_on_root = jobs_on_root
        self.extra = extra

    def respond(self, schema, payload):
        if schema.__name__ == "JobExtractionBatch":
            if "drilling" in str(payload):          # the destination page
                job = JobExtraction(sector="2", required_skills=["well planning"],
                                    **self.extra)
                return JobExtractionBatch(jobs=[job] * self.jobs_on_root)
            return JobExtractionBatch(jobs=[JobExtraction(
                sector="2", required_skills=["oil and gas"])])
        return super().respond(schema, payload)


def run_one(store=None, root_text=ROOT_TEXT, **llm_kwargs):
    store = store if store is not None else FakeStore()
    fetcher = FakeRootFetcher({EMPLOYER_URL: root_text})
    pipe = make_pipeline(store, root_fetcher=fetcher, llm=RootLLM(**llm_kwargs))
    summary = pipe.run([posting(links=(EMPLOYER_URL,))])
    return store, summary, next(iter(store.rows.values()))


# ---------------------------------------------------------------------------
# the destination itself
# ---------------------------------------------------------------------------
def test_the_employer_page_is_recorded_as_where_you_apply():
    """The link was already being followed and extracted from. Recording it is
    the difference between knowing the employer's page and sending the user
    there."""
    _, _, row = run_one()
    assert row.final_url == EMPLOYER_URL
    assert row.source_url.startswith("https://el7far.example/")   # unchanged


def test_identity_does_not_move_to_the_destination():
    """`posting_id = sha(source, source_url)` stays as it is. Making the employer
    URL the identity would re-mint every row, age the old ones out, and
    double-count demand across the overlap — finding E3 of the Agent B audit."""
    store, _, row = run_one()
    pid = next(iter(store.rows))
    _, _, again = run_one(store=store)
    assert list(store.rows) == [pid]
    assert again.final_url == EMPLOYER_URL


def test_a_hub_page_never_becomes_an_apply_link():
    """A page listing several vacancies is a hub. Nobody applies on a hub, and
    the existing enrichment already declines to harvest from one — the URL must
    be declined on the same evidence."""
    _, _, row = run_one(jobs_on_root=2)
    assert row.final_url is None


def test_a_destination_that_teaches_us_nothing_is_still_the_destination():
    """"We learned no skills from that page" and "that is not where you apply"
    are different statements. Only the second should withhold the link."""
    store = FakeStore()
    fetcher = FakeRootFetcher({EMPLOYER_URL: ROOT_TEXT})

    class NoSkills(RootLLM):
        def respond(self, schema, payload):
            if schema.__name__ == "JobExtractionBatch" and "drilling" in str(payload):
                return JobExtractionBatch(jobs=[JobExtraction(sector="2",
                                                              required_skills=[])])
            return super().respond(schema, payload)

    pipe = make_pipeline(store, root_fetcher=fetcher, llm=NoSkills())
    summary = pipe.run([posting(links=(EMPLOYER_URL,))])
    row = next(iter(store.rows.values()))

    assert row.final_url == EMPLOYER_URL
    assert summary.root_enrichments == 0            # nothing was learned
    assert row.required_skills == ["oil and gas"]   # aggregator's kept, not erased


def test_an_unreachable_destination_leaves_no_link():
    """A robots refusal or a timeout must not produce a URL we never resolved."""
    store = FakeStore()
    pipe = make_pipeline(store, root_fetcher=FakeRootFetcher({}), llm=RootLLM())
    pipe.run([posting(links=(EMPLOYER_URL,))])
    assert next(iter(store.rows.values())).final_url is None


# ---------------------------------------------------------------------------
# null unless the page says so — the rule that matters
# ---------------------------------------------------------------------------
def test_a_silent_destination_leaves_every_new_field_null():
    """The default case, and the one worth failing a build over.

    A defaulted `onsite` reads as a fact about the employer; a salary of 0 reads
    as unpaid; a defaulted `full_time` misdescribes an internship. Silence must
    survive the whole path — schema, merge, row — as silence.
    """
    _, _, row = run_one()
    assert row.work_arrangement is None
    assert row.employment_type is None
    assert row.salary_min is None and row.salary_max is None
    assert row.salary_currency is None and row.salary_period is None


STATED = ("Well Engineer internship at Eni. Remote. drilling. "
          "Salary 800 - 1,200 OMR per month.")


def test_what_the_destination_states_is_recorded():
    _, _, row = run_one(root_text=STATED,
                        work_arrangement="remote", employment_type="internship",
                        salary_min=800, salary_max=1200,
                        salary_currency="OMR", salary_period="month")
    assert row.work_arrangement == "remote"
    assert row.employment_type == "internship"
    assert (row.salary_min, row.salary_max) == (800, 1200)
    assert (row.salary_currency, row.salary_period) == ("OMR", "month")


def test_a_stated_zero_salary_is_kept_because_it_is_an_answer():
    """`or` would drop it — 0 is falsy. An unpaid internship saying so is a fact
    about the role, and it is exactly the fact a graduate needs before applying."""
    _, _, row = run_one(root_text="Unpaid internship. drilling. 0 OMR per month.",
                        salary_min=0, salary_max=0, salary_period="month")
    assert row.salary_min == 0 and row.salary_max == 0


# ---------------------------------------------------------------------------
# the fence, end to end
# ---------------------------------------------------------------------------
def test_an_arrangement_the_page_never_stated_does_not_reach_the_row():
    """MEASURED on the first live cycle: 19 of 19 postings came back 'onsite'
    and NOT ONE of them contained an arrangement phrase in either language. The
    prompt says "an office address is NOT a statement that the role is onsite"
    and the model read the city name anyway.

    That is this repo's standing finding on its third field. The instruction
    stays; this is what enforces it.
    """
    store, summary, row = run_one(root_text="Well Engineer at Eni, Muscat. drilling.",
                                  work_arrangement="onsite",
                                  employment_type="full_time")
    assert row.work_arrangement is None
    assert row.employment_type is None
    assert summary.unstated_fields_dropped >= 2


def test_a_salary_figure_absent_from_the_page_is_dropped_whole():
    """The most damaging field on the list to invent — it is what a person makes
    a decision on. Same rule as Agent E's rationale verifier: every figure must
    appear in the text it claims to come from."""
    _, _, row = run_one(root_text="Well Engineer. drilling. Competitive salary.",
                        salary_min=900, salary_currency="OMR",
                        salary_period="month")
    assert row.salary_min is None
    # And the currency and period go with it: they describe an amount that is
    # no longer there.
    assert row.salary_currency is None and row.salary_period is None


def test_a_page_naming_two_arrangements_records_neither():
    """"Remote or hybrid considered" does not settle the question, and picking
    one would be the same guess in a smaller coat."""
    _, _, row = run_one(root_text="Well Engineer. drilling. Remote or hybrid considered.",
                        work_arrangement="remote")
    assert row.work_arrangement is None


def test_the_schema_refuses_a_value_outside_the_vocabulary():
    """The database CHECK is the last line, not the first. A model answering
    'wfh' or 'freelance' should fail at the schema, where the run can see it."""
    import pytest
    from pydantic import ValidationError

    for bad in ({"work_arrangement": "wfh"}, {"employment_type": "freelance"},
                {"salary_period": "fortnight"}):
        with pytest.raises(ValidationError):
            JobExtraction(sector="2", required_skills=[], **bad)


# ---------------------------------------------------------------------------
# one vacancy, however many aggregators carried it
# ---------------------------------------------------------------------------
def test_two_aggregators_carrying_one_vacancy_collapse_to_one_row():
    """The case embedding near-dup is worst at: one vacancy written up twice, in
    different words. A shared destination settles it exactly rather than
    probably."""
    store = FakeStore()
    fetcher = FakeRootFetcher({EMPLOYER_URL: ROOT_TEXT})
    pipe = make_pipeline(store, root_fetcher=fetcher, llm=RootLLM())

    summary = pipe.run([
        posting(source="el7far", links=(EMPLOYER_URL,)),
        posting(source="dubizzle", links=(EMPLOYER_URL,), title="Well Engineer"),
    ])

    assert summary.destination_duplicates == 1
    canonicals = [r for r in store.rows.values() if r.duplicate_of is None]
    dups = [r for r in store.rows.values() if r.duplicate_of is not None]
    assert len(canonicals) == 1 and len(dups) == 1
    # Both rows survive — a duplicate is recorded, never deleted, so demand is
    # counted once while the second source's coverage stays auditable.
    assert dups[0].duplicate_of == canonicals[0].posting_id


def test_a_stored_canonical_wins_over_anything_in_this_batch():
    """Otherwise a later cycle would flip which row is canonical, and every
    consumer holding the old id would be pointing at a duplicate."""
    store = FakeStore()
    fetcher = FakeRootFetcher({EMPLOYER_URL: ROOT_TEXT})
    pipe = make_pipeline(store, root_fetcher=fetcher, llm=RootLLM())

    pipe.run([posting(source="el7far", links=(EMPLOYER_URL,))])
    first = next(iter(store.rows))

    pipe.run([posting(source="dubizzle", links=(EMPLOYER_URL,))])
    newcomer = store.rows[[p for p in store.rows if p != first][0]]
    assert newcomer.duplicate_of == first
    assert store.rows[first].duplicate_of is None


def test_postings_without_a_destination_are_never_pooled():
    """Every row's `final_url` is NULL today and will be for older rows for ever
    (the change is forward-only). If NULL were treated as a shared destination,
    the first cycle would collapse the whole corpus into one vacancy."""
    store = FakeStore()
    pipe = make_pipeline(store, root_fetcher=FakeRootFetcher({}), llm=RootLLM())

    summary = pipe.run([
        posting(source="el7far", links=(HUB_URL,)),
        posting(source="dubizzle", links=(HUB_URL,), title="Something else"),
    ])

    assert summary.destination_duplicates == 0
    assert all(r.duplicate_of is None for r in store.rows.values())
