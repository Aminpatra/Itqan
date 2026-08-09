"""GulfTalent: sitemap enumeration, JSON-LD parsing, and the Oman-only filter.

Offline throughout — a FakeClient serves canned XML and HTML, so the whole
adapter is exercised without a request. The fixtures are shaped from the real
responses measured on 2026-08-08, including the JSON-LD block verbatim in
structure (`hiringOrganization`, `employmentType`, `baseSalary`).

**The country tests are the ones that matter most.** GulfTalent lists 28,947 job
ads and 430 of them are Oman. A filter that fails open does not degrade this
corpus, it replaces it: every gap score Agent C publishes would describe a
labour market our users do not live in.
"""

from __future__ import annotations

import json

import pytest

from agents.agent_b_job_ingest.sources.gulftalent import GulfTalentAdapter
from shared.config import Config

CFG = Config(user_agent="ItqanTestBot/1.0 (+t@e.test)")

SITEMAP_INDEX = """<?xml version="1.0" encoding="UTF-8"?>
<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <sitemap><loc>https://www.gulftalent.com/sitemaps/sitemap_jc000.xml</loc></sitemap>
  <sitemap><loc>https://www.gulftalent.com/sitemaps/sitemap_jx000.xml</loc></sitemap>
</sitemapindex>"""


def ad_sitemap(*urls: str) -> str:
    entries = "".join(f"<url><loc>{u}</loc></url>" for u in urls)
    return ('<?xml version="1.0" encoding="UTF-8"?>'
            '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
            f"{entries}</urlset>")


def ad_page(**overrides) -> str:
    payload = {
        "@context": "https://schema.org",
        "@type": "JobPosting",
        "title": "Area Manager",
        "description": "<p>Lead the region.</p><p>Own the P&amp;L.</p>",
        "hiringOrganization": {"@type": "Organization",
                               "name": "Al Toobi New Enterprises"},
        "jobLocation": {"@type": "Place",
                        "address": {"@type": "PostalAddress",
                                    "addressLocality": "Muscat",
                                    "addressCountry": "Oman"}},
        "employmentType": ["FULL_TIME"],
        "baseSalary": {"@type": "MonetaryAmount", "currency": "OMR",
                       "value": {"@type": "QuantitativeValue",
                                 "minValue": 650, "maxValue": 800,
                                 "unitText": "MONTH"}},
        "datePosted": "2026-07-05T00:00:00+00:00",
        "validThrough": "2026-10-03T00:00:00+00:00",
    }
    payload.update(overrides)
    for key, value in list(payload.items()):
        if value is None:
            del payload[key]
    return (f'<html><body><h1>{payload.get("title", "")}</h1>'
            f'<script type="application/ld+json">{json.dumps(payload)}</script>'
            "</body></html>")


class FakeClient:
    def __init__(self, pages: dict[str, str]):
        self.pages = pages
        self.fetched: list[str] = []
        self.bytes_fetched = 0

    def get_text(self, url: str, **_: object) -> str:
        self.fetched.append(url)
        if url not in self.pages:
            raise RuntimeError(f"404 {url}")
        return self.pages[url]

    def close(self) -> None:
        pass


class Yes:
    def can_fetch(self, url: str):
        from shared.scraping.robots import RobotsDecision
        return RobotsDecision(True, "test")

    def require(self, url: str) -> None:
        pass


OMAN_AD = "https://www.gulftalent.com/oman/jobs/area-manager-604518"


def build(pages, **kw):
    return GulfTalentAdapter(client=FakeClient(pages), robots=Yes(), config=CFG, **kw)


def base_pages(*ad_urls, page=None):
    pages = {
        "https://www.gulftalent.com/sitemap.xml": SITEMAP_INDEX,
        "https://www.gulftalent.com/sitemaps/sitemap_jx000.xml": ad_sitemap(*ad_urls),
    }
    for url in ad_urls:
        pages[url] = page or ad_page()
    return pages


# ---------------------------------------------------------------------------
# OMAN ONLY — the filter whose failure mode is a replaced corpus
# ---------------------------------------------------------------------------
def test_only_oman_ads_survive_a_gulf_wide_sitemap():
    """The live sitemap holds 28,947 ads across the Gulf and 430 are Oman. If
    this filter fails open, `skill_demand_stats` stops describing Oman."""
    urls = [
        OMAN_AD,
        "https://www.gulftalent.com/uae/jobs/sales-manager-611001",
        "https://www.gulftalent.com/qatar/jobs/executive-account-manager-619597",
        "https://www.gulftalent.com/saudi-arabia/jobs/site-engineer-600002",
        "https://www.gulftalent.com/bahrain/jobs/analyst-600003",
    ]
    adapter = build(base_pages(*urls))
    result = adapter.fetch()

    assert [p.source_url for p in result.postings] == [OMAN_AD]


def test_a_dubai_ad_with_omani_in_its_slug_is_not_oman():
    """`'oman' in url` would take this. The country lives in a path SEGMENT, and
    this is the near-miss that passes review and fails in production."""
    trap = "https://www.gulftalent.com/uae/jobs/omani-driver-wanted-612345"
    adapter = build(base_pages(OMAN_AD, trap))
    result = adapter.fetch()

    assert [p.source_url for p in result.postings] == [OMAN_AD]
    assert trap not in adapter._client.fetched, "we should not even fetch it"


def test_a_page_claiming_a_different_country_than_its_url_is_dropped():
    """CHECK 2. A URL saying Oman and a payload saying UAE means our filter is
    wrong or we followed a redirect. Either way the row is not wanted, and
    counting it makes the disagreement visible instead of silent."""
    pages = base_pages(OMAN_AD, page=ad_page(
        jobLocation={"@type": "Place",
                     "address": {"@type": "PostalAddress",
                                 "addressLocality": "Dubai",
                                 "addressCountry": "United Arab Emirates"}}))
    adapter = build(pages)
    result = adapter.fetch()

    assert result.postings == []
    assert adapter.out_of_scope == 1


def test_a_page_stating_no_country_is_kept_on_the_url_evidence():
    """Silence is not a contradiction. The URL already placed it in Oman, and
    dropping a posting because its JSON-LD omitted an optional property would
    lose real vacancies.

    And the country is still RECORDED, from the publisher's own filing. Leaving
    it NULL would be quietly fatal: `export_for_agent_c` filters on country, so
    the row would be stored and never retrieved. Measured on the first live
    cycle — 21 of 28 rows had a NULL country before this.
    """
    pages = base_pages(OMAN_AD, page=ad_page(jobLocation=None))
    result = build(pages).fetch()
    assert len(result.postings) == 1
    assert result.postings[0].source_url == OMAN_AD
    assert result.postings[0].country == "OM"


def test_the_country_reaches_the_row_even_when_the_model_states_none():
    """`row.country` came only from the LLM, which reads prose and frequently
    says nothing. The source knows — it filed the ad under a country."""
    from agents.agent_b_job_ingest.pipeline import IngestPipeline
    from agents.agent_b_job_ingest.prompts.extraction import EXTRACTION_PROMPT
    from agents.agent_b_job_ingest.prompts.legitimacy import LEGITIMACY_PROMPT
    from agents.agent_b_job_ingest.schemas import JobExtractionBatch, LegitimacyVerdict
    from shared.llm import structured
    from tests.agent_b.fake_store import FakeStore
    from tests.fake_embedder import FakeEmbedder
    from tests.fake_llm import FakeStructuredLLM

    store = FakeStore()
    llm = FakeStructuredLLM()
    pipe = IngestPipeline(
        store=store,
        extractor=EXTRACTION_PROMPT | structured(llm, JobExtractionBatch),
        adjudicator=LEGITIMACY_PROMPT | structured(llm, LegitimacyVerdict),
        embedder=FakeEmbedder(), config=CFG, model_name="fake",
    )
    pages = base_pages(OMAN_AD, page=ad_page(jobLocation=None))
    pipe.run(build(pages).fetch().postings)

    assert next(iter(store.rows.values())).country == "OM"


def test_category_pages_are_not_mistaken_for_ads():
    """`/oman/jobs/sales` is a category; `/oman/jobs/area-manager-604518` is an
    ad. The trailing numeric id is the whole difference."""
    adapter = build(base_pages(OMAN_AD, "https://www.gulftalent.com/oman/jobs/sales"))
    assert [p.source_url for p in adapter.fetch().postings] == [OMAN_AD]


# ---------------------------------------------------------------------------
# what the publisher states
# ---------------------------------------------------------------------------
def test_the_structured_fields_are_read_from_json_ld():
    posting = build(base_pages(OMAN_AD)).fetch().postings[0]

    assert posting.title == "Area Manager"
    assert posting.company == "Al Toobi New Enterprises"
    assert posting.employment_type == "full_time"
    # The corpus's first salary. 0 of 487 rows carried one before this source.
    assert (posting.salary_min, posting.salary_max) == (650.0, 800.0)
    assert posting.salary_currency == "OMR"
    assert posting.salary_period == "month"
    assert posting.posted_date.year == 2026
    assert posting.expires_at.month == 10
    assert posting.location_text == "Muscat"
    # A jobs board, and the publisher named the employer in a machine-readable
    # property. Both matter: `poster_type='unknown'` is INELIGIBLE for
    # aggregation under migration 0006, so an ad that reaches here as unknown is
    # stored and never counted — the trap dubizzle sits in.
    assert posting.listing_intent == "vacancy"
    assert posting.poster_type == "company"


def test_the_description_is_html_and_is_flattened_to_text():
    posting = build(base_pages(OMAN_AD)).fetch().postings[0]
    assert "<p>" not in posting.raw_description
    assert "Lead the region." in posting.raw_description
    assert "P&L" in posting.raw_description       # entities decoded


def test_attribution_travels_on_every_row():
    posting = build(base_pages(OMAN_AD)).fetch().postings[0]
    assert posting.attribution == "GulfTalent"
    assert posting.terms_url == "https://www.gulftalent.com/terms"


@pytest.mark.parametrize("value,want", [
    (["FULL_TIME"], "full_time"),
    ("PART_TIME", "part_time"),
    (["CONTRACTOR"], "contract"),
    (["INTERN"], "internship"),
    # Vocabulary we do not model. None, never a plausible-looking default.
    (["VOLUNTEER"], None),
    (["SOMETHING_NEW"], None),
    (None, None),
])
def test_employment_type_maps_or_stays_silent(value, want):
    pages = base_pages(OMAN_AD, page=ad_page(employmentType=value))
    assert build(pages).fetch().postings[0].employment_type == want


def test_a_single_salary_figure_is_not_widened_into_a_range():
    """The publisher stated one number. Recording it as an open range would
    widen a claim they did not make."""
    pages = base_pages(OMAN_AD, page=ad_page(baseSalary={
        "@type": "MonetaryAmount", "currency": "OMR",
        "value": {"@type": "QuantitativeValue", "value": 900, "unitText": "MONTH"}}))
    posting = build(pages).fetch().postings[0]
    assert (posting.salary_min, posting.salary_max) == (900.0, 900.0)


def test_a_backwards_salary_range_is_refused_rather_than_stored():
    """Migration 0011's CHECK would reject the row and lose the whole posting.
    An impossible range is an error at the source, so drop the salary and keep
    the vacancy."""
    pages = base_pages(OMAN_AD, page=ad_page(baseSalary={
        "@type": "MonetaryAmount", "currency": "OMR",
        "value": {"@type": "QuantitativeValue", "minValue": 900, "maxValue": 100}}))
    posting = build(pages).fetch().postings[0]
    assert posting.salary_min is None and posting.salary_max is None


def test_no_salary_stated_stays_null():
    pages = base_pages(OMAN_AD, page=ad_page(baseSalary=None))
    posting = build(pages).fetch().postings[0]
    assert posting.salary_min is None and posting.salary_currency is None


# ---------------------------------------------------------------------------
# failing loudly
# ---------------------------------------------------------------------------
def test_a_sitemap_with_no_ad_submap_fails_rather_than_reporting_empty():
    pages = {"https://www.gulftalent.com/sitemap.xml":
             SITEMAP_INDEX.replace("sitemap_jx000", "sitemap_zz000")}
    result = build(pages).fetch()
    assert result.error and "sub-sitemap" in result.error
    assert not result.may_age_inventory


def test_no_oman_urls_fails_rather_than_ageing_the_inventory():
    """"Zero Oman ads today" is far more likely to mean the URL shape moved than
    that a Gulf-wide board listed nothing for Oman. Reporting it as a clean empty
    fetch would age every row we hold toward deletion."""
    result = build(base_pages("https://www.gulftalent.com/uae/jobs/x-1234")).fetch()
    assert result.error and "URL shape" in result.error
    assert not result.may_age_inventory


def test_ad_pages_without_json_ld_fail_rather_than_reporting_empty():
    """The anchor-miss guard: pages loaded, none parsed, so the layout moved."""
    pages = base_pages(OMAN_AD, page="<html><body><h1>Area Manager</h1></body></html>")
    result = build(pages).fetch()
    assert result.error and "JobPosting" in result.error


def test_one_unreachable_ad_costs_one_ad():
    urls = [OMAN_AD, "https://www.gulftalent.com/oman/jobs/analyst-604519"]
    pages = base_pages(*urls)
    del pages[urls[1]]                       # that one 404s
    result = build(pages).fetch()
    assert [p.source_url for p in result.postings] == [OMAN_AD]
    assert result.skipped == 1
    assert result.error is None


# ---------------------------------------------------------------------------
# the warm cycle — a politeness property, not an optimisation
# ---------------------------------------------------------------------------
def test_a_known_posting_is_not_fetched_again():
    """430 ads at a 12-hour cadence is 860 requests a day. The site throttles SEO
    crawlers to one request per 30 seconds; taking that much without needing to
    would be the rudest thing this crawler does."""
    urls = [OMAN_AD, "https://www.gulftalent.com/oman/jobs/analyst-604519"]
    adapter = build(base_pages(*urls), is_known_unchanged=lambda p: p.source_url == OMAN_AD)
    result = adapter.fetch()

    assert OMAN_AD not in adapter._client.fetched
    assert [p.source_url for p in result.postings] == [urls[1]]


def test_a_skipped_posting_is_still_reported_so_it_can_be_touched():
    """THE bug this would otherwise be. A posting the adapter never emits is
    never touched, `age_missed` counts it missing, and three cycles later the
    source has aged its own unchanged inventory toward deletion."""
    from agents.agent_b_job_ingest.hashing import posting_id

    adapter = build(base_pages(OMAN_AD), is_known_unchanged=lambda p: True)
    result = adapter.fetch()

    assert result.postings == []
    assert result.seen_unchanged_ids == [posting_id("gulftalent", OMAN_AD)]


def test_a_capped_fetch_does_not_claim_to_be_a_census():
    urls = [OMAN_AD, "https://www.gulftalent.com/oman/jobs/analyst-604519"]
    result = build(base_pages(*urls)).fetch(limit=1)
    assert len(result.postings) == 1
    assert result.truncated is True
    assert not result.may_age_inventory


def test_an_ad_with_no_named_employer_is_not_claimed_to_be_one():
    """No `hiringOrganization` means the publisher did not say. `unknown` is the
    honest answer, and it costs the row its place in the demand statistics —
    which is the correct price for not knowing."""
    pages = base_pages(OMAN_AD, page=ad_page(hiringOrganization=None))
    posting = build(pages).fetch().postings[0]
    assert posting.company is None
    assert posting.poster_type == "unknown"


def test_the_stated_employer_reaches_the_persisted_row():
    """The publisher's `hiringOrganization` must survive the pipeline, which
    otherwise takes `company` from the LLM alone. Without this the employer is
    discarded, poster_type never becomes 'company', and every row this source
    produces is stored and never counted."""
    from agents.agent_b_job_ingest.pipeline import IngestPipeline
    from agents.agent_b_job_ingest.prompts.extraction import EXTRACTION_PROMPT
    from agents.agent_b_job_ingest.prompts.legitimacy import LEGITIMACY_PROMPT
    from agents.agent_b_job_ingest.schemas import JobExtractionBatch, LegitimacyVerdict
    from shared.llm import structured
    from tests.agent_b.fake_store import FakeStore
    from tests.fake_embedder import FakeEmbedder
    from tests.fake_llm import FakeStructuredLLM

    store = FakeStore()
    llm = FakeStructuredLLM()
    pipe = IngestPipeline(
        store=store,
        extractor=EXTRACTION_PROMPT | structured(llm, JobExtractionBatch),
        adjudicator=LEGITIMACY_PROMPT | structured(llm, LegitimacyVerdict),
        embedder=FakeEmbedder(),
        config=CFG,
        model_name="fake",
    )
    pipe.run(build(base_pages(OMAN_AD)).fetch().postings)

    row = next(iter(store.rows.values()))
    assert row.company == "Al Toobi New Enterprises"
    assert row.poster_type == "company"
    assert row.listing_intent == "vacancy"
    # And the publisher's structured facts survive too.
    assert row.employment_type == "full_time"
    assert (row.salary_min, row.salary_max) == (650.0, 800.0)
    assert row.attribution == "GulfTalent"
