"""The ingestion tail: a batch of RawPostings in, rows written out.

This is the linear sequence the plan calls dedup -> legitimacy -> extract ->
embed -> neardup -> upsert. Phase 6 wraps it in the LangGraph fan-out and the run
log; here it is a plain, testable function so the phase gate — "a second run does
zero LLM and zero embedding work" — can be asserted without a graph.

The ordering is not arbitrary; each step exists to stop the next one doing work
it does not need to:

  * change detection first, so an unchanged posting costs one UPDATE and nothing
    else — no LLM, no embedding. This is the property that makes a 12-hour cycle
    cheap once the corpus is warm.
  * legitimacy before extraction, so a posting the rules reject outright costs no
    extraction and no embedding at all. Rejected rows are still written (audit),
    just empty of everything expensive.
  * link dedup before embedding, so a Telegram repost that links to its blog
    original is resolved deterministically and never embedded.

Every expensive call is counted in the returned summary, because "did the second
run really skip the work?" is answerable only if the work is counted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional
from urllib.parse import urlsplit

from shared.grounding import normalize, verify_quote

from . import legitimacy as legit
from .hashing import child_source_url, content_hash, id_for, posting_id
from .records import PersistedPosting
from .root_fetch import candidate_job_link
from .schemas import JobExtraction, JobExtractionBatch, LegitimacyVerdict
from .sources.base import RawPosting
from .stated_facts import verify_stated_facts


@dataclass
class IngestSummary:
    """Counters the phase gate and the run log both read."""

    received: int = 0
    unchanged: int = 0
    changed: int = 0
    new: int = 0
    rejected: int = 0
    needs_review: int = 0
    link_duplicates: int = 0
    destination_duplicates: int = 0
    embed_duplicates: int = 0
    extractions: int = 0
    adjudications: int = 0
    embeddings: int = 0
    written: int = 0
    # Root-source enrichment: pages fetched, and postings whose skills were
    # replaced by the real job page's.
    root_fetches: int = 0
    root_enrichments: int = 0
    # Fields the model asserted and the source could not support, dropped to
    # NULL. NOT an error count — it is the verifier working. A cycle where this
    # is 0 across hundreds of postings means the check stopped running, which is
    # exactly the failure that would otherwise be invisible.
    unstated_fields_dropped: int = 0
    # Postings whose extraction raised — a malformed model response, a rejected
    # sector, an API error. One posting each, never a batch. Counted so a spike
    # is visible: a handful is the model being the model, a hundred means the
    # prompt or the schema has drifted.
    extraction_failures: int = 0
    # Postings a `require_destination` source produced that resolved to no
    # employer page, and were therefore never written. Counted and printed every
    # cycle: this is the number that says how much a source is actually
    # contributing under the rule, and a silent 180 would look identical to a
    # silent 0.
    skipped_no_destination: int = 0
    # Split vacancies that kept the whole post body because the model's quoted
    # span could not be verified against it. Counted rather than hidden: a run
    # where this equals the split count means the prompt stopped working, and
    # the rows would be silently clustered again.
    unsliced_vacancies: int = 0

    def as_dict(self) -> dict[str, int]:
        return {k: v for k, v in self.__dict__.items()}

    def merge(self, other: "IngestSummary") -> "IngestSummary":
        """Accumulate another batch's counters into this one.

        The cycle processes each source's postings in its own transaction, so the
        run-level totals are the sum across batches."""
        for key, value in other.__dict__.items():
            setattr(self, key, getattr(self, key) + value)
        return self


@dataclass
class _Work:
    """One VACANCY in flight through the tail — a single source post may fan out
    into several of these when it is a multi-job roundup."""

    raw: RawPosting
    posting_id: str
    content_hash: str
    row: PersistedPosting
    # Set once dedup resolves it; suppresses embedding.
    is_link_duplicate: bool = False
    # True when this row is one of several split out of a multi-vacancy post. Such
    # a row has no per-job outbound link (the links belong to the whole post), so
    # link-dedup skips it and relies on essence near-dup instead.
    is_split: bool = False


class IngestPipeline:
    def __init__(
        self,
        *,
        store: Any,
        extractor: Any,
        adjudicator: Any,
        embedder: Any,
        config: Any,
        model_name: str = "unknown",
        root_fetcher: Any = None,
        source_configs: Any = (),
    ) -> None:
        self.store = store
        self.extractor = extractor          # prompt | structured(llm, JobExtractionBatch)
        self.adjudicator = adjudicator      # prompt | structured(llm, LegitimacyVerdict)
        self.embedder = embedder
        # Optional: fetches a posting's root job page for real skills. None
        # disables root enrichment (offline tests, dry runs).
        self.root_fetcher = root_fetcher
        self.config = config
        self.model_name = model_name
        # The source registry, INJECTED rather than imported.
        #
        # `require_destination` and `link_back_required` are properties of a
        # named production source, and a pipeline constructed directly — every
        # offline test — must not inherit them from whatever fixture name it
        # happened to pick. Defaulting to empty means a test using
        # `source="el7far"` as a stand-in gets stand-in behaviour; production
        # reaches this through `GraphDeps.pipeline()`, which passes the real
        # registry.
        self.source_configs = tuple(source_configs or ())

    # ------------------------------------------------------------------
    def run(self, postings: list[RawPosting]) -> IngestSummary:
        summary = IngestSummary(received=len(postings))
        self._unstated = 0
        self._extraction_errors: list[str] = []
        self._unsliced = 0

        deduped = _dedupe_within_batch(postings)
        # Change detection is at the POST level: one source post may split into
        # several vacancy rows, so "has this changed?" is decided once, by the
        # post's content_hash, before re-extracting the vacancies it contains.
        stored = self.store.lookup_posts([p.source_url for p in deduped])

        unchanged_ids: list[str] = []
        to_extract: list[tuple[RawPosting, str]] = []

        for raw in deduped:
            post_url = raw.source_url
            post_hash = content_hash(raw)
            entry = stored.get(post_url)
            if entry and entry["content_hash"] == post_hash:
                # Unchanged post: touch all the rows it produced last cycle.
                unchanged_ids.extend(entry["posting_ids"])
                continue
            summary.new += post_url not in stored
            summary.changed += post_url in stored
            to_extract.append((raw, post_hash))

        summary.unchanged = self.store.touch_seen(unchanged_ids)

        # legitimacy gate (on the whole POST) then extraction -> one _Work per
        # vacancy. A rejected post yields a single audit row, never a split.
        work: list[_Work] = []
        for raw, post_hash in to_extract:
            assessment, band, verdict = self._assess(raw, summary)

            # Rejecting a MULTI-VACANCY post is a much heavier act than rejecting a
            # single one: the score is computed over the whole text, so one
            # "registration fee" line in a footer — common on aggregator pages —
            # condemns every genuine vacancy alongside it, and this module's own
            # docstring notes that a wrongly-rejected posting is invisible by
            # construction (it is excluded from the very statistics anyone would
            # use to notice). The evidence is also weaker per vacancy: a signal
            # drawn from text about job 7 says little about job 2.
            #
            # So a roundup is held for review rather than deleted. needs_review is
            # not aggregable, so it still cannot reach the demand statistics —
            # the posting is quarantined, not trusted, and not destroyed.
            roundup = "mass_vacancy_no_detail" in assessment.codes
            if band == "reject" and roundup:
                item = self._rejected_work(raw, post_hash, assessment)
                item.row.status = "needs_review"
                item.row.review_reason = (
                    f"multi-vacancy post scored as a scam on whole-post evidence "
                    f"({', '.join(assessment.codes)}); held for review rather than "
                    f"rejecting every vacancy it advertises"
                )
                work.append(item)
                summary.needs_review += 1
                continue

            if band == "reject":
                work.append(self._rejected_work(raw, post_hash, assessment))
                summary.rejected += 1
                continue

            # The adjudicator said "scam" but the blended score did not clear the
            # reject line. That used to be discarded entirely: the row was written
            # `active` with no reason recorded, so a model call we paid for and a
            # judgement it stated plainly had no effect whatsoever — it needed
            # scam_confidence >= 0.70 to matter, while the schema's default is 0.5.
            # Flagging keeps the posting for a human and, because needs_review is
            # not aggregable, out of the demand statistics meanwhile.
            flagged = None
            if verdict is not None and verdict.is_scam:
                flagged = (
                    f"adjudicator: {verdict.reasoning.strip()} "
                    f"(confidence {verdict.scam_confidence:.2f}; "
                    f"cited: {verdict.evidence_quote.strip()!r})"
                )

            # ONE posting's extraction, inside its own boundary.
            #
            # The batch already had a boundary (in `nodes.make_ingest`), but it
            # wraps the WHOLE source, so any failure here cost every posting in
            # the cycle. Measured 2026-08-08: a 352-posting GulfTalent census
            # died on ONE model response with country="null" — 367 pages
            # fetched politely over 20 minutes, 352 extractions paid for, zero
            # rows written.
            #
            # The audit's own P4 said "one bad posting must cost one posting".
            # It was true of the batch and not of the posting. It is now both.
            try:
                jobs = self._extract_jobs(raw, summary)
            except Exception as exc:      # noqa: BLE001 - deliberate boundary
                # Includes the pydantic ValidationError that `_known_sector` and
                # `_iso_alpha2` raise BY DESIGN on bad model output. Refusing the
                # value is right; refusing the other 351 postings is not.
                summary.extraction_failures += 1
                self._extraction_errors.append(f"{raw.source_url}: {type(exc).__name__}: {exc}")
                continue

            for index, job in enumerate(jobs):
                item = self._child_work(raw, post_hash, job, index, len(jobs), assessment)
                if flagged:
                    item.row.status = "needs_review"
                    item.row.review_reason = flagged
                    summary.needs_review += 1
                work.append(item)

        live = [w for w in work if w.row.status != "rejected"]

        # root-source enrichment — replace thin aggregator skills with the real
        # job page's, before embedding so the essence reflects the true skills.
        self._enrich_from_root(live, summary)

        # Destination dedup FIRST: it is the strongest signal we have, and it is
        # exact. Two aggregators summarising one vacancy differently defeat
        # embedding similarity, but they cannot disagree about which page they
        # link to. Runs after _enrich_from_root because that is what sets it.
        self._resolve_destination_duplicates(live, summary)

        # link dedup — deterministic, before any embedding
        self._resolve_link_duplicates(live, summary)

        # embedding — skips rejected (already excluded from `live`) and link dups.
        # With no embedder (--no-embed) the embedding and near-dup steps are
        # skipped entirely; link dedup still runs, but similarity-based near-dup
        # detection is disabled, which the CLI help states plainly.
        if self.embedder is not None:
            to_embed = [w for w in live if not w.is_link_duplicate]
            self._embed(to_embed, summary)
            self._resolve_neardup(to_embed, work, summary)

        # duplicate_of must point at a ROOT before anything is written.
        self._flatten_duplicate_chains(work)

        # A source that only wants final destinations does not store a posting
        # that has none. THE ENFORCEMENT POINT for that decision: deleting such
        # rows afterwards buys twelve hours, because the next cycle sees the
        # posts as new and writes every one of them back.
        #
        # Applied here, after root enrichment has had its chance, so the test is
        # "did this posting resolve to a vacancy page?" rather than "did it come
        # with a link?". A rejected posting is not written at all — not stored
        # as rejected, not stored as needs_review; it simply does not enter the
        # corpus, which is what "I only want final destinations" asks for.
        rows = [w.row for w in work]
        if rows:
            keep, refused = [], 0
            for row in rows:
                if self._requires_destination(row.source) and not row.final_url:
                    refused += 1
                    continue
                keep.append(row)
            summary.skipped_no_destination = refused
            rows = keep

        self.store.upsert_batch(rows)
        summary.written = len(rows)
        summary.unstated_fields_dropped = self._unstated
        summary.unsliced_vacancies = self._unsliced
        # Surfaced on the pipeline so the ingest node can warn with the actual
        # URLs. A counter alone tells an operator something broke; the URLs tell
        # them what, which is the difference between a number and a lead.
        self.extraction_errors = list(self._extraction_errors)
        return summary

    def _requires_destination(self, source: str) -> bool:
        """Does this source only contribute postings that reach a vacancy page?

        User decision 2026-08-09: "I only want final destinations." Measured on
        el7far, that means ~5 postings a cycle instead of ~180 — the rest are
        articles saying "send your CV to hr@..." with nothing to follow.
        """
        for cfg in self.source_configs:
            if cfg.name == source:
                return cfg.require_destination
        return False

    # ------------------------------------------------------------------
    def _base_row(self, raw: RawPosting, pid: str, src_url: str, chash: str,
                  title: str) -> PersistedPosting:
        return PersistedPosting(
            posting_id=pid,
            source=raw.source,
            source_group=raw.source_group,
            source_type=raw.source_type,
            source_url=src_url,
            source_post_url=raw.source_url,
            title=title,
            raw_description=raw.raw_description,
            content_hash=chash,
            posted_date=raw.posted_date.date() if raw.posted_date else None,
            listing_intent=raw.listing_intent,
            poster_type=raw.poster_type,
            extraction_model=self.model_name,
            # Who published it and on what terms. A row must be able to say this
            # for as long as it exists, independently of the source registry.
            attribution=raw.attribution,
            terms_url=raw.terms_url,
            expires_at=raw.expires_at,
        )

    def _rejected_work(self, raw: RawPosting, chash: str, assessment: Any) -> _Work:
        """A post the legitimacy gate rejected — one audit row for the WHOLE post,
        never split (there is nothing worth splitting a scam into)."""
        pid = id_for(raw)
        row = self._base_row(raw, pid, raw.source_url, chash, raw.title)
        row.legitimacy_score = assessment.score
        row.status = "rejected"
        row.review_reason = f"legitimacy: {', '.join(assessment.codes) or 'adjudicated'}"
        return _Work(raw=raw, posting_id=pid, content_hash=chash, row=row)

    def _child_work(self, raw: RawPosting, chash: str, job: JobExtraction,
                    index: int, total: int, assessment: Any) -> _Work:
        """One vacancy -> one row. A single-vacancy post keeps the post's own URL
        and id (byte-identical to before); a split vacancy gets post_url#role so
        each is a distinct, stable key under UNIQUE (source, source_url)."""
        is_split = total > 1
        title = (job.title or "").strip() or raw.title
        if is_split:
            src_url = child_source_url(raw.source_url, job.title, index)
            pid = posting_id(raw.source, src_url)
        else:
            src_url = raw.source_url
            pid = id_for(raw)

        row = self._base_row(raw, pid, src_url, chash, title)
        # A vacancy split out of a roundup keeps only its OWN text. Measured
        # before this: every child of all 40 el7far roundups stored the entire
        # article, so a row titled "Accounts Payable In-Charge" carried 20,000
        # characters describing 38 unrelated jobs.
        #
        # Only when the post actually split. A single-vacancy post's body IS
        # about that vacancy, and slicing it could only lose context.
        if is_split:
            slice_ = _verified_slice(job, raw.raw_description, title)
            if slice_:
                row.raw_description = slice_
            else:
                self._unsliced += 1
        row.legitimacy_score = assessment.score
        row.sector = job.sector
        row.required_skills = job.required_skills
        # Source-stated wins, model fills the silence. A publisher naming the
        # employer in a STRUCTURED field (schema.org `hiringOrganization`) is
        # better evidence than a model finding a name in prose, and it is not
        # subject to the text-grounding check below for the same reason: the
        # employer's name legitimately appears nowhere in the description.
        row.company = raw.company or job.company
        # Source-stated wins, model fills the silence — the same rule as the
        # facts below and as listing_intent/poster_type above.
        row.seniority_level = raw.seniority_level or job.seniority_level
        row.location = job.location
        # Source-stated wins. GulfTalent files every ad under a country in its
        # own URL and repeats it in `addressCountry`; the model reads prose and
        # frequently states nothing, which would leave `country` NULL — and
        # `export_for_agent_c` filters on country, so those rows would be stored
        # and never retrieved.
        row.country = raw.country or job.country
        # Verified, not taken on trust. Measured on the first live cycle: 19 of
        # 19 postings came back 'onsite' and NONE of the 19 contained an
        # arrangement phrase in either language — the model was reading a city
        # name, exactly as the prompt tells it not to. See `stated_facts.py`.
        #
        # A value the SOURCE stated in a structured field is a different thing
        # and takes precedence: GulfTalent's `employmentType: ["FULL_TIME"]` is
        # the publisher's own schema.org property, not an interpretation, and
        # putting it through a text-mention check would discard it because the
        # description prose never repeats the phrase. Same rule `listing_intent`
        # and `poster_type` already follow — the model may only fill what the
        # source left silent.
        facts = verify_stated_facts(job, raw.raw_description)
        row.work_arrangement = raw.work_arrangement or facts.work_arrangement
        row.employment_type = raw.employment_type or facts.employment_type
        row.salary_min = raw.salary_min if raw.salary_min is not None else facts.salary_min
        row.salary_max = raw.salary_max if raw.salary_max is not None else facts.salary_max
        row.salary_currency = raw.salary_currency or facts.salary_currency
        row.salary_period = raw.salary_period or facts.salary_period
        self._unstated += len(facts.dropped)

        # Ground the employer the same way Agent A grounds a skill span: a company
        # the model named must actually appear in the posting, or it is a
        # hallucination. Not a stored column, but it gates poster_type below — a
        # claim of 'company' is only trustworthy if a real employer name backs it.
        grounded_company = (
            job.company
            if job.company and normalize(job.company) in normalize(raw.raw_description)
            else None
        )
        # Extraction reads the content, so it may refine intent/poster the adapter
        # left unknown — but never overwrites a value the source already stated.
        if raw.listing_intent == "unknown":
            row.listing_intent = job.listing_intent
        if raw.poster_type == "unknown":
            if job.poster_type == "company" and grounded_company is None:
                row.poster_type = "unknown"
            else:
                row.poster_type = job.poster_type

        return _Work(raw=raw, posting_id=pid, content_hash=chash, row=row, is_split=is_split)

    # ---- legitimacy --------------------------------------------------
    def _assess(self, raw: RawPosting, summary: IngestSummary) -> tuple[Any, str, Any]:
        """Score the WHOLE post for legitimacy, before any split. employer_
        extracted=False: extraction has not run yet, so the no_employer_named
        signal must stay silent (firing it here scores every posting employer-
        less); the phrase/contact/role signals all read from the raw text.

        Returns ``(assessment, band, verdict)`` — the verdict is the adjudicator's
        answer when one was sought AND its quote checked out, so the caller can act
        on the model's judgement and record the evidence behind it.
        """
        assessment = legit.score_text(
            raw.raw_description, company=None, title=raw.title, employer_extracted=False,
        )
        # The module's stated backstop, now actually called: a scorer bug that
        # attaches a confident number to evidence that is not in the posting fails
        # loudly here instead of being published. Cheap (a substring check per
        # quoted signal) and it only fires on a genuine defect.
        legit.assert_auditable(assessment, legit.audit_source(raw.title, raw.raw_description))
        band = assessment.band(self.config)
        verdict = None
        if band == "adjudicate":
            assessment, verdict = self._adjudicate(raw, assessment, summary)
            band = assessment.band(self.config)
        return assessment, band, verdict

    def _adjudicate(
        self, raw: RawPosting, assessment: Any, summary: IngestSummary
    ) -> tuple[Any, Any]:
        """One LLM call for a post the rules could not resolve. Trusted only if
        its quote is real; a fabricated quote is discarded and the rule score
        stands, so the model cannot reject on evidence it cannot produce.

        Returns ``(assessment, verdict)``; verdict is None when the model could not
        produce a real quote, because an unevidenced verdict must not travel.
        """
        summary.adjudications += 1
        verdict: LegitimacyVerdict = self.adjudicator.invoke({"body": raw.raw_description})
        if not verify_quote(verdict.evidence_quote, raw.raw_description):
            return assessment, None  # ungrounded — rules stand
        if verdict.is_scam:
            # Blend: the stronger of rule risk and model confidence.
            risk = max(assessment.risk, verdict.scam_confidence)
            return legit.Assessment(risk=risk, signals=assessment.signals), verdict
        # Cleared with real evidence: relax toward legitimate but never below the
        # band floor, so one clearance cannot whitewash strong rule signals.
        return legit.Assessment(risk=min(assessment.risk, 0.35), signals=assessment.signals), verdict

    # ---- extraction --------------------------------------------------
    def _extract_jobs(self, raw: RawPosting, summary: IngestSummary) -> list[JobExtraction]:
        """Extract the vacancies in one post. ONE call per post (so the warm-cycle
        gate still holds), returning one JobExtraction per distinct vacancy — a
        list of one for an ordinary posting, several for a roundup. Tolerates an
        extractor that returns a bare JobExtraction (the offline test harness) as
        well as a JobExtractionBatch (the live runner)."""
        summary.extractions += 1
        result = self.extractor.invoke({"title": raw.title, "body": raw.raw_description})
        jobs = list(result.jobs) if isinstance(result, JobExtractionBatch) else [result]
        # A post the model read nothing usable out of is still one row (thin, but
        # honestly thin), exactly as a single empty extraction was before.
        return jobs or [JobExtraction()]

    # ---- root-source enrichment -------------------------------------
    def _enrich_from_root(self, live: list[_Work], summary: IngestSummary) -> None:
        """Replace a posting's aggregator skills with the real job page's.

        Bounded + robots-respecting (the fetcher enforces both): follow a link
        ONLY when it is a single external job-detail page; re-extract it; enrich
        ONLY when that page is a SINGLE job (several jobs means it was a hub —
        skip, never split external content). Enrich-only: a failed fetch, a hub,
        or an empty page leaves the aggregator's skills untouched. Split children
        are skipped — their post's links are hubs, not per-role pages.
        """
        if self.root_fetcher is None or not self.config.enrich_from_root_source:
            return

        budget = self.config.max_root_fetches_per_cycle
        for item in live:
            # A per-cycle ceiling, not just a per-host interval. The docstrings
            # called this "bounded" while the only limits were one-per-posting and
            # a 1s gap — i.e. unbounded in the dimension that matters when a batch
            # is large.
            if budget is not None and summary.root_fetches >= budget:
                break
            if item.is_split:
                continue
            self_host = urlsplit(item.raw.source_url).netloc.lower()
            link = candidate_job_link(item.raw.outbound_links, self_host)
            if not link:
                continue

            text = self.root_fetcher.fetch(link)
            summary.root_fetches += 1
            if not text:
                continue

            result = self.extractor.invoke({"title": item.raw.title, "body": text})
            jobs = list(result.jobs) if isinstance(result, JobExtractionBatch) else [result]
            # Exactly one job = a real single job page. More = a listing/hub we do
            # not deep-crawl, and a hub is NOT where anyone applies — so it never
            # becomes a final_url.
            if len(jobs) != 1:
                continue

            # The URL is recorded even when the page yields no skills. It resolved
            # to one vacancy, so it IS the destination; "we learned nothing from
            # it" and "it is not the place to apply" are different statements, and
            # only the second should withhold the link.
            #
            # UNLESS the publisher's terms require linking back to them. Then the
            # employer's page is not ours to send people to: GulfTalent permits
            # this crawl only on condition that each snippet links back to the
            # corresponding GulfTalent page, and quietly redirecting the apply
            # button to the employer would take their content AND their traffic.
            # The skills are still harvested below — reading the destination is
            # not the same act as replacing the link to it.
            if not _links_back_to_source(item.raw, self.source_configs or None):
                item.row.final_url = link

            if not jobs[0].required_skills:
                continue

            # Verified against the DESTINATION's text, not the aggregator's —
            # this is the page that states these things, so it is the page that
            # has to support them.
            self._merge_root(item.row, jobs[0], text)
            summary.root_enrichments += 1

    def _merge_root(self, row: PersistedPosting, job: JobExtraction,
                    source_text: str = "") -> None:
        """Root page is authoritative for the job's facts: its skills REPLACE the
        aggregator's, and each scalar fact is taken from the root when it states
        one, else the aggregator's value is kept."""
        row.required_skills = job.required_skills
        row.sector = job.sector or row.sector
        row.company = job.company or row.company
        row.seniority_level = job.seniority_level or row.seniority_level
        row.location = job.location or row.location
        row.country = job.country or row.country

        # The fields the employer's page states and an aggregator summary almost
        # never does. Same rule as the scalars above — the root wins when it says
        # something, and silence never overwrites — but only AFTER the page has
        # been made to support the claim.
        facts = verify_stated_facts(job, source_text)
        self._unstated += len(facts.dropped)

        row.work_arrangement = facts.work_arrangement or row.work_arrangement
        row.employment_type = facts.employment_type or row.employment_type
        # `is not None`, not `or`: a salary of 0 is falsy, and an unpaid
        # internship stating 0 is a real answer that must not be read as silence.
        if facts.salary_min is not None:
            row.salary_min = facts.salary_min
        if facts.salary_max is not None:
            row.salary_max = facts.salary_max
        row.salary_currency = facts.salary_currency or row.salary_currency
        row.salary_period = facts.salary_period or row.salary_period

    # ---- destination dedup -------------------------------------------
    def _resolve_destination_duplicates(self, live: list[_Work],
                                        summary: IngestSummary) -> None:
        """Two postings pointing at one employer page are one vacancy.

        Cheaper and more certain than the embedding near-dup this supplements:
        the evidence is an exact URL rather than a similarity above a threshold,
        and it catches the case similarity is worst at — the same job written up
        twice, in different words, by two different aggregators.

        A stored canonical row wins over anything in this batch; within the batch
        the lowest posting_id wins, so the outcome does not depend on the order
        sources happened to be fetched in.
        """
        candidates = [w for w in live
                      if w.row.final_url and not w.is_link_duplicate]
        if not candidates:
            return

        stored = self.store.find_by_final_urls(
            sorted({w.row.final_url for w in candidates}))

        # Deterministic winner per destination inside this batch.
        canonical: dict[str, str] = {}
        for item in sorted(candidates, key=lambda w: w.posting_id):
            canonical.setdefault(item.row.final_url, item.posting_id)

        for item in candidates:
            target = stored.get(item.row.final_url) or canonical[item.row.final_url]
            if target and target != item.posting_id:
                item.row.duplicate_of = target
                item.is_link_duplicate = True   # also skips embedding, as link dups do
                summary.destination_duplicates += 1

    # ---- link dedup --------------------------------------------------
    def _resolve_link_duplicates(self, live: list[_Work], summary: IngestSummary) -> None:
        """Resolve a posting that links to another posting's canonical URL.

        Direction rule: the linker is the duplicate, the link target is
        canonical. Deterministic — the shared URL is the evidence, no embedding
        needed. Only the FIRST outbound link is treated as the subject; later
        links are unrelated (a Telegram post appends an unrelated next job), so
        merging on any link would collapse two real vacancies.

        Split vacancies are excluded: the outbound links belong to the WHOLE post,
        not to any one of the roles split out of it, so attributing them to a
        child would merge every sibling into the post's link target. Splits rely
        on essence near-dup instead.
        """
        candidates = [w for w in live if not w.is_split]
        # url -> posting_id, from the store and from this very batch.
        in_batch = {w.row.source_url: w.posting_id for w in candidates}
        all_links = [w.raw.outbound_links[0] for w in candidates if w.raw.outbound_links]
        stored = self.store.find_by_source_urls(all_links) if all_links else {}

        for item in candidates:
            if not item.raw.outbound_links:
                continue
            target_url = item.raw.outbound_links[0]
            target_id = stored.get(target_url) or in_batch.get(target_url)
            if target_id and target_id != item.posting_id:
                item.row.duplicate_of = target_id
                item.is_link_duplicate = True
                summary.link_duplicates += 1

    # ---- embedding ---------------------------------------------------
    def _embed(self, items: list[_Work], summary: IngestSummary) -> None:
        """Embed the ESSENCE of each posting — title, skills, seniority,
        location — never the full description.

        Learned from the first full live run, the hard way: this source's
        descriptions share ~20K chars of template (application instructions,
        closing dates, page furniture), so full-description embeddings put a CFO
        and a CEO at the same company above 0.95 similarity and auto-merge
        collapsed ~28 distinct vacancies into other jobs — undercounting real
        demand, the exact fabrication class this agent must not commit. The
        extracted essence is what makes a posting THIS job; it is also the right
        retrieval surface for Agent C, whose queries are candidate skills, not
        boilerplate. Embedding runs after extraction precisely so this text can
        be built from the extracted fields.
        """
        if not items:
            return
        vectors = self.embedder.embed_documents([_essence_text(w) for w in items])
        if len(vectors) != len(items):
            raise RuntimeError("embedder returned a different number of vectors than texts")
        summary.embeddings += len(items)
        for item, vector in zip(items, vectors):
            item.row.embedding = list(vector)

    # ---- near-dup ----------------------------------------------------
    def _resolve_neardup(
        self, embedded: list[_Work], all_work: list[_Work], summary: IngestSummary
    ) -> None:
        """Classify each embedded posting against recent postings.

        Candidate set is the union of the store and this cycle, so ordering
        within the fan-out does not matter. In-group similarity >= 0.93 auto-
        merges; cross-group >= 0.97 flags for human review but NEVER auto-merges
        — a wrong cross-publisher merge erases a real, independent demand signal.
        In-group wins when both fire.
        """
        recent = self.config.neardup_recent_days
        in_group_t = self.config.neardup_in_group_threshold
        cross_t = self.config.neardup_cross_group_threshold
        k = self.config.neardup_candidates

        for item in embedded:
            if item.row.duplicate_of is not None:
                continue  # already a link duplicate

            best_in_group: tuple[float, str] | None = None
            best_cross: tuple[float, str] | None = None

            # in-cycle candidates (canonicals only, and not itself)
            for other in embedded:
                if other is item or other.row.embedding is None:
                    continue
                if other.row.duplicate_of is not None:
                    continue
                sim = _cosine(item.row.embedding, other.row.embedding)
                same_group = other.raw.source_group == item.raw.source_group
                # Cross-group review does not set duplicate_of, so two identical
                # in-cycle postings would otherwise BOTH flag each other. Break
                # the symmetry deterministically: only the larger id flags
                # itself, so the pair produces exactly one review and the other
                # stays active — keeping the legitimate side aggregable.
                allow_cross = item.posting_id > other.posting_id
                best_in_group, best_cross = _accumulate(
                    best_in_group, best_cross, sim, other.posting_id,
                    same_group=same_group, allow_cross=allow_cross,
                )

            # stored candidates — already committed, so the NEW posting always
            # flags against them; no symmetry to break.
            for cand in self.store.find_neardup_candidates(
                item.row.embedding, recent_days=recent, limit=k, exclude_id=item.posting_id
            ):
                sim = float(cand["similarity"])
                best_in_group, best_cross = _accumulate(
                    best_in_group, best_cross, sim, cand["posting_id"],
                    same_group=cand["source_group"] == item.raw.source_group, allow_cross=True,
                )

            if best_in_group and best_in_group[0] >= in_group_t:
                item.row.duplicate_of = best_in_group[1]
                summary.embed_duplicates += 1
            elif best_cross and best_cross[0] >= cross_t:
                item.row.status = "needs_review"
                item.row.review_reason = "cross_group_duplicate"
                summary.needs_review += 1


    # ---- root resolution -----------------------------------------------
    def _flatten_duplicate_chains(self, work: list[_Work]) -> None:
        """Rewrite every ``duplicate_of`` to point at a ROOT.

        The near-dup pass processes items in order, so A can merge into B before
        B itself merges into C — leaving A pointing at another duplicate. That is
        wrong twice over: semantically (``duplicate_of`` means "the canonical",
        not "the next hop"), and operationally — the upsert sorts canonicals
        before duplicates, so A can hit the FK before B exists. The FIRST FULL
        LIVE RUN failed on exactly this, which is why the flattening is a
        separate, unconditional pass rather than an invariant each merge site is
        trusted to maintain.

        Cycle-safe: a mutual loop (which no current merge path should produce,
        but this code must outlive that assumption) is broken deterministically
        by promoting the smallest posting_id in the loop to canonical.
        """
        by_id = {w.posting_id: w for w in work}

        for item in work:
            if item.row.duplicate_of is None:
                continue
            trail = [item.posting_id]
            target = item.row.duplicate_of
            # Depth-capped: a chain longer than the batch is definitionally a cycle.
            for _ in range(len(work) + 1):
                node = by_id.get(target)
                if node is None or node.row.duplicate_of is None:
                    break  # a stored id, or an in-batch canonical — a real root
                if target in trail:
                    canonical = min(trail + [target])
                    for pid in trail + [target]:
                        member = by_id.get(pid)
                        if member is not None:
                            member.row.duplicate_of = None if pid == canonical else canonical
                    target = canonical
                    break
                trail.append(target)
                target = node.row.duplicate_of
            item.row.duplicate_of = None if target == item.posting_id else target


# ---------------------------------------------------------------------------
# An absolute floor, deliberately low.
#
# It started at 80 on the theory that anything shorter was the model echoing the
# title back. Measured against the real corpus, that rejected 37 of 38 CORRECT
# answers: a Majees-style roundup is a LIST of role titles, so all the article
# says about one vacancy is
#
#     Project Coordinator - ELV
#     - Muscat, Oman
#
# — 40 characters, and genuinely the whole of it. A floor tuned to an imagined
# failure was discarding the source's actual shape. The real guard is not length
# but `_adds_to_title` below: a span has to say something the row does not
# already know.
MIN_SLICE_CHARS = 25
# And a "slice" that is essentially the whole article has sliced nothing. The
# failure it guards against is the model returning the input unchanged, which
# would leave every child clustered while reporting success.
MAX_SLICE_RATIO = 0.9


def _adds_to_title(span: str, title: str) -> bool:
    """Does this span say anything beyond the role title the row already has?

    "Project Coordinator - ELV" alone is not a description, it is the title
    twice. "Project Coordinator - ELV / Muscat, Oman" is — it adds where the job
    is. Compared on normalised text so punctuation and spacing do not decide it.
    """
    from shared.grounding import normalize

    if not title:
        return True
    return len(normalize(span)) > len(normalize(title)) + 5


def _verified_slice(job: JobExtraction, body: str, title: str = "") -> Optional[str]:
    """The text of ONE vacancy, if the model quoted it and the posting contains it.

    Returns None whenever the slice cannot be trusted, and every caller then
    keeps the full body. That asymmetry is deliberate: this runs over 245 live
    rows, and the worst outcome it can produce is the status quo.

    Four ways to fail, in order of how often they fire:
      * nothing quoted (a single-vacancy post — the normal case);
      * too short to be a description rather than an echoed title;
      * so long it is the article again, i.e. no slicing happened;
      * not actually in the posting — a paraphrase, which is the one that
        matters. `verify_quote` is the same check `evidence_quote` goes through,
        for the same reason: the model may POINT at text, never compose it.
    """
    span = (getattr(job, "vacancy_text", None) or "").strip()
    if not span or len(span) < MIN_SLICE_CHARS:
        return None
    # A span that is only the role title tells the row nothing it does not
    # already store in `title`. This is the check the length floor was reaching
    # for and getting wrong — "is it informative?", not "is it long?".
    if not _adds_to_title(span, title or (job.title or "")):
        return None
    if len(span) > len(body) * MAX_SLICE_RATIO:
        return None
    if not verify_quote(span, body):
        return None
    # `verify_quote` has confirmed this text is in the posting under
    # normalisation (whitespace, Arabic orthography, digit forms). Stored as the
    # model returned it: re-deriving raw offsets from a normalised match is
    # fiddly and buys nothing a reader would notice.
    return span


def _links_back_to_source(raw: RawPosting, source_configs: Any = None) -> bool:
    """Must this posting's apply link stay on the publisher's own page?

    True when the source was configured `link_back_required` — a terms
    condition, not a preference. Read from the registry so that changing the
    obligation is one edit in one place, and so a row already written cannot
    disagree with the terms now in force.
    """
    from .sources.config import DEFAULT_SOURCES

    for cfg in (source_configs if source_configs is not None else DEFAULT_SOURCES):
        if cfg.name == raw.source:
            return cfg.link_back_required
    return False


def _dedupe_within_batch(postings: list[RawPosting]) -> list[RawPosting]:
    """Same URL twice in one cycle collapses to one, keeping the first seen."""
    seen: set[str] = set()
    out: list[RawPosting] = []
    for posting in postings:
        pid = id_for(posting)
        if pid not in seen:
            seen.add(pid)
            out.append(posting)
    return out


def _essence_text(item: _Work) -> str:
    """The differentiating content of a posting, post-extraction.

    Title first (it names the job), then the scalar facts, then the skills. A
    posting whose extraction found nothing (a Telegram teaser) embeds as its
    title alone — thin, but honestly thin, and those mostly resolve by link
    before ever reaching the embedder anyway.
    """
    row = item.row
    # The row's title (the per-role title for a split vacancy), not the raw
    # post title — otherwise every role split out of one roundup shares a title
    # line and near-dup would wrongly merge distinct vacancies.
    parts = [(row.title or item.raw.title or "").strip()]
    scalars = " ".join(p for p in (row.seniority_level, row.location, row.country) if p)
    if scalars:
        parts.append(scalars)
    if row.required_skills:
        parts.append("skills: " + ", ".join(row.required_skills))
    return "\n".join(p for p in parts if p)


def _cosine(a: list[float], b: list[float]) -> float:
    """Dot product. Both vectors are unit-normalised by the embedder, so this IS
    cosine similarity — no division, and no chance of the distance/similarity
    inversion that the SQL path guards against separately."""
    return sum(x * y for x, y in zip(a, b))


def _accumulate(best_in_group, best_cross, sim, pid, *, same_group, allow_cross=True):
    if same_group:
        if best_in_group is None or sim > best_in_group[0]:
            best_in_group = (sim, pid)
    elif allow_cross:
        if best_cross is None or sim > best_cross[0]:
            best_cross = (sim, pid)
    return best_in_group, best_cross
