"""Operator backfills — repairs to rows already in the table.

A normal cycle only touches postings a source hands it, gated on `content_hash`,
which is what keeps a warm cycle cheap. That gate also means an improvement to
the pipeline never reaches rows already stored. These commands close that gap for
two specific defects, both measured on the live corpus, and both bounded so a
mistake costs a field rather than a corpus.

**`reslice_roundups`** — every child of a multi-vacancy post stored the WHOLE
article. Measured: all 40 el7far roundups, 245 rows, 164 pinned at the 20,000
character cap, 4.1 MB of duplicated text, and one distinct description per
roundup. The source text is already in the database (that IS the defect), so
this needs no network at all: re-extract from what is stored, and write each
vacancy its own verified span.

**`backfill_destinations`** — `final_url` only ever gets set during enrichment,
which only runs for new or changed posts, so rows ingested before that shipped
have none. Outbound links were never stored, so this one does re-fetch the
article before following the link.

Both are **narrowing operations by construction**: they replace a field with a
verified substring of itself, or set one that was NULL. Neither can delete a
posting, and neither writes text the source did not publish.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from shared.config import Config

from shared.grounding import normalize

from .pipeline import _verified_slice
from .schemas import JobExtraction, JobExtractionBatch


@dataclass
class BackfillReport:
    """What a backfill did, in the terms an operator would ask about."""

    posts_considered: int = 0
    rows_updated: int = 0
    rows_unchanged: int = 0
    # Vacancies whose quoted span did not verify against the post. NOT an error:
    # the row keeps the full body, which is exactly what it had before. Reported
    # because a run where this equals the row count means the prompt broke.
    unverified: int = 0
    # Employers recovered for rows that had none. A roundup names its employer
    # in the header, so narrowing a vacancy to its own line would otherwise
    # lose it — the one thing this repair could plausibly take away.
    employers_recorded: int = 0
    # destination_status -> count. The audit trail for a prune: it answers
    # "why did these rows go?" in the terms the decision was made in.
    outcomes: dict[str, int] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {k: (len(v) if isinstance(v, list) else v)
                for k, v in self.__dict__.items()}


# ---------------------------------------------------------------------------
# R2 — give each split vacancy its own text
# ---------------------------------------------------------------------------
def reslice_roundups(
    *,
    store: Any,
    extractor: Any,
    sources: Optional[list[str]] = None,
    limit: Optional[int] = None,
    dry_run: bool = False,
    config: Optional[Config] = None,
) -> BackfillReport:
    """Re-extract stored roundups and narrow each child to its own vacancy text.

    Matched by ``posting_id``, which is ``sha(source, post_url#role-slug)`` and
    therefore stable across re-extraction: a vacancy whose title the model
    phrases differently this time simply does not match, and its row is LEFT
    ALONE. That is the deliberate choice — this command exists to narrow a text
    field, so it must not be able to remove a posting or re-mint an id.
    """
    report = BackfillReport()
    config = config or Config()

    for post in store.clustered_posts(sources=sources, limit=limit):
        report.posts_considered += 1
        body = post["body"]
        stored_ids = set(post["posting_ids"])

        # Already narrowed by an earlier run (or by a cycle since). Idempotent
        # by observation rather than by a flag column: if no child still holds
        # the whole body, there is nothing to do.
        if not post["needs_reslice"]:
            report.rows_unchanged += len(stored_ids)
            continue

        try:
            result = extractor.invoke({"title": _headline(body), "body": body})
        except Exception as exc:      # noqa: BLE001 - one post must cost one post
            report.failures.append(f"{post['source_post_url']}: {type(exc).__name__}: {exc}")
            continue

        jobs = list(result.jobs) if isinstance(result, JobExtractionBatch) else [result]
        by_title = _rows_by_title(post)
        without_employer = {c["posting_id"] for c in (post.get("children_rows") or [])
                            if not c.get("company")}
        updates: dict[str, str] = {}
        employers: dict[str, str] = {}
        for index, job in enumerate(jobs):
            pid = _match_row(post, job, index, len(jobs), by_title)
            if pid is None or pid not in stored_ids:
                # A vacancy this post did not previously yield — a different
                # split. Nothing to update, and creating a row here would mint
                # postings from a repair command.
                continue

            # Narrowing a row's text should not cost it the employer, which a
            # roundup names once in its header and not in each vacancy's line.
            # Measured: only 76 of 245 clustered rows carried one, while 227
            # asserted poster_type='company' — the name was read at ingest and
            # never stored. Grounded against the POST body for the same reason
            # the pipeline grounds it there: that is where a roundup says it.
            if pid in without_employer and job.company:
                if normalize(job.company) in normalize(body):
                    employers[pid] = job.company

            span = _verified_slice(job, body, job.title or "")
            if span is None:
                report.unverified += 1
                continue
            updates[pid] = span

        if not dry_run:
            if updates:
                store.narrow_descriptions(updates)
            for pid, name in employers.items():
                store.update_posting(pid, {"company": name})
        report.rows_updated += len(updates)
        report.employers_recorded += len(employers)
        report.rows_unchanged += len(stored_ids) - len(updates)

    return report


def _headline(body: str) -> str:
    """The article's own title, which is its first non-empty line.

    A roundup's headline is stored nowhere: its children keep their ROLE titles.
    Handing one of those back to the extractor tells it the article is about
    that single job — measured, a 19-vacancy post came back with one vacancy
    named after whichever child sorted first.
    """
    for line in (body or "").splitlines():
        if line.strip():
            return line.strip()[:300]
    return ""


def _rows_by_title(post: dict[str, Any]) -> dict[str, list[str]]:
    """Stored children keyed by normalised role title."""
    from shared.grounding import normalize

    out: dict[str, list[str]] = {}
    for child in post.get("children_rows") or []:
        key = normalize(child.get("title") or "")
        if key:
            out.setdefault(key, []).append(child["posting_id"])
    return out


def _match_row(post: dict[str, Any], job: JobExtraction, index: int, total: int,
               by_title: dict[str, list[str]]) -> Optional[str]:
    """Which stored row is this re-extracted vacancy?

    **By title first, position second.** The minted id is
    `sha(source, post_url#slug-index)`, so matching on it requires the model to
    return the same vacancies in the same ORDER — finding E3 of the Agent B
    audit, and far too brittle to rest a repair on. The role title is the stable
    thing, so it decides; the id rule is the fallback for a title that is absent
    or ambiguous.

    An ambiguous title (two stored rows normalising the same) matches NOTHING
    rather than guessing. Writing one vacancy's text onto another's row is worse
    than leaving both as they are.
    """
    from shared.grounding import normalize

    key = normalize(job.title or "")
    candidates = by_title.get(key, [])
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        return None
    return _child_id(post, job, index, total)


def _child_id(post: dict[str, Any], job: JobExtraction, index: int, total: int) -> str:
    """The posting_id this vacancy would have, by the same rule that minted it.

    ALWAYS the split-child rule. These rows were selected because
    `source_post_url <> source_url`, so they are children by definition — and a
    re-extraction that happens to return one vacancy this time must not be
    matched against the single-posting id, which belongs to a row that does not
    exist.

    The fragment is `slug-index`, so a match needs the same title AND the same
    position. That is finding E3 of the Agent B audit ("split-vacancy IDs depend
    on LLM output order") inherited rather than solved: a reordered extraction
    simply does not match, and the row is left exactly as it is. Skipping is the
    safe direction; re-keying rows from a repair command is not.
    """
    from .hashing import child_source_url, posting_id

    return posting_id(post["source"],
                      child_source_url(post["source_post_url"], job.title, index))


# ---------------------------------------------------------------------------
# R3 — follow the destinations that were never followed
# ---------------------------------------------------------------------------
def backfill_destinations(
    *,
    store: Any,
    extractor: Any,
    root_fetcher: Any,
    article_fetch: Any,
    sources: Optional[list[str]] = None,
    limit: Optional[int] = None,
    dry_run: bool = False,
    config: Optional[Config] = None,
) -> BackfillReport:
    """Follow each stored posting's outbound link to the employer's own page.

    Same path a live cycle takes — `candidate_job_link` to choose the link,
    `RootFetcher` to fetch it (robots fail-closed, refused hosts remembered),
    the extractor to read it — so a backfilled row is indistinguishable from one
    enriched at ingest. The difference is only where the input rows come from.

    Outbound links were never persisted, so the article is re-fetched to recover
    them. Measured yield on el7far: **14%** — most of its articles describe the
    job themselves and give an email address, with no ATS link to follow.
    """
    from .pipeline import _links_back_to_source
    from .root_fetch import candidate_job_link
    from .sources.base import RawPosting
    from .stated_facts import verify_stated_facts

    report = BackfillReport()
    config = config or Config()

    from urllib.parse import urlsplit

    def record(pid: str, status: str) -> None:
        """Why this row has no destination — see migration 0013.

        Written even on failure, and that is the point: a prune that deletes
        rows for lacking a destination has to be able to say WHICH failure each
        row hit, months later, from the database rather than from a terminal
        that has scrolled away.
        """
        report.outcomes[status] = report.outcomes.get(status, 0) + 1
        if not dry_run:
            store.update_posting(pid, {"destination_status": status})

    # Roundup children included: when postings are about to be DELETED for
    # lacking a destination, every row deserves its one attempt first.
    for row in store.rows_without_destination(sources=sources, limit=limit,
                                             single_vacancy_only=False):
        report.posts_considered += 1

        # A vacancy SPLIT out of a roundup has no destination of its own, and
        # this needs no fetch to know: the article carries ONE outbound link
        # belonging to the whole roundup, so handing it to each child claims
        # that every one of 19 roles is advertised at the same URL.
        #
        # Measured after doing exactly that: 49 of 55 surviving rows shared a
        # destination, and "Business Development Manager" pointed at
        # `.../jobs/senior-lowcode-developer-381`. That is worse than having no
        # destination — it is a false claim about where to apply, which is the
        # one thing this whole change exists to prevent. `_enrich_from_root`
        # has always skipped split children for this reason.
        if row.get("is_split"):
            record(row["posting_id"], "hub")
            report.rows_unchanged += 1
            continue

        try:
            links = article_fetch(row["source_url"])
        except Exception as exc:      # noqa: BLE001
            report.failures.append(f"{row['source_url']}: {type(exc).__name__}: {exc}")
            record(row["posting_id"], "unreachable")
            continue

        link = candidate_job_link(tuple(links), urlsplit(row["source_url"]).netloc.lower())
        if not link:
            # No link at all vs a link we decline to follow are different facts
            # about the source, and the prune's audit trail needs both. Measured
            # on 30 el7far postings: 16 no link, 11 hub-or-social.
            external = [h for h in links
                        if urlsplit(h).netloc.lower()
                        and urlsplit(h).netloc.lower() != urlsplit(row["source_url"]).netloc.lower()]
            record(row["posting_id"], "hub" if external else "no_link")
            report.rows_unchanged += 1
            continue

        # A publisher whose terms require linking back keeps its own page as the
        # apply link. Enriching the SKILLS from the destination is still fine —
        # reading a page and redirecting traffic away from the publisher are
        # different acts, and only the second is what the condition forbids.
        record_url = not _links_back_to_source(
            RawPosting(source=row["source"], source_group=row["source"],
                       source_type=row["source_type"], source_url=row["source_url"],
                       title=row["title"], raw_description=row["title"]))

        text = root_fetcher.fetch(link)
        if not text:
            # robots refused it, or it would not load. `RootFetcher` folds both
            # into None; `blocked` on the fetcher separates them at run level.
            record(row["posting_id"], "unreachable")
            report.rows_unchanged += 1
            continue

        try:
            result = extractor.invoke({"title": row["title"], "body": text})
        except Exception as exc:      # noqa: BLE001
            report.failures.append(f"{link}: {type(exc).__name__}: {exc}")
            continue

        jobs = list(result.jobs) if isinstance(result, JobExtractionBatch) else [result]
        # Exactly one vacancy means a real job page. Several means a hub, and a
        # hub is not where anyone applies.
        if len(jobs) != 1:
            record(row["posting_id"], "hub")     # a listing, not a vacancy
            report.rows_unchanged += 1
            continue

        job = jobs[0]
        facts = verify_stated_facts(job, text)
        update: dict[str, Any] = {}
        if record_url:
            update["final_url"] = link
        if job.required_skills:
            update["required_skills"] = job.required_skills
        for name in ("work_arrangement", "employment_type", "salary_currency",
                     "salary_period"):
            value = getattr(facts, name)
            if value:
                update[name] = value
        for name in ("salary_min", "salary_max"):
            value = getattr(facts, name)
            if value is not None:      # 0 is a real answer
                update[name] = value
        if job.seniority_level:
            update["seniority_level"] = job.seniority_level

        if not update:
            record(row["posting_id"], "hub")
            report.rows_unchanged += 1
            continue
        update["destination_status"] = "resolved" if record_url else "source_is_destination"
        report.outcomes[update["destination_status"]] =             report.outcomes.get(update["destination_status"], 0) + 1
        if not dry_run:
            store.update_posting(row["posting_id"], update)
        report.rows_updated += 1

    return report
