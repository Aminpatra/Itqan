"""A read before a transaction must not silently discard everything that follows.

This pins a bug that reported perfect success and lost all of it.

Wiring `is_known_unchanged` into production (so a source costing one request per
posting does not re-fetch 430 ads every cycle) meant an adapter now queries the
database DURING the scrape phase — before the cycle's first `transaction()`
block. psycopg opens an implicit transaction on the first statement of a
connection, so that read left one open. Every later `with conn.transaction()`
then nested inside it as a SAVEPOINT, and releasing a savepoint commits nothing.

The cycle logged `written=25`, raised nothing, warned about nothing, and the
table was empty afterwards. There is no louder failure available to notice: the
run log said the work was done.

The fix is a dedicated autocommit connection for scrape-phase reads, which is
also what makes them safe when LangGraph fans several scrape branches out
concurrently.
"""

from __future__ import annotations

import psycopg
import pytest

from agents.agent_b_job_ingest.db.store import JobStore
from agents.agent_b_job_ingest.records import PersistedPosting


def _row(pid: str, url: str) -> PersistedPosting:
    return PersistedPosting(
        posting_id=pid, source="probe", source_group="probe",
        source_type="html_scrape", source_url=url,
        title="Probe", raw_description="body", content_hash="h" * 32,
        listing_intent="vacancy", poster_type="company", country="OM",
    )


def test_a_scrape_read_does_not_leave_a_transaction_open(store: JobStore):
    """The mechanism itself. A read on the scrape path must leave the MAIN
    connection untouched, so the next `transaction()` is a real transaction
    rather than a savepoint inside an implicit one."""
    main = store.connect()
    assert main.info.transaction_status == psycopg.pq.TransactionStatus.IDLE

    store.exists_during_scrape(["nothing" * 4])

    assert main.info.transaction_status == psycopg.pq.TransactionStatus.IDLE, (
        "a scrape-phase read opened a transaction on the cycle's main "
        "connection; every subsequent transaction() becomes a savepoint and "
        "commits nothing"
    )


def test_writes_survive_a_scrape_read_that_came_first(store: JobStore, migrated_dsn: str):
    """The consequence, end to end and in the real order a cycle runs in:
    read during scraping, then write inside a transaction, then close."""
    store.exists_during_scrape(["a" * 32])          # the scrape-phase read

    with store.transaction():
        store.upsert_batch([_row("s" * 32, "https://probe.test/jobs/1")])
    store.close()                                    # as the runner does

    with psycopg.connect(migrated_dsn) as check:
        count = check.execute(
            "SELECT count(*) FROM job_postings WHERE posting_id = %s", ("s" * 32,)
        ).fetchone()[0]
    assert count == 1, (
        "the write was rolled back — a scrape-phase read had left an implicit "
        "transaction open on the same connection"
    )


def test_the_scrape_read_actually_answers_the_question(store: JobStore):
    """It must be correct as well as safe, or the warm cycle either re-fetches
    everything or skips postings it has never seen."""
    with store.transaction():
        store.upsert_batch([_row("k" * 32, "https://probe.test/jobs/known")])

    assert store.exists_during_scrape(["k" * 32]) == {"k" * 32}
    assert store.exists_during_scrape(["u" * 32]) == set()
    assert store.exists_during_scrape([]) == set()


def test_it_is_usable_from_several_threads(store: JobStore):
    """Scrape branches fan out concurrently via `Send`, and a psycopg connection
    is not safe for concurrent use — which is the second reason this method
    exists rather than reusing the main one."""
    import threading

    with store.transaction():
        store.upsert_batch([_row("t" * 32, "https://probe.test/jobs/threaded")])

    errors: list[Exception] = []
    results: list[set[str]] = []

    def probe() -> None:
        try:
            results.append(store.exists_during_scrape(["t" * 32]))
        except Exception as exc:            # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=probe) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"concurrent scrape reads raised: {errors[:2]}"
    assert results == [{"t" * 32}] * 8


# ---------------------------------------------------------------------------
# the general form of the same bug
# ---------------------------------------------------------------------------
def test_any_read_before_a_transaction_still_lets_writes_commit(store: JobStore,
                                                                migrated_dsn: str):
    """The scrape lookup was not the only caller that read first.

    A `destination_survey` taken before a prune did exactly the same thing: the
    command printed "deleted 432 posting(s)" and 432 rows were still there,
    because the DELETE ran inside a savepoint of an implicit transaction nobody
    committed. Fixing it per-caller was clearly losing the argument, so the
    connection is autocommit and `transaction()` opens a real transaction.

    This test is deliberately about ANY read, not about scraping, because the
    next caller to trip it will not be a scraper either.
    """
    store.counts()                                    # a plain read, no transaction

    with store.transaction():
        store.upsert_batch([_row("g" * 32, "https://probe.test/jobs/general")])

    with store.transaction():
        deleted = store.purge_source("probe")
    store.close()

    with psycopg.connect(migrated_dsn) as check:
        left = check.execute(
            "SELECT count(*) FROM job_postings WHERE posting_id = %s", ("g" * 32,)
        ).fetchone()[0]
    assert deleted == 1 and left == 0, (
        "a write after a plain read was rolled back — the connection is holding "
        "an implicit transaction and every transaction() is only a savepoint"
    )


def test_a_failed_transaction_still_rolls_its_own_work_back(store: JobStore,
                                                            migrated_dsn: str):
    """Autocommit must not cost the batch guarantee it was chosen to protect:
    a source that fails part-way still writes nothing."""
    try:
        with store.transaction():
            store.upsert_batch([_row("r" * 32, "https://probe.test/jobs/rollback")])
            raise RuntimeError("something went wrong mid-batch")
    except RuntimeError:
        pass
    store.close()

    with psycopg.connect(migrated_dsn) as check:
        left = check.execute(
            "SELECT count(*) FROM job_postings WHERE posting_id = %s", ("r" * 32,)
        ).fetchone()[0]
    assert left == 0, "a failed transaction committed its partial work"
