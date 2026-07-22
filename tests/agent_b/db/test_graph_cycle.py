"""The whole cycle graph against real Postgres, with fake adapters.

Two properties are the phase gate:

  * isolation — a normal cycle ingests, writes a run log, and a second identical
    cycle does no expensive work and adds no rows.
  * mid-cycle block — a source that fetches partially still has its postings
    ingested, is NOT aged (so its live jobs are not pushed toward deletion), and
    is recorded as a failure; the cycle reports itself partial.
"""

from __future__ import annotations

import json

from agents.agent_b_job_ingest.graph import build_ingest_graph
from tests.agent_b.graph_helpers import BLOG, TG, FakeAdapter, make_deps, raw


def _initial(tmp_path, **kw):
    state = {"run_id": "cycle1", "output_dir": str(tmp_path), "source_names": [],
             "limit": None, "dry_run": False}
    state.update(kw)
    return state


def test_a_normal_cycle_ingests_and_writes_a_run_log(store, tmp_path):
    adapters = {
        "blog": FakeAdapter("blog", postings=[
            raw("https://blog.test/a.html", source="blog", group="g1", stype="blogger_feed"),
            raw("https://blog.test/b.html", source="blog", group="g1", stype="blogger_feed",
                title="Data Analyst role"),
        ]),
        "tg": FakeAdapter("tg", postings=[
            raw("https://t.me/testchannel/1", source="tg", group="g1", stype="telegram",
                links=("https://blog.test/a.html",)),  # links to the blog post
        ]),
    }
    deps, _ = make_deps(store, adapters)
    graph = build_ingest_graph(deps)

    out = graph.invoke(_initial(tmp_path))

    assert out["ingest_summary"]["written"] == 3
    assert out["partial_cycle"] is False
    # The telegram post linked to the blog post → resolved by link, not embedded.
    assert out["ingest_summary"]["link_duplicates"] == 1

    # Run log written and self-consistent.
    log = json.loads((tmp_path / "cycle1" / "ingest_cycle.json").read_text(encoding="utf-8"))
    assert {s["source"] for s in log["sources"]} == {"blog", "tg"}
    assert log["signals"]["embeddings_skipped_link_dup"] == 1


def test_a_second_identical_cycle_does_no_work(store, tmp_path):
    adapters = {
        "blog": FakeAdapter("blog", postings=[
            raw("https://blog.test/a.html", source="blog", group="g1", stype="blogger_feed"),
        ]),
        "tg": FakeAdapter("tg", postings=[]),
    }
    deps, _ = make_deps(store, adapters)
    graph = build_ingest_graph(deps)

    graph.invoke(_initial(tmp_path))
    deps2, llm2 = make_deps(store, adapters)
    out = build_ingest_graph(deps2).invoke(_initial(tmp_path, run_id="cycle2"))

    assert out["ingest_summary"]["unchanged"] == 1
    assert out["ingest_summary"]["extractions"] == 0
    assert out["ingest_summary"]["embeddings"] == 0
    assert llm2.calls == []


def test_a_source_with_no_postings_still_ages_its_inventory(store, tmp_path):
    """An empty day for a HEALTHY source means its old inventory should age."""
    # First cycle: blog has one posting.
    a = {"blog": FakeAdapter("blog", postings=[
            raw("https://blog.test/a.html", source="blog", group="g1", stype="blogger_feed")]),
         "tg": FakeAdapter("tg", postings=[])}
    build_ingest_graph(make_deps(store, a)[0]).invoke(_initial(tmp_path))

    # Second cycle: blog now returns nothing (posting gone) — it should age.
    b = {"blog": FakeAdapter("blog", postings=[]), "tg": FakeAdapter("tg", postings=[])}
    build_ingest_graph(make_deps(store, b)[0]).invoke(_initial(tmp_path, run_id="c2"))

    conn = store.connect()
    with conn.cursor() as cur:
        cur.execute("SELECT missed_cycles FROM job_postings WHERE source='blog'")
        assert cur.fetchone()["missed_cycles"] == 1


# ---------------------------------------------------------------------------
# mid-cycle block
# ---------------------------------------------------------------------------
def test_a_partial_source_is_ingested_but_not_aged_and_marked_failed(store, tmp_path):
    # Seed: blog has a posting from a prior clean cycle.
    seed = {"blog": FakeAdapter("blog", postings=[
                raw("https://blog.test/old.html", source="blog", group="g1", stype="blogger_feed")]),
            "tg": FakeAdapter("tg", postings=[])}
    build_ingest_graph(make_deps(store, seed)[0]).invoke(_initial(tmp_path))

    # Now blog fetches PARTIALLY: it returns a new posting but was blocked.
    blocked = {
        "blog": FakeAdapter("blog", partial=True, error="429 mid-stream", postings=[
            raw("https://blog.test/new.html", source="blog", group="g1", stype="blogger_feed",
                title="A brand new role")]),
        "tg": FakeAdapter("tg", postings=[]),
    }
    out = build_ingest_graph(make_deps(store, blocked)[0]).invoke(_initial(tmp_path, run_id="c2"))

    # (a) the cycle knows it was partial
    assert out["partial_cycle"] is True

    # (b) the partial source's new posting WAS ingested
    conn = store.connect()
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) AS n FROM job_postings WHERE source_url='https://blog.test/new.html'")
        assert cur.fetchone()["n"] == 1

        # (c) the OLD posting was NOT aged — a partial fetch must not push live
        # inventory toward deletion because of a 429.
        cur.execute("SELECT missed_cycles FROM job_postings WHERE source_url='https://blog.test/old.html'")
        assert cur.fetchone()["missed_cycles"] == 0, "a blocked source aged its own inventory"

    # (d) health recorded the failure
    health = store.get_source_health("blog")
    assert health["consecutive_failures"] == 1
    assert health["last_error"] == "429 mid-stream"


def test_the_run_log_reports_a_partial_cycle(store, tmp_path):
    blocked = {
        "blog": FakeAdapter("blog", partial=True, error="blocked", postings=[]),
        "tg": FakeAdapter("tg", postings=[]),
    }
    build_ingest_graph(make_deps(store, blocked)[0]).invoke(_initial(tmp_path))

    log = json.loads((tmp_path / "cycle1" / "ingest_cycle.json").read_text(encoding="utf-8"))
    assert log["partial_cycle"] is True
    blog = [s for s in log["sources"] if s["source"] == "blog"][0]
    assert blog["ok"] is False and blog["partial"] is True
