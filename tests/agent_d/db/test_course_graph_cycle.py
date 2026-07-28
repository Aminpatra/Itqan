"""The whole course cycle graph against real Postgres, with fake adapters.

Isolation (a second identical cycle does no work, adds no rows) and the
mid-cycle block (a partial source is ingested but not aged, and reported failed)
are the phase gate — same properties Agent B's graph proves.
"""

from __future__ import annotations

import json
from pathlib import Path

from agents.agent_b_job_ingest.esco_map import sync_taxonomy
from agents.agent_d_course_ingest.graph import build_course_ingest_graph
from agents.agent_d_course_ingest.nodes import GraphDeps
from agents.agent_d_course_ingest.prompts.extraction import EXTRACTION_PROMPT
from agents.agent_d_course_ingest.schemas import CourseExtraction
from agents.agent_d_course_ingest.sources.base import AdapterResult, RawCourse
from agents.agent_d_course_ingest.sources.config import CourseSourceConfig
from shared.config import Config
from shared.llm import structured
from tests.fake_embedder import FakeEmbedder
from tests.fake_llm import FakeStructuredLLM

ESCO_FIXTURE = Path(__file__).parents[2] / "agent_b" / "fixtures" / "esco_skills_sample.csv"

COURSERA = CourseSourceConfig(name="coursera", source_group="coursera", source_type="api",
                              base_url="https://api.coursera.org", terms_reviewed=True,
                              census=False)
FCC = CourseSourceConfig(name="freecodecamp", source_group="freecodecamp",
                         source_type="html_scrape", base_url="https://www.freecodecamp.org",
                         census=True)


def raw(url, *, source, group, stype, name="Accounting Basics"):
    return RawCourse(source=source, source_group=group, source_type=stype, source_url=url,
                     name=name, raw_description="A course on accounting.", provider="IBM",
                     primary_language="en")


class FakeAdapter:
    def __init__(self, name, *, courses, error=None, partial=False, truncated=False):
        self.name, self._courses, self._error, self._partial = name, courses, error, partial
        self._truncated = truncated

    def fetch(self, *, limit=None):
        r = AdapterResult(source=self.name)
        r.courses = list(self._courses)[:limit] if limit else list(self._courses)
        r.pages_fetched = 1
        r.truncated = self._truncated
        if self._error is not None:
            r.error, r.partial = self._error, self._partial
        return r


def make_deps(store, adapters):
    llm = FakeStructuredLLM(CourseExtraction=CourseExtraction(taught_skills=["accounting"]))
    return GraphDeps(
        config=Config(database_url=store.dsn), store=store,
        extractor=EXTRACTION_PROMPT | structured(llm, CourseExtraction),
        embedder=FakeEmbedder(), model_name="fake",
        source_configs=(COURSERA, FCC),
        adapter_for=lambda cfg, config: adapters[cfg.name],
    )


def _initial(tmp_path, **kw):
    s = {"run_id": "c1", "output_dir": str(tmp_path), "source_names": [], "limit": None, "dry_run": False}
    s.update(kw); return s


def test_a_normal_cycle_ingests_maps_and_writes_a_run_log(store, tmp_path):
    from agents.agent_b_job_ingest.db import JobStore
    with JobStore(store.dsn) as jb:
        sync_taxonomy(jb, FakeEmbedder(), path=ESCO_FIXTURE, version="test-1")

    adapters = {
        "coursera": FakeAdapter("coursera", courses=[
            raw("https://www.coursera.org/learn/a", source="coursera", group="coursera", stype="api"),
        ]),
        "freecodecamp": FakeAdapter("freecodecamp", courses=[
            raw("https://www.freecodecamp.org/learn/b", source="freecodecamp",
                group="freecodecamp", stype="html_scrape", name="Relational Database"),
        ]),
    }
    out = build_course_ingest_graph(make_deps(store, adapters)).invoke(_initial(tmp_path))

    assert out["ingest_summary"]["written"] == 2
    assert out["partial_cycle"] is False
    # accounting was mapped and aggregated
    assert out["aggregation_summary"]["rows_written"] >= 1
    log = json.loads((tmp_path / "c1" / "course_cycle.json").read_text(encoding="utf-8"))
    assert {s["source"] for s in log["sources"]} == {"coursera", "freecodecamp"}


def test_a_second_identical_cycle_does_no_work(store, tmp_path):
    adapters = {
        "coursera": FakeAdapter("coursera", courses=[
            raw("https://www.coursera.org/learn/a", source="coursera", group="coursera", stype="api")]),
        "freecodecamp": FakeAdapter("freecodecamp", courses=[]),
    }
    build_course_ingest_graph(make_deps(store, adapters)).invoke(_initial(tmp_path))
    out = build_course_ingest_graph(make_deps(store, adapters)).invoke(_initial(tmp_path, run_id="c2"))
    assert out["ingest_summary"]["unchanged"] == 1
    assert out["ingest_summary"]["extractions"] == 0
    assert out["ingest_summary"]["embeddings"] == 0


def test_a_partial_source_is_ingested_but_not_aged_and_marked_failed(store, tmp_path):
    seed = {"coursera": FakeAdapter("coursera", courses=[
                raw("https://www.coursera.org/learn/old", source="coursera", group="coursera", stype="api")]),
            "freecodecamp": FakeAdapter("freecodecamp", courses=[])}
    build_course_ingest_graph(make_deps(store, seed)).invoke(_initial(tmp_path))

    blocked = {
        "coursera": FakeAdapter("coursera", partial=True, error="429 mid-stream", courses=[
            raw("https://www.coursera.org/learn/new", source="coursera", group="coursera",
                stype="api", name="New Course")]),
        "freecodecamp": FakeAdapter("freecodecamp", courses=[]),
    }
    out = build_course_ingest_graph(make_deps(store, blocked)).invoke(_initial(tmp_path, run_id="c2"))

    assert out["partial_cycle"] is True
    conn = store.connect()
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) AS n FROM courses WHERE source_url='https://www.coursera.org/learn/new'")
        assert cur.fetchone()["n"] == 1
        cur.execute("SELECT missed_cycles FROM courses WHERE source_url='https://www.coursera.org/learn/old'")
        assert cur.fetchone()["missed_cycles"] == 0, "a blocked source aged its own inventory"
    assert store.get_source_health("coursera")["consecutive_failures"] == 1


# ---------------------------------------------------------------------------
# census: whether absence from a fetch may age a course toward deletion
# ---------------------------------------------------------------------------
def _missed(store, url: str) -> int:
    with store.connect().cursor() as cur:
        cur.execute("SELECT missed_cycles FROM courses WHERE source_url = %s", (url,))
        return cur.fetchone()["missed_cycles"]


def test_a_non_census_source_never_ages_what_this_slice_did_not_contain(store, tmp_path):
    """Coursera is 23,101 courses read 800 at a time. A course missing from one
    cycle's slice has not been withdrawn — we simply did not page to it. Ageing
    it is how `--once --limit 20` came to delete the corpus.
    """
    old = "https://www.coursera.org/learn/old"
    seed = {"coursera": FakeAdapter("coursera", courses=[
                raw(old, source="coursera", group="coursera", stype="api")]),
            "freecodecamp": FakeAdapter("freecodecamp", courses=[])}
    build_course_ingest_graph(make_deps(store, seed)).invoke(_initial(tmp_path))

    # A perfectly healthy, complete fetch — that simply does not include `old`.
    later = {"coursera": FakeAdapter("coursera", courses=[
                raw("https://www.coursera.org/learn/new", source="coursera",
                    group="coursera", stype="api", name="New Course")]),
             "freecodecamp": FakeAdapter("freecodecamp", courses=[])}
    out = build_course_ingest_graph(make_deps(store, later)).invoke(_initial(tmp_path, run_id="c2"))

    assert out["partial_cycle"] is False, "the fetch itself was healthy"
    assert _missed(store, old) == 0, "a sampled source aged a course it never looked for"
    assert "coursera" in out["ageing"]["not_aged"]


def test_a_census_source_does_age_what_it_no_longer_lists(store, tmp_path):
    """The other half: freeCodeCamp's index is read in full every cycle, so a
    course's absence really is a withdrawal, and staleness must still work."""
    gone = "https://www.freecodecamp.org/learn/retired"
    seed = {"coursera": FakeAdapter("coursera", courses=[]),
            "freecodecamp": FakeAdapter("freecodecamp", courses=[
                raw(gone, source="freecodecamp", group="freecodecamp",
                    stype="html_scrape", name="Retired Cert")])}
    build_course_ingest_graph(make_deps(store, seed)).invoke(_initial(tmp_path))
    assert _missed(store, gone) == 0

    still = {"coursera": FakeAdapter("coursera", courses=[]),
             "freecodecamp": FakeAdapter("freecodecamp", courses=[
                 raw("https://www.freecodecamp.org/learn/current", source="freecodecamp",
                     group="freecodecamp", stype="html_scrape", name="Current Cert")])}
    out = build_course_ingest_graph(make_deps(store, still)).invoke(_initial(tmp_path, run_id="c2"))

    assert _missed(store, gone) == 1, "a real census stopped ageing withdrawn courses"
    assert "freecodecamp" in out["ageing"]["aged"]


def test_a_truncated_census_fetch_does_not_age(store, tmp_path):
    """`--limit` on a census source is still only a slice."""
    url = "https://www.freecodecamp.org/learn/a"
    seed = {"coursera": FakeAdapter("coursera", courses=[]),
            "freecodecamp": FakeAdapter("freecodecamp", courses=[
                raw(url, source="freecodecamp", group="freecodecamp", stype="html_scrape")])}
    build_course_ingest_graph(make_deps(store, seed)).invoke(_initial(tmp_path))

    capped = {"coursera": FakeAdapter("coursera", courses=[]),
              "freecodecamp": FakeAdapter("freecodecamp", truncated=True, courses=[
                  raw("https://www.freecodecamp.org/learn/b", source="freecodecamp",
                      group="freecodecamp", stype="html_scrape", name="Other")])}
    out = build_course_ingest_graph(make_deps(store, capped)).invoke(_initial(tmp_path, run_id="c2"))

    assert _missed(store, url) == 0
    assert "freecodecamp" in out["ageing"]["not_aged"]


def test_a_failing_batch_rolls_back_the_ageing_it_shared_a_transaction_with(store, tmp_path):
    """Ageing used to commit in its own earlier transaction, so a batch that
    then died aged the whole source with nothing reset — the failure itself
    pushed live courses toward deletion. Now they share one transaction, so a
    lost batch loses the ageing too."""
    url = "https://www.freecodecamp.org/learn/a"
    seed = {"coursera": FakeAdapter("coursera", courses=[]),
            "freecodecamp": FakeAdapter("freecodecamp", courses=[
                raw(url, source="freecodecamp", group="freecodecamp", stype="html_scrape")])}
    build_course_ingest_graph(make_deps(store, seed)).invoke(_initial(tmp_path))

    deps = make_deps(store, {
        "coursera": FakeAdapter("coursera", courses=[]),
        "freecodecamp": FakeAdapter("freecodecamp", courses=[
            raw("https://www.freecodecamp.org/learn/b", source="freecodecamp",
                group="freecodecamp", stype="html_scrape", name="Other")])})

    original = store.upsert_batch
    def _boom(rows):
        raise RuntimeError("connection lost mid-write")
    store.upsert_batch = _boom
    try:
        out = build_course_ingest_graph(deps).invoke(_initial(tmp_path, run_id="c2"))
    finally:
        store.upsert_batch = original

    assert out["ingest_errors"], "a lost batch was not reported"
    assert out["partial_cycle"] is True, "a lost batch still exited clean"
    assert _missed(store, url) == 0, "the failure itself aged the source"


def test_an_extraction_failure_is_reported_without_losing_the_cycle(store, tmp_path):
    """The per-course boundary: the cycle stays healthy, the loss is counted, and
    nothing is written for the course we could not read."""
    deps = make_deps(store, {
        "coursera": FakeAdapter("coursera", courses=[]),
        "freecodecamp": FakeAdapter("freecodecamp", courses=[
            raw("https://www.freecodecamp.org/learn/x", source="freecodecamp",
                group="freecodecamp", stype="html_scrape")])})

    class _Boom:
        def invoke(self, _payload):
            raise RuntimeError("model unavailable")

    deps.extractor = _Boom()
    out = build_course_ingest_graph(deps).invoke(_initial(tmp_path))

    assert out["ingest_summary"]["extraction_failed"] == 1
    assert out["ingest_summary"]["written"] == 0
    assert out["ingest_summary"]["rejected"] == 0, (
        "'we could not look' was recorded as 'the course teaches nothing'")
