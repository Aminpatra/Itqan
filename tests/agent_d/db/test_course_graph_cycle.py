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
                              base_url="https://api.coursera.org", terms_reviewed=True)
FCC = CourseSourceConfig(name="freecodecamp", source_group="freecodecamp",
                         source_type="html_scrape", base_url="https://www.freecodecamp.org")


def raw(url, *, source, group, stype, name="Accounting Basics"):
    return RawCourse(source=source, source_group=group, source_type=stype, source_url=url,
                     name=name, raw_description="A course on accounting.", provider="IBM",
                     primary_language="en")


class FakeAdapter:
    def __init__(self, name, *, courses, error=None, partial=False):
        self.name, self._courses, self._error, self._partial = name, courses, error, partial

    def fetch(self, *, limit=None):
        r = AdapterResult(source=self.name)
        r.courses = list(self._courses)[:limit] if limit else list(self._courses)
        r.pages_fetched = 1
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
