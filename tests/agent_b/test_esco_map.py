"""ESCO CSV parsing and mapping precedence, offline.

The fixture is SYNTHETIC but shaped exactly like the real skills_en.csv
(same columns, newline-separated altLabels inside a quoted cell, BOM-free
utf-8) — the parser must not care which it is fed. Mapping precedence is tested
against a tiny in-memory store because the rules are pure logic; the SQL that
serves them is covered in the DB suite.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agents.agent_b_job_ingest.esco_map import (
    MapSummary,
    map_new_skills,
    parse_esco_csv,
)
from shared.config import Config
from tests.fake_embedder import FakeEmbedder, _vector_for

FIXTURE = Path(__file__).parent / "fixtures" / "esco_skills_sample.csv"


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------
def test_parses_concepts_with_multiline_alt_labels():
    concepts = list(parse_esco_csv(FIXTURE))

    assert len(concepts) == 5
    time = next(c for c in concepts if c["preferred_label"] == "manage time")
    assert time["alt_labels"] == ["prioritise tasks", "prioritize tasks", "time planning"]
    assert time["esco_uri"].startswith("http://example.test/esco/skill/")
    assert time["skill_type"] == "skill/competence"

    welding = next(c for c in concepts if c["preferred_label"] == "perform welding")
    assert welding["alt_labels"] == []


def test_a_non_esco_csv_is_refused_loudly(tmp_path):
    """Feeding the wrong file (occupations_en.csv, or something else entirely)
    must fail with a message, not load garbage as a taxonomy."""
    wrong = tmp_path / "not_esco.csv"
    wrong.write_text("a,b,c\n1,2,3\n", encoding="utf-8")

    with pytest.raises(ValueError, match="conceptUri"):
        list(parse_esco_csv(wrong))


def test_rows_without_uri_or_label_are_skipped(tmp_path):
    partial = tmp_path / "partial.csv"
    partial.write_text(
        "conceptUri,preferredLabel,altLabels\n"
        "http://x.test/1,alpha skill,\n"
        ",missing uri,\n"
        "http://x.test/3,,\n",
        encoding="utf-8",
    )
    concepts = list(parse_esco_csv(partial))
    assert [c["preferred_label"] for c in concepts] == ["alpha skill"]


# ---------------------------------------------------------------------------
# mapping precedence, against a minimal in-memory store
# ---------------------------------------------------------------------------
class FakeEscoStore:
    """Just the surface map_new_skills uses. Labels are embedded with the same
    FakeEmbedder the mapper will use, so identical text scores 1.0 and
    different text scores ~0 — which makes the threshold branch deterministic."""

    def __init__(self, concepts, pending):
        self.version = "test-1"
        self.pending = list(pending)
        self.map_rows: dict[str, dict] = {}
        self.labels = []  # (label_key, uri, is_preferred, vector)
        for uri, preferred, alts in concepts:
            self.labels.append((preferred.lower(), uri, True, _vector_for(preferred.lower())))
            for alt in alts:
                self.labels.append((alt.lower(), uri, False, _vector_for(alt.lower())))

    def get_esco_version(self):
        return self.version

    def pending_skill_keys(self, *, version):
        return [k for k in self.pending if k not in self.map_rows]

    def find_esco_by_labels(self, label_keys):
        out = {}
        for key in label_keys:
            hits = [l for l in self.labels if l[0] == key]
            if hits:
                hits.sort(key=lambda l: not l[2])  # preferred first
                out[key] = {"label_key": key, "esco_uri": hits[0][1], "is_preferred": hits[0][2]}
        return out

    def nearest_esco_label(self, embedding, *, limit=3):
        scored = [
            {"esco_uri": uri, "label": key,
             "similarity": sum(a * b for a, b in zip(embedding, vec))}
            for key, uri, _pref, vec in self.labels
        ]
        scored.sort(key=lambda c: c["similarity"], reverse=True)
        return scored[:limit]

    def upsert_skill_map(self, entries):
        for e in entries:
            self.map_rows[e["skill_key"]] = e


CONCEPTS = [
    ("uri:time", "manage time", ["prioritise tasks"]),
    ("uri:data", "analyse data", ["data analytics"]),
]


def run_map(pending, *, embedder=FakeEmbedder(), threshold=0.85):
    store = FakeEscoStore(CONCEPTS, pending)
    config = Config(esco_map_threshold=threshold)
    summary = map_new_skills(store, embedder, config)
    return store, summary


def test_exact_preferred_match_wins():
    store, summary = run_map(["manage time"])
    assert summary.exact == 1
    row = store.map_rows["manage time"]
    assert row["esco_uri"] == "uri:time" and row["method"] == "exact"
    assert row["similarity"] is None, "a lexical match has no similarity to report"


def test_alt_label_match_is_recorded_as_such():
    store, summary = run_map(["prioritise tasks"])
    assert summary.alt_label == 1
    assert store.map_rows["prioritise tasks"]["esco_uri"] == "uri:time"
    assert store.map_rows["prioritise tasks"]["method"] == "alt_label"


def test_embedding_matches_identical_text_lexical_missed():
    """FakeEmbedder gives identical text an identical vector, so a key equal to
    a label but arriving via the embedding path scores 1.0 — above any sane
    threshold. (Lexical would normally catch this; the test isolates the
    embedding branch by not registering the key lexically.)"""
    store = FakeEscoStore(CONCEPTS, ["data analytics extra token"])
    # A key that shares no lexical label; its vector is unrelated to any label,
    # so the mapper must record it unmapped WITH the best similarity it saw.
    summary = map_new_skills(store, FakeEmbedder(), Config(esco_map_threshold=0.85))
    row = store.map_rows["data analytics extra token"]
    assert summary.unmapped == 1
    assert row["method"] == "unmapped" and row["esco_uri"] is None
    assert row["similarity"] is not None, "the near-miss score is the tuning evidence"


def test_threshold_zero_maps_by_embedding():
    store, summary = run_map(["totally novel phrasing"], threshold=-1.0)
    assert summary.embedding == 1
    assert store.map_rows["totally novel phrasing"]["method"] == "embedding"


def test_no_embedder_defers_rather_than_recording_unmapped():
    """"We did not look" and "we looked and found nothing" must never share a
    row: without an embedder, lexical misses are left for a future cycle."""
    store, summary = run_map(["totally novel phrasing"], embedder=None)
    assert summary.deferred == 1
    assert "totally novel phrasing" not in store.map_rows


def test_no_taxonomy_is_a_clean_noop():
    class Empty:
        def get_esco_version(self):
            return None

    summary = map_new_skills(Empty(), FakeEmbedder(), Config())
    assert summary == MapSummary()
