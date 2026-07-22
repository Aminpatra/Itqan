"""Structural boundaries around the scraping layer.

These assert facts about the import graph rather than about behaviour, because
the properties they protect are ones a reviewer cannot verify by reading a diff
and a comment cannot enforce.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SOURCES = Path(__file__).resolve().parents[2] / "agents" / "agent_b_job_ingest" / "sources"


def imported_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            prefix = "." * node.level + (node.module or "")
            names.add(prefix)
            names.update(f"{prefix}.{alias.name}" for alias in node.names)
    return names


def code_identifiers(path: Path) -> set[str]:
    """Every name the CODE refers to — imports, attributes, calls, literals.

    Deliberately not a substring search over the file: prose in a docstring
    explaining *why* a thing is not referenced would match it, so the test would
    fail on its own explanation. What matters is whether the module can reach
    the name, and that is an AST question.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set(imported_names(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            # SQL and table names would arrive as strings; docstrings are
            # excluded because they are not referenced by anything.
            if node is not ast.get_docstring(tree, clean=False):
                names.add(node.value)
    docstrings = {
        ast.get_docstring(n, clean=False)
        for n in ast.walk(tree)
        if isinstance(n, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    }
    return {n for n in names if n and n not in docstrings}


SOURCE_FILES = sorted(SOURCES.glob("*.py"))


def test_there_are_source_files_to_check():
    """Guards the parametrized tests below from passing vacuously if the
    directory is ever moved."""
    assert SOURCE_FILES


@pytest.mark.parametrize("path", SOURCE_FILES, ids=lambda p: p.name)
def test_no_adapter_can_reach_the_database(path):
    """Adapters fetch and parse; nothing more.

    This is what keeps every adapter testable offline against a fixture, and
    what makes it impossible for a scraping bug to reach a write path.
    """
    for name in imported_names(path):
        assert "db" not in name.split("."), f"{path.name} imports {name}"
        assert "psycopg" not in name
        assert "store" not in name.split(".")


@pytest.mark.parametrize("path", SOURCE_FILES, ids=lambda p: p.name)
def test_nothing_under_sources_reads_or_writes_source_health(path):
    """Health is recorded in the runlog node and nowhere else.

    Enforced structurally rather than by convention: if no adapter can see the
    failure history, no code path EXISTS that could respond to a block by
    retrying harder. Degradation stays a fact reported to a human, not an input
    to the scraper's own behaviour.
    """
    for name in code_identifiers(path):
        assert "source_health" not in name, f"{path.name} references source_health via {name}"


@pytest.mark.parametrize("path", SOURCE_FILES, ids=lambda p: p.name)
def test_adapters_do_not_import_agent_a(path):
    """Agent B is fully decoupled from the CV pipeline."""
    for name in imported_names(path):
        assert "agent_a_cv_extraction" not in name


def test_the_legitimacy_scorer_does_not_import_an_llm():
    """Phase 2b is the deterministic rule scorer only. The adjudicator arrives
    in phase 4, after the score distribution has been measured on real postings
    — the rules must be tunable without an API key."""
    path = SOURCES.parent / "legitimacy.py"
    for name in imported_names(path):
        assert "langchain" not in name
        assert "openai" not in name
