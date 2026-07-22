"""source_health may only be touched where degradation is REPORTED, never where
the scraper could react to it.

The migration states this as an invariant: the table is read and written only in
the run-log path, so no code exists that could escalate retries in response to a
recorded failure. This test enforces it structurally — a grep would be fooled by
a comment, so it checks the actual identifiers the code references.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

PKG = Path(__file__).resolve().parents[2] / "agents" / "agent_b_job_ingest"

# The health METHODS may be called only where degradation is reported: the store
# owns them, the runlog node (nodes.py) is the one caller. A state field merely
# NAMED source_health (carrying the rows runlog produced) is not touching the
# table and is fine — the invariant is about who can act on health, not the word.
HEALTH_METHODS = {"record_source_health", "get_source_health"}
ALLOWED = {PKG / "db" / "store.py", PKG / "nodes.py"}

# Modules that own the operational layer. sources/ and the pure transforms
# (pipeline, aggregate, legitimacy) must not reach the health table's SQL either.
SCRAPER_AND_TRANSFORMS = [
    PKG / "pipeline.py", PKG / "aggregate.py", PKG / "legitimacy.py",
    *sorted((PKG / "sources").glob("*.py")),
]

MODULES = sorted(p for p in PKG.rglob("*.py") if "__pycache__" not in p.parts)


def _attributes_and_calls(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            out.add(node.attr)
        elif isinstance(node, ast.Name):
            out.add(node.id)
    return out


def _sql_table_refs(path: Path) -> set[str]:
    """SQL statements that read or write the source_health TABLE."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    hits: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            text = node.value.lower()
            for verb in ("from source_health", "into source_health",
                         "update source_health", "table source_health"):
                if verb in text:
                    hits.add(verb)
    return hits


def test_modules_exist():
    assert MODULES


@pytest.mark.parametrize("path", MODULES, ids=lambda p: p.name)
def test_health_methods_are_called_only_in_the_allowed_modules(path):
    if path in ALLOWED:
        return
    offenders = _attributes_and_calls(path) & HEALTH_METHODS
    assert not offenders, f"{path.name} calls a health method: {offenders}"


@pytest.mark.parametrize("path", SCRAPER_AND_TRANSFORMS, ids=lambda p: p.name)
def test_the_health_table_sql_never_leaks_into_the_scraper_or_transforms(path):
    """No code path under sources/ or in the pure transforms can read or write
    the health table — so none of them can react to a recorded failure."""
    assert not _sql_table_refs(path), f"{path.name} contains source_health table SQL"


def test_the_runlog_is_the_only_caller_of_record_source_health():
    """Beyond the table name: the WRITE method has exactly one call site, the
    runlog node. If another node started recording health, the single-writer
    guarantee the migration relies on would quietly break."""
    callers = []
    for path in MODULES:
        if path.name == "store.py":
            continue  # the definition lives here
        if "record_source_health" in path.read_text(encoding="utf-8"):
            callers.append(path.name)
    assert callers == ["nodes.py"], f"unexpected callers of record_source_health: {callers}"
