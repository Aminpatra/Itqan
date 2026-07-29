"""Every Config field the code reads must exist on Config.

This suite exists because it did not. Four fields were dropped from
``shared/config.py`` in an unrelated edit; three call sites used
``getattr(config, name, default)`` and degraded quietly, but
``shared/course_market.py`` read one directly — so **Agent E raised
AttributeError on every run** and nothing caught it until someone ran the agent.

A dataclass has no compile-time link to the modules that read it, so the guard
has to be a test. This one walks the source for ``config.<name>`` accesses and
asserts each one resolves, which turns that class of outage into a red test.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from shared.config import Config

ROOT = Path(__file__).resolve().parent.parent
SEARCH_DIRS = ("shared", "agents")

# Attributes reached through `config.` that are methods or intentionally dynamic.
_NOT_FIELDS = {
    "require_api_key", "require_database_url", "require_identified_user_agent",
    "get", "copy", "keys", "items", "values",
}


def _python_files() -> list[Path]:
    files: list[Path] = []
    for name in SEARCH_DIRS:
        files.extend(p for p in (ROOT / name).rglob("*.py") if "__pycache__" not in p.parts)
    return files


def _direct_config_attributes(path: Path) -> set[str]:
    """`config.foo` / `self.config.foo` / `deps.config.foo`, excluding getattr()
    lookups — those carry their own default and survive a missing field."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:                                   # pragma: no cover
        return set()

    found: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        base = node.value
        name = None
        if isinstance(base, ast.Name):
            name = base.id
        elif isinstance(base, ast.Attribute):
            name = base.attr
        if name == "config" and node.attr not in _NOT_FIELDS:
            found.add(node.attr)
    return found


def test_every_config_attribute_the_code_reads_actually_exists():
    config = Config()
    missing: dict[str, set[str]] = {}
    for path in _python_files():
        absent = {a for a in _direct_config_attributes(path) if not hasattr(config, a)}
        if absent:
            missing[str(path.relative_to(ROOT))] = absent

    assert not missing, (
        "Config fields are read but not defined — this is the shape of an outage, "
        f"not a lint warning: {missing}"
    )


@pytest.mark.parametrize("name", [
    # The four dropped in the 2026-07-28 edit. Named explicitly as well as
    # caught generically, so the regression has a test that says what it was.
    "agent_e_max_candidates_per_skill",
    "course_ingest_chunk_size",
    "course_min_text_chars",
    "course_backfill_pages",
])
def test_the_fields_a_previous_edit_dropped_are_present(name):
    assert hasattr(Config(), name)


# Fallbacks that deliberately differ from the declared default, with the reason.
# These are FAIL-CLOSED: if the field disappears, the safe behaviour is the one
# the fallback picks, not the one the default declares. Listed rather than
# skipped so the divergence is a recorded decision instead of an accident.
_DELIBERATE_DIVERGENCE = {
    # Losing the flag must not silently enable a model call. Agent C falls back
    # to fully deterministic matching, which is what it did before the tier
    # existed.
    "agent_c_llm_matching": False,
}


def test_getattr_defaults_agree_with_the_declared_field():
    """A `getattr(config, x, D)` fallback that disagrees with the real default
    means the code behaves one way in production and another if the field is ever
    removed — a silent divergence, which is what made the outage hard to see."""
    config = Config()
    pattern = re.compile(r"getattr\(\s*(?:self\.|deps\.)?config\s*,\s*[\"'](\w+)[\"']\s*,\s*([^)]+)\)")
    mismatches: list[str] = []
    for path in _python_files():
        for name, literal in pattern.findall(path.read_text(encoding="utf-8")):
            if not hasattr(config, name):
                continue
            try:
                fallback = ast.literal_eval(literal.strip())
            except (ValueError, SyntaxError):
                continue                                   # a non-literal default
            actual = getattr(config, name)
            if fallback == actual or _DELIBERATE_DIVERGENCE.get(name, object()) == fallback:
                continue
            mismatches.append(
                f"{path.relative_to(ROOT)}: {name} default={actual!r} fallback={fallback!r}")
    assert not mismatches, (
        "getattr fallbacks disagree with Config. If the divergence is a deliberate "
        "fail-closed choice, add it to _DELIBERATE_DIVERGENCE with the reason: "
        + "; ".join(mismatches)
    )
