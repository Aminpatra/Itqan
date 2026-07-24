"""Course source registry validation — the api terms-gate and uniqueness."""

from __future__ import annotations

import pytest

from agents.agent_d_course_ingest.sources.config import (
    DEFAULT_SOURCES,
    CourseSourceConfig,
    SourceConfigError,
    select_sources,
    validate_source_config,
)


def api(name, host, **kw):
    return CourseSourceConfig(name=name, source_group=kw.pop("group", "g"),
                              source_type="api", base_url=f"https://{host}",
                              terms_reviewed=kw.pop("terms_reviewed", True), **kw)


def test_an_api_source_needs_human_reviewed_terms():
    """An api source is governed by its terms, not robots — a human must accept
    them. Not True, not --dry-run => startup fails."""
    with pytest.raises(SourceConfigError, match="terms_reviewed"):
        validate_source_config([api("x", "api.x.test", terms_reviewed=False)])


def test_dry_run_may_inspect_an_unreviewed_api_source():
    assert [s.name for s in validate_source_config(
        [api("x", "api.x.test", terms_reviewed=False)], dry_run=True)] == ["x"]


def test_html_scrape_source_needs_no_terms_gate():
    """A web scrape is governed by robots (checked at fetch time), so it has no
    terms_reviewed requirement at config load."""
    fcc = CourseSourceConfig(name="fcc", source_group="g", source_type="html_scrape",
                             base_url="https://www.freecodecamp.org")
    assert [s.name for s in validate_source_config([fcc])] == ["fcc"]


def test_duplicate_identities_fail_startup():
    with pytest.raises(SourceConfigError, match="duplicate"):
        validate_source_config([api("a", "api.x.test"), api("b", "api.x.test")])


def test_the_shipped_registry_is_valid_and_has_both_sources():
    names = {s.name for s in validate_source_config(DEFAULT_SOURCES)}
    assert names == {"coursera", "freecodecamp"}


def test_coursera_ships_terms_reviewed_true_by_user_approval():
    coursera = [s for s in DEFAULT_SOURCES if s.name == "coursera"][0]
    assert coursera.terms_reviewed is True and coursera.enabled


def test_unknown_source_is_rejected():
    with pytest.raises(SourceConfigError, match="unknown source"):
        select_sources("nope", dry_run=True)
