"""Source registry validation — the checks that must be fatal, not advisory."""

from __future__ import annotations

import pytest

from agents.agent_b_job_ingest.sources.config import (
    DEFAULT_SOURCES,
    SourceConfig,
    SourceConfigError,
    normalize_handle,
    select_sources,
    validate_source_config,
)


def telegram(name: str, handle: str, **kw) -> SourceConfig:
    return SourceConfig(
        name=name,
        source_group=kw.pop("source_group", "g"),
        source_type="telegram",
        handle=handle,
        terms_reviewed=kw.pop("terms_reviewed", True),
        **kw,
    )


# ---------------------------------------------------------------------------
# handle normalization
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "raw",
    [
        "@OmanJob1",
        "OmanJob1",
        "omanjob1",
        "t.me/OmanJob1",
        "https://t.me/OmanJob1",
        "https://t.me/s/OmanJob1",
        "https://t.me/s/OmanJob1/",
        "https://t.me/s/omanjob1?before=1",
        "  @omanjob1  ",
    ],
)
def test_every_written_form_of_one_channel_normalizes_alike(raw):
    assert normalize_handle(raw) == "omanjob1"


@pytest.mark.parametrize("bad", ["", "@", "abc", "1channel", "has-a-dash", "https://t.me/s/"])
def test_invalid_handles_are_rejected_at_config_load(bad):
    with pytest.raises(SourceConfigError):
        normalize_handle(bad)


# ---------------------------------------------------------------------------
# uniqueness — the check that must kill startup
# ---------------------------------------------------------------------------
def test_two_entries_for_one_channel_fail_startup():
    """Not a warning.

    Two entries produce two PoliteClients with two independent rate limiters
    aimed at one host, halving the real interval while every log line still
    reports the polite number. Downstream it is worse: identical postings get
    distinct ids, which double-counts every frequency_count in the table Agent C
    trusts — a fabricated number that appears nowhere in the run log.
    """
    sources = [telegram("a", "@OmanJob1"), telegram("b", "https://t.me/s/omanjob1/")]

    with pytest.raises(SourceConfigError, match="duplicate source identities"):
        validate_source_config(sources)


def test_the_error_names_both_offending_entries():
    """An operator has to be able to find them; "duplicate detected" would send
    them reading the whole registry."""
    sources = [telegram("first", "@OmanJob1"), telegram("second", "@omanjob1")]

    with pytest.raises(SourceConfigError) as exc:
        validate_source_config(sources)

    assert "first" in str(exc.value) and "second" in str(exc.value)


def test_a_collision_involving_a_disabled_entry_still_fails():
    """A collision left in the file is a collision waiting for someone to flip
    `enabled`, at which point it fires in production instead of at review."""
    sources = [telegram("a", "@OmanJob1"), telegram("b", "@omanjob1", enabled=False)]

    with pytest.raises(SourceConfigError, match="duplicate"):
        validate_source_config(sources)


def test_uniqueness_is_not_relaxed_by_dry_run():
    sources = [telegram("a", "@OmanJob1"), telegram("b", "@omanjob1")]

    with pytest.raises(SourceConfigError, match="duplicate"):
        validate_source_config(sources, dry_run=True)


def test_two_web_sources_on_one_host_also_collide():
    sources = [
        SourceConfig(name="a", source_group="g", source_type="blogger_feed", base_url="https://x.test"),
        SourceConfig(name="b", source_group="g", source_type="blogger_feed", base_url="https://X.test/feed"),
    ]

    with pytest.raises(SourceConfigError, match="duplicate"):
        validate_source_config(sources)


def test_differently_named_entries_still_collide_on_identity():
    """Comparing names would miss the collision that matters — the point is the
    host or channel being hit twice, not the label in the config file."""
    sources = [telegram("north", "@OmanJob1"), telegram("south", "@OMANJOB1")]

    with pytest.raises(SourceConfigError, match="duplicate"):
        validate_source_config(sources)


# ---------------------------------------------------------------------------
# terms_reviewed — the human gate
# ---------------------------------------------------------------------------
def test_a_telegram_channel_cannot_go_live_without_human_review():
    """The adapter is generic, so adding a channel is a one-line change. That
    is exactly why the gate exists: it is too easy to leave to intention."""
    sources = [telegram("tg", "@somechannel", terms_reviewed=False)]

    with pytest.raises(SourceConfigError, match="terms_reviewed"):
        validate_source_config(sources)


def test_dry_run_may_inspect_an_unreviewed_channel():
    """A dry run writes nothing and is how a human looks at a channel in order
    to decide whether to review it. Refusing that would make the gate
    impossible to satisfy."""
    sources = [telegram("tg", "@somechannel", terms_reviewed=False)]

    assert [s.name for s in validate_source_config(sources, dry_run=True)] == ["tg"]


def test_terms_reviewed_defaults_to_false():
    """Never inferred, never set in code, never carried over from the channel
    simply rendering."""
    assert SourceConfig(
        name="x", source_group="g", source_type="telegram", handle="@somechannel"
    ).terms_reviewed is False


def test_every_live_telegram_source_carries_an_explicit_review():
    """Not an assertion about WHICH channels are enabled — that is the human's
    call and changes over time. What must stay true is that no channel reaches a
    live cycle without someone having set the flag for it individually, so a
    reviewed channel never confers approval on one added beside it later.
    """
    for source in DEFAULT_SOURCES:
        if source.source_type == "telegram" and source.enabled:
            assert source.terms_reviewed is True, (
                f"{source.name} is enabled but unreviewed; it would fail startup"
            )

    # And the gate still bites: a NEW channel added next to a reviewed one gets
    # no benefit from its neighbour.
    with pytest.raises(SourceConfigError, match="terms_reviewed"):
        validate_source_config(
            list(DEFAULT_SOURCES) + [telegram("tg_new", "@another", terms_reviewed=False)]
        )


def test_the_shipped_registry_is_internally_valid():
    """It must at least pass uniqueness, which is the check no human can be
    relied on to run by eye."""
    assert select_sources("el7far", dry_run=True)


# ---------------------------------------------------------------------------
def test_a_disabled_source_can_be_inspected_by_name_under_dry_run():
    """How an operator looks at a source before deciding to enable it. A dry
    run writes nothing, so this cannot ingest anything."""
    assert [s.name for s in select_sources("dubizzle", dry_run=True)] == ["dubizzle"]


def test_a_disabled_source_is_never_in_the_default_set():
    """"Disabled" must still mean "does not run" — the dry-run allowance only
    stops it meaning "cannot be looked at"."""
    assert "dubizzle" not in [s.name for s in select_sources(None, dry_run=True)]
    assert "dubizzle" not in [s.name for s in validate_source_config(DEFAULT_SOURCES)]


def test_a_disabled_source_cannot_be_selected_for_a_live_run():
    with pytest.raises(SourceConfigError, match="unknown source"):
        select_sources("dubizzle", dry_run=False)


def test_unknown_source_names_are_rejected_with_the_available_list():
    with pytest.raises(SourceConfigError, match="unknown source"):
        select_sources("nope", dry_run=True)


def test_validation_runs_over_the_whole_registry_before_filtering():
    """A duplicate must fail startup whether or not this particular invocation
    selects the entries involved."""
    sources = [
        SourceConfig(name="ok", source_group="g", source_type="blogger_feed", base_url="https://a.test"),
        telegram("a", "@OmanJob1"),
        telegram("b", "@omanjob1"),
    ]

    with pytest.raises(SourceConfigError, match="duplicate"):
        select_sources("ok", sources=tuple(sources), dry_run=True)
