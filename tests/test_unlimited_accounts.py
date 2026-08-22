"""Developer accounts the assistant's quotas do not stop.

`test_a_lookalike_address_is_not_exempt` is the test that carries this file. The
exemption is decided by an email, and an email is the one identifier an attacker
can choose freely — so the comparison has to be exact equality after casefolding.
A substring or suffix test would hand unlimited model spend to anyone who can
register `something@gmail.com.attacker.net`.

`test_the_ceiling_and_the_displayed_limit_agree` is the other one. The claim and
the number shown to the user come from one function on purpose: raising only the
claim leaves a developer reading "0 of 30 remaining" while their messages keep
working, which is the product lying to the four people most likely to believe it.
"""

from __future__ import annotations

import pytest

from api.assistant import limit_for, quota_state
from shared.config import Config

DEV = "dev@example.com"
OTHER = "someone@example.com"


@pytest.fixture
def dev_config(monkeypatch) -> Config:
    monkeypatch.setenv("ITQAN_UNLIMITED_EMAILS", f"{DEV}, second@example.com ")
    return Config()


class _Store:
    """Only what `quota_state` reads."""

    def __init__(self, used: int = 0) -> None:
        self._used = used

    def quota_used(self, user_id, *, kind, period_start):  # noqa: ARG002
        return self._used


# ---------------------------------------------------------------------------
# who is exempt
# ---------------------------------------------------------------------------
def test_nobody_is_exempt_by_default():
    """Unset is the state of every fresh clone and every CI run, and it must mean
    the limits behave exactly as they do for a real user."""
    config = Config()
    assert config.unlimited_emails == ()
    assert config.is_unlimited(DEV) is False
    assert limit_for(config, "message", DEV) == config.assistant_daily_messages


def test_a_listed_address_is_exempt(dev_config):
    assert dev_config.is_unlimited(DEV) is True


@pytest.mark.parametrize("variant", [
    "DEV@Example.COM",      # casefolded
    "  dev@example.com  ",  # the list and the row may both carry whitespace
])
def test_matching_survives_case_and_whitespace(dev_config, variant):
    assert dev_config.is_unlimited(variant) is True


@pytest.mark.parametrize("impostor", [
    "dev@example.com.attacker.net",   # suffix
    "notdev@example.com",             # prefix
    "dev@example.co",                 # near miss
    "",
    None,
])
def test_a_lookalike_address_is_not_exempt(dev_config, impostor):
    """THE test. Exact equality, not containment — the address is attacker-chosen."""
    assert dev_config.is_unlimited(impostor) is False


# ---------------------------------------------------------------------------
# what exempt means
# ---------------------------------------------------------------------------
def test_the_ceiling_is_raised_for_both_kinds(dev_config):
    assert limit_for(dev_config, "message", DEV) == dev_config.unlimited_daily_messages
    assert limit_for(dev_config, "rerun", DEV) == dev_config.unlimited_weekly_reruns
    # And a normal account is untouched.
    assert limit_for(dev_config, "message", OTHER) == 30
    assert limit_for(dev_config, "rerun", OTHER) == 1


def test_the_ceiling_and_the_displayed_limit_agree(dev_config):
    """What the claim allows and what the interface reports must be one number."""
    for kind in ("message", "rerun"):
        shown = quota_state(_Store(used=99), dev_config, "u_1", kind, DEV)
        assert shown["limit"] == limit_for(dev_config, kind, DEV)
        assert shown["remaining"] == shown["limit"] - 99

        normal = quota_state(_Store(used=99), dev_config, "u_2", kind, OTHER)
        assert normal["limit"] == limit_for(dev_config, kind, OTHER)
        # Over their real limit, so nothing left — and never a negative.
        assert normal["remaining"] == 0


def test_an_exempt_ceiling_is_high_but_finite(dev_config):
    """Not a bypass. `claim_quota` still runs, so usage is still counted and a
    runaway loop still meets a wall — the guard its own docstring calls the only
    thing between a chat box and an unbounded bill keeps doing its job."""
    assert dev_config.unlimited_daily_messages > dev_config.assistant_daily_messages
    assert dev_config.unlimited_weekly_reruns > dev_config.assistant_weekly_reruns
    assert dev_config.unlimited_daily_messages < 1_000_000


def test_outbound_mail_limits_are_not_relaxed(dev_config):
    """Stated as a test so its absence reads as a decision rather than an
    oversight: ~200 test-suite messages through the live relay has already
    happened here, and 'developers may send unlimited verification mail' is the
    one exemption that would actively hurt."""
    assert dev_config.reset_requests_per_email_hour == 3
    assert dev_config.reset_requests_per_ip_hour == 10
    # And nothing in the exemption reaches them: the only knobs it moves are the
    # two assistant ceilings.
    assert dev_config.reset_requests_per_email_hour == Config().reset_requests_per_email_hour
