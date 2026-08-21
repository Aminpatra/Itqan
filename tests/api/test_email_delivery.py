"""What the mail layer can and cannot see, and who it refuses to write to.

Both tests here exist because of one incident. On 2026-08-18 no message reached
anyone for hours while `SEND_FAILURES` read 0 and had read 0 since the feature
shipped: the relay accepted every message with a 250 and delivered none, so
nothing raised and nothing was counted. Shortly before, six probe accounts had
been created on production at `@itqan.test` — a reserved TLD that can never
resolve — so roughly seven guaranteed hard bounces were charged against a sender
that otherwise handles a handful of real messages a day.

`test_production_refuses_an_address_that_can_never_receive` stops the cause.
`test_a_successful_send_is_visible` closes the gap that let it run for hours.

**Rewritten 2026-08-21, because the diagnosis above was wrong about the scale.**
It was not seven probe messages. This suite was sending ~200 per run: the guard
was gated on production, the `relay` fixture protected two files out of twelve,
and `.env` gives a test run the production relay. `test_a_test_domain_is_refused_in_development_too`
is the inversion of a test that used to live here asserting the opposite -- it
was called `test_development_still_sends_to_a_test_domain`, and it pinned the
leak open on an assumption nobody had checked.
"""

from __future__ import annotations

import pytest

from api import email as email_module


@pytest.fixture(autouse=True)
def counters_reset(monkeypatch):
    for name in ("SENDS_ATTEMPTED", "SENDS_ACCEPTED", "SENDS_REFUSED",
                 "SENDS_SUPPRESSED", "SEND_FAILURES"):
        monkeypatch.setattr(email_module, name, 0)
    monkeypatch.setattr(email_module, "LAST_SEND_AT", None)


@pytest.mark.parametrize("address", [
    "probe@itqan.test",         # the exact shape that caused it
    "someone@foo.invalid",
    "x@example.com",            # reserved second-level, not a suffix match
    "y@app.localhost",
    "no-at-sign",
])
def test_an_undeliverable_address_is_recognised(address):
    assert email_module.is_undeliverable(address) is True


@pytest.mark.parametrize("address", [
    "aminpatranew@gmail.com",
    "hr@tryitqan.com",
    "someone@myexample.com",    # NOT example.com; a real domain must still pass
])
def test_a_real_address_is_left_alone(address):
    assert email_module.is_undeliverable(address) is False


def test_production_refuses_an_address_that_can_never_receive(monkeypatch, caplog):
    """Refused BEFORE the socket opens, so a guaranteed bounce never reaches the
    relay. A hard bounce is charged against sender reputation, and every later
    message pays it — including the one a real person is waiting for."""
    monkeypatch.setenv("ITQAN_ENV", "production")
    opened = []
    monkeypatch.setattr(email_module.smtplib, "SMTP",
                        lambda *a, **k: opened.append(a) or pytest.fail("a socket was opened"))

    from shared.config import Config
    with caplog.at_level("ERROR"):
        with pytest.raises(email_module.Undeliverable):
            email_module.send(to="probe@itqan.test", subject="s", body="b",
                              config=Config(smtp_host="relay.test"), purpose="verification")

    assert opened == []
    assert email_module.SENDS_REFUSED == 1
    assert email_module.SENDS_ATTEMPTED == 0
    # Redacted, as everywhere else in this module.
    assert "probe@itqan.test" not in caplog.text
    assert "p***@itqan.test" in caplog.text


def test_a_test_domain_is_refused_in_development_too(monkeypatch):
    """THE inverted test, and the reason this file needed rewriting.

    Its predecessor was called `test_development_still_sends_to_a_test_domain`
    and asserted the opposite, on the stated assumption that "local runs point at
    a sink that is happy to accept `@itqan.test`". They never did: `shared/config`
    calls `load_dotenv()` unconditionally, so dev, the test suite and production
    all read the SAME relay credentials. That assumption held the door open for
    ~200 test-suite bounces through the live account on 2026-08-20.

    A reserved TLD cannot resolve on any host, in any environment, on any day. The
    environment was never relevant to the fact -- only to a belief about the relay
    that was never checked.
    """
    monkeypatch.setenv("ITQAN_ENV", "development")
    monkeypatch.setenv("ITQAN_SMTP_ENABLED", "1")   # even with sending forced ON
    sent = []
    monkeypatch.setattr(email_module, "_RecordingSMTP", _FakeSMTP(sent))

    from shared.config import Config
    with pytest.raises(email_module.Undeliverable):
        email_module.send(to="probe@itqan.test", subject="s", body="b",
                          config=Config(smtp_host="relay.test"))
    assert sent == []
    assert email_module.SENDS_REFUSED == 1


# ---------------------------------------------------------------------------
# Mail is off unless this is production, or somebody deliberately said otherwise.
# ---------------------------------------------------------------------------
def test_a_development_run_sends_nothing_at_all(monkeypatch, caplog):
    """THE guard, for the case the other two miss: a REAL address, from a dev box,
    against the production relay credentials in `.env`.

    Neither the socket ban in `tests/conftest.py` (which covers the suite) nor the
    reserved-TLD refusal (which covers fake addresses) would stop this one. It is
    what "do not send to anyone" actually requires.
    """
    monkeypatch.setenv("ITQAN_ENV", "development")
    monkeypatch.delenv("ITQAN_SMTP_ENABLED", raising=False)
    monkeypatch.setattr(email_module, "_RecordingSMTP",
                        lambda *a, **k: pytest.fail("a socket was opened"))

    from shared.config import Config
    with caplog.at_level("INFO"):
        email_module.send(to="aminpatranew@gmail.com", subject="s", body="b",
                          config=Config(smtp_host="smtp-relay.brevo.com"),
                          purpose="verification")

    assert email_module.SENDS_SUPPRESSED == 1
    # Not counted as attempted, accepted or failed: nothing was handed to a relay,
    # and folding a chosen silence into any of those would blunt all three.
    assert email_module.SENDS_ATTEMPTED == 0
    assert email_module.SENDS_ACCEPTED == 0
    assert email_module.SEND_FAILURES == 0


def test_a_suppressed_message_is_shown_to_whoever_ran_it(monkeypatch, caplog):
    """The deliberate exception to "the code is never logged".

    With nothing delivered, the log is the only way to walk a signup locally --
    and the alternative, reading the code out of the database by hand, is the kind
    of friction that ends with somebody switching real sending back on.
    """
    monkeypatch.setenv("ITQAN_ENV", "development")
    monkeypatch.delenv("ITQAN_SMTP_ENABLED", raising=False)

    from shared.config import Config
    with caplog.at_level("INFO"):
        email_module.send(to="maryam@itqan.test", subject="Your Itqan code",
                          body="428173", config=Config(smtp_host="relay.test"),
                          purpose="verification")

    assert "428173" in caplog.text
    assert "NOT sent" in caplog.text


def test_the_override_is_the_only_way_to_send_outside_production(monkeypatch):
    """A person, on purpose, for one run. Never a default."""
    monkeypatch.setenv("ITQAN_ENV", "development")
    monkeypatch.delenv("ITQAN_SMTP_ENABLED", raising=False)
    assert email_module.mail_enabled() is False

    monkeypatch.setenv("ITQAN_SMTP_ENABLED", "1")
    assert email_module.mail_enabled() is True


def test_production_is_untouched_by_all_of_this(monkeypatch):
    """A fix that quietly stopped mailing real users would be worse than the leak:
    the endpoints answer 200 either way, so nobody would find out."""
    monkeypatch.setenv("ITQAN_ENV", "production")
    monkeypatch.delenv("ITQAN_SMTP_ENABLED", raising=False)
    assert email_module.mail_enabled() is True

    sent = []
    monkeypatch.setattr(email_module, "_RecordingSMTP", _FakeSMTP(sent))
    from shared.config import Config
    email_module.send(to="someone@gmail.com", subject="s", body="b",
                      config=Config(smtp_host="relay.test"), purpose="verification")
    assert len(sent) == 1
    assert email_module.SENDS_ACCEPTED == 1
    assert email_module.SENDS_SUPPRESSED == 0


def test_a_successful_send_is_visible(monkeypatch, caplog):
    """THE gap. Before this, a successful send logged nothing and moved no
    counter, so "has anything left this box today?" had no answer short of
    asking the provider.

    Production, because that is now the only environment in which a message
    reaches a relay at all -- see `mail_enabled`."""
    monkeypatch.setenv("ITQAN_ENV", "production")
    sent = []
    monkeypatch.setattr(email_module, "_RecordingSMTP", _FakeSMTP(sent))

    from shared.config import Config
    with caplog.at_level("INFO"):
        email_module.send(to="someone@gmail.com", subject="s", body="b",
                          config=Config(smtp_host="relay.test"), purpose="verification")

    assert email_module.SENDS_ATTEMPTED == 1
    assert email_module.SENDS_ACCEPTED == 1
    assert email_module.LAST_SEND_AT is not None
    # The relay's own id for the message: what makes a provider-side lookup
    # possible instead of an argument.
    assert "queued as ABC123" in caplog.text
    assert "s***@gmail.com" in caplog.text
    assert "someone@gmail.com" not in caplog.text


def test_counters_are_reported_for_a_health_check():
    snapshot = email_module.counters()
    assert set(snapshot) == {"attempted", "accepted", "refused", "suppressed",
                             "failed", "lastSendAt"}


class _FakeSMTP:
    """Stands in for the recording client, including its `last_reply`."""

    def __init__(self, sink):
        self.sink = sink

    def __call__(self, *args, **kwargs):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def starttls(self):
        pass

    def login(self, user, password):
        pass

    def send_message(self, message):
        self.sink.append(message)

    last_reply = (250, b"2.0.0 Ok: queued as ABC123")


def test_a_refusal_is_not_counted_as_a_relay_failure(monkeypatch):
    """`failed` means "the relay would not take it". Folding our own deliberate
    refusals into it would make a burst of test signups read as an outage,
    blunting the one number an operator is meant to trust.

    Found live: the first production signup to a .test address after the guard
    shipped reported refused=1 AND failed=1.
    """
    monkeypatch.setenv("ITQAN_ENV", "production")
    from shared.config import Config

    thread = email_module.send_in_background(
        to="probe@itqan.test", subject="s", body="b",
        config=Config(smtp_host="relay.test"), purpose="verification")
    thread.join(timeout=3)

    assert email_module.SENDS_REFUSED == 1
    assert email_module.SEND_FAILURES == 0


def test_the_handoff_line_is_actually_emitted(monkeypatch, capsys):
    """Instrumentation that cannot be read is not instrumentation.

    Found in production minutes after shipping it: `itqan.email` inherited a root
    with no handlers, so `logging.lastResort` printed the ERROR refusal and
    dropped the INFO hand-off — discarding the relay queue id, which was the one
    artifact worth having.
    """
    import logging

    from api.main import _configure_logging

    _configure_logging()
    assert logging.getLogger("itqan.email").isEnabledFor(logging.INFO)


def test_a_signup_is_captured_rather_than_sent(unverified, relay):
    """THE regression, at the level it actually happened.

    This file never defined a `relay` fixture and never asked for one -- which is
    exactly the position the other ten API test files were in when they sent ~200
    real messages. It inherits the shared fixture from `tests/api/conftest.py`,
    which the `client` fixture applies whether a file asks or not, because
    "remember to opt in" is the design that failed.
    """
    import time

    deadline = time.time() + 3.0
    while time.time() < deadline and not relay:
        time.sleep(0.02)

    assert len(relay) == 1, "signup should produce exactly one message"
    assert relay[0]["to"] == "maryam@itqan.test"
    # Captured, never handed to anything: the counters only move inside `send`,
    # which the relay stands in for.
    assert email_module.SENDS_ATTEMPTED == 0
    assert email_module.SENDS_ACCEPTED == 0
