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
"""

from __future__ import annotations

import pytest

from api import email as email_module


@pytest.fixture(autouse=True)
def counters_reset(monkeypatch):
    for name in ("SENDS_ATTEMPTED", "SENDS_ACCEPTED", "SENDS_REFUSED", "SEND_FAILURES"):
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


def test_development_still_sends_to_a_test_domain(monkeypatch):
    """Local runs point at a sink that is happy to accept `@itqan.test`, and the
    live walkthroughs of signup depend on it. The guard is a production rule."""
    monkeypatch.setenv("ITQAN_ENV", "development")
    sent = []
    monkeypatch.setattr(email_module, "_RecordingSMTP", _FakeSMTP(sent))

    from shared.config import Config
    email_module.send(to="probe@itqan.test", subject="s", body="b",
                      config=Config(smtp_host="relay.test"))
    assert len(sent) == 1


def test_a_successful_send_is_visible(monkeypatch, caplog):
    """THE gap. Before this, a successful send logged nothing and moved no
    counter, so "has anything left this box today?" had no answer short of
    asking the provider."""
    monkeypatch.setenv("ITQAN_ENV", "development")
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
    assert set(snapshot) == {"attempted", "accepted", "refused", "failed", "lastSendAt"}


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
