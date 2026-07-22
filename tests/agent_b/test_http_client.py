"""PoliteClient and the boundaries it enforces on every outbound request."""

from __future__ import annotations

import httpx
import pytest

from agents.agent_b_job_ingest.sources.http import (
    Blocked,
    PoliteClient,
    ResponseTooLarge,
    SourcePolicy,
)
from shared.config import Config

UA = "ItqanJobBot/0.1 (+test@example.test)"


def client(handler, **policy_kw) -> PoliteClient:
    transport = httpx.MockTransport(handler)
    policy = SourcePolicy(min_interval_s=0.0, **policy_kw)
    return PoliteClient(
        source="test",
        policy=policy,
        client=httpx.Client(transport=transport, headers={"User-Agent": UA}),
        user_agent=UA,
    )


# ---------------------------------------------------------------------------
# size cap
# ---------------------------------------------------------------------------
def test_an_oversized_body_is_refused_mid_stream():
    """The cap is not a tuning knob. Reading an unbounded response into memory
    before handing it to the XML parser is how a malformed or hostile document
    turns into an out-of-memory kill of an unattended scheduled job."""
    big = "x" * 5000

    with client(lambda r: httpx.Response(200, text=big), max_bytes=1000) as c:
        with pytest.raises(ResponseTooLarge):
            c.get_text("https://x.test/feed")


def test_a_lying_content_length_does_not_defeat_the_cap():
    """content-length is checked first because it is cheap and lets us refuse
    before transferring anything — but it is absent on chunked responses and a
    server may simply lie, so the streaming accumulator is the real
    enforcement."""

    def handler(request):
        return httpx.Response(200, text="x" * 5000, headers={"content-length": "10"})

    with client(handler, max_bytes=1000) as c:
        with pytest.raises(ResponseTooLarge):
            c.get_text("https://x.test/feed")


def test_a_body_under_the_cap_is_returned():
    with client(lambda r: httpx.Response(200, text="ok"), max_bytes=1000) as c:
        assert c.get_text("https://x.test/feed") == "ok"


# ---------------------------------------------------------------------------
# blocks vs failures
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("status", [403, 429])
def test_a_block_is_raised_immediately_and_never_retried(status):
    """A block is a decision by the operator. Retrying it harder is precisely
    the behaviour the project constraints forbid — so it is a distinct exception
    type from a transport failure, not a status code checked at the call site."""
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(status)

    with client(handler, max_retries=3) as c:
        with pytest.raises(Blocked) as exc:
            c.get_text("https://x.test/feed")

    assert exc.value.status == status
    assert len(calls) == 1, "a block was retried"


def test_a_transient_server_error_is_retried_then_succeeds():
    responses = [httpx.Response(503), httpx.Response(200, text="ok")]

    with client(lambda r: responses.pop(0), max_retries=2, backoff_s=0.0) as c:
        assert c.get_text("https://x.test/feed") == "ok"


def test_a_404_is_an_answer_not_a_failure_to_retry():
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(404)

    with client(handler, max_retries=3, backoff_s=0.0) as c:
        with pytest.raises(httpx.HTTPStatusError):
            c.get_text("https://x.test/robots.txt")

    assert len(calls) == 1


def test_retries_are_exhausted_then_the_error_surfaces():
    """A silent empty string would look identical to a source that published
    nothing, and the cycle would age real inventory toward deletion."""
    with client(lambda r: httpx.Response(503), max_retries=1, backoff_s=0.0) as c:
        with pytest.raises(httpx.HTTPStatusError):
            c.get_text("https://x.test/feed")


# ---------------------------------------------------------------------------
# identification
# ---------------------------------------------------------------------------
def test_the_user_agent_is_sent_on_every_request():
    seen = []

    def handler(request):
        seen.append(request.headers.get("user-agent"))
        return httpx.Response(200, text="ok")

    with client(handler) as c:
        c.get_text("https://x.test/a")
        c.get_text("https://x.test/b")

    assert seen == [UA, UA]


def test_a_live_client_refuses_to_exist_without_a_contact_address():
    """Enforced at the last point before bytes leave the machine, rather than
    documented — a scraper that does not identify itself gives an operator no
    option except to block it, and no way to reach us first."""
    config = Config(user_agent="ItqanJobBot/0.1 (+contact-not-configured)")

    with pytest.raises(RuntimeError, match="no contact address"):
        PoliteClient(source="test", config=config)


def test_bytes_fetched_accumulates_for_the_run_log():
    with client(lambda r: httpx.Response(200, text="12345")) as c:
        c.get_text("https://x.test/a")
        c.get_text("https://x.test/b")

    assert c.bytes_fetched == 10
