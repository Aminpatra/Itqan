"""How the chat model is built. No network — this inspects the client's settings.

`test_no_output_ceiling_by_default` is the test that carries this file, and it
exists because of a measured incident rather than a hypothetical.

A `max_tokens=8192` ceiling was added on 2026-08-21 to satisfy a gateway that
pre-authorises against the reservation instead of billing what is produced. It
looked generous: ordinary calls here return a few hundred tokens. On 2026-08-22,
against a real CV and transcript, Agent A's coursework derivation consumed the
entire 8192 on REASONING alone -- `completion_tokens=8192,
reasoning_tokens=8192` -- returned nothing parseable, and fell open. Fifteen
skills the transcript evidenced never reached the profile. A person noticed;
no test did.

The lesson is in the sizing, not the number: reasoning tokens count against this
limit, so a ceiling chosen from expected OUTPUT length is measuring the wrong
quantity, and how much a model reasons is not knowable in advance.
"""

from __future__ import annotations

import pytest

from shared.config import Config
from shared.llm import build_llm


@pytest.fixture(autouse=True)
def _key(monkeypatch):
    """A key so the factory builds; no request is ever made."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-a-real-key")
    monkeypatch.delenv("ITQAN_API_BASE", raising=False)


def _max_tokens(llm) -> object:
    """Whatever the client will send as the output ceiling, or None."""
    return getattr(llm, "max_tokens", None)


def test_no_output_ceiling_by_default():
    """THE test. An unasked-for ceiling truncates mid-reasoning and the failure
    reads as 'the model returned nothing useful', which is indistinguishable from
    a model that is simply worse."""
    assert Config().max_output_tokens == 0
    assert _max_tokens(build_llm(Config())) is None


def test_a_ceiling_is_applied_when_one_is_asked_for(monkeypatch):
    """The knob stays for a pre-authorising gateway, where the limit comes from
    that gateway rather than from a guess about this system."""
    monkeypatch.setenv("ITQAN_MAX_OUTPUT_TOKENS", "4096")
    config = Config()
    assert config.max_output_tokens == 4096
    assert _max_tokens(build_llm(config)) == 4096


def test_reasoning_effort_is_set_not_inherited():
    """The GPT-5.6 family defaults to `medium`, measured at twice the latency and
    four times the reasoning of `low` for identical probe results. Inheriting it
    is a decision nobody made."""
    assert Config().reasoning_effort == "low"


def test_reasoning_effort_can_be_disabled_for_a_model_without_one(monkeypatch):
    """A model with no reasoning mode rejects the parameter outright rather than
    ignoring it, so there has to be a way to not send it.

    The sentinel is the word `off`, not an empty string: compose passes optional
    variables as `${VAR:-}`, so empty arrives for every unset variable and must
    mean "use the default" rather than "change the behaviour"."""
    monkeypatch.setenv("ITQAN_REASONING_EFFORT", "off")
    llm = build_llm(Config())
    assert not getattr(llm, "reasoning_effort", None)


def test_the_gateway_is_off_unless_configured():
    """Unset, everything goes straight to OpenAI on the same key it always did —
    which is what makes the gateway support safe to carry while unused."""
    config = Config()
    assert config.api_base == ""
    assert config.require_chat_key() == config.require_api_key()


def test_an_empty_env_var_does_not_override_a_default(monkeypatch):
    """THE compose trap, pinned.

    `docker-compose.yml` passes optional variables through as `${VAR:-}`, so a
    variable that is simply unset on the host arrives in the container as an
    empty STRING. `os.getenv(name, default)` returns that empty string rather
    than the default, which would have meant: every call naming an empty model,
    reasoning silently disabled on every deploy, and `int("")` raising at import
    so the API never booted at all.
    """
    for name in ("ITQAN_MODEL", "ITQAN_REASONING_EFFORT", "ITQAN_MAX_OUTPUT_TOKENS",
                 "ITQAN_API_BASE", "ITQAN_UNLIMITED_EMAILS"):
        monkeypatch.setenv(name, "")

    config = Config()                      # must not raise
    assert config.model == "gpt-5.6-luna"
    assert config.reasoning_effort == "low"
    assert config.max_output_tokens == 0
    # These two read empty as "none", which is the right answer for both.
    assert config.api_base == ""
    assert config.unlimited_emails == ()
