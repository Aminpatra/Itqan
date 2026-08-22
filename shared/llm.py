"""Chat-model factory shared by every agent.

The only subtlety here is ``structured``. Our schemas are deliberately
``Optional``-heavy — a null means "this was not in the document", which is the
whole anti-hallucination mechanism. OpenAI's strict ``json_schema`` mode is picky
about optional-with-default fields, so we fall back to function calling rather
than letting a schema quirk take down the run.

**That fallback matters more now that the model may not be OpenAI's.** With
``Config.api_base`` pointed at a gateway, ``model`` can name any provider's model,
and how faithfully each one honours a JSON schema is exactly the thing that
varies. Every LLM call in this system except Agent E's rationale is structured, so
a model that cannot be constrained is not a degraded model here — it is an
unusable one, which is why the last line raises rather than returning prose.
"""

from __future__ import annotations

from typing import Any, TypeVar

from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from .config import Config

TSchema = TypeVar("TSchema", bound=BaseModel)


def build_llm(config: Config | None = None, **overrides: Any) -> ChatOpenAI:
    """The chat model, from OpenAI or from a gateway that speaks its API.

    `ChatOpenAI` regardless, and that is the point rather than a shortcut:
    OpenRouter is OpenAI-API-shaped, so pointing `base_url` at it reaches any
    model it serves — `google/gemini-3.7-flash`, a Claude, anything — without a
    second SDK, a provider registry, or a prefix parser. Trying a different model
    becomes one config value, which is what makes a bake-off cheap enough to
    actually re-run.

    With `api_base` unset this is byte-for-byte the previous behaviour.
    """
    config = config or Config()
    params: dict[str, Any] = {
        "model": config.model,
        "temperature": config.temperature,
        # The CHAT key, which is not necessarily the embedding key any more.
        "api_key": config.require_chat_key(),
    }
    # Only when someone asked for a ceiling. Capping this by default truncated
    # Agent A's coursework derivation mid-reasoning and cost a profile 15 skills
    # — see `max_output_tokens` for the measurement.
    if config.max_output_tokens > 0:
        params["max_tokens"] = config.max_output_tokens
    # Only when set, because a model without a reasoning mode rejects it outright.
    # Empty is the escape hatch for exactly that case — see `reasoning_effort`.
    if config.reasoning_effort:
        params["reasoning_effort"] = config.reasoning_effort
    if config.api_base:
        params["base_url"] = config.api_base
    params.update(overrides)
    return ChatOpenAI(**params)


def structured(llm: Any, schema: type[TSchema]) -> Any:
    """Bind ``schema`` as the model's output contract.

    Tries strict json_schema first (better guarantees), then function calling.
    A model that cannot be constrained at all is a hard failure — we would rather
    stop than let free-form prose flow into the extraction pipeline.
    """
    try:
        return llm.with_structured_output(schema, method="json_schema", strict=True)
    except Exception:
        return llm.with_structured_output(schema, method="function_calling")


def as_dict(result: Any) -> dict:
    """Normalize a structured-output result to a plain dict.

    Graph state holds dicts, not BaseModel instances — the checkpointer
    serializes state on every superstep and plain dicts avoid round-trip
    surprises (and keep interrupt payloads directly JSON-printable).
    """
    if isinstance(result, BaseModel):
        return result.model_dump()
    if isinstance(result, dict):
        return result
    raise TypeError(f"Unexpected structured-output result type: {type(result)!r}")
