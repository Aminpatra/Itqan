"""Chat-model factory shared by every agent.

The only subtlety here is ``structured``. Our schemas are deliberately
``Optional``-heavy — a null means "this was not in the document", which is the
whole anti-hallucination mechanism. OpenAI's strict ``json_schema`` mode is picky
about optional-with-default fields, so we fall back to function calling rather
than letting a schema quirk take down the run.
"""

from __future__ import annotations

from typing import Any, TypeVar

from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from .config import Config

TSchema = TypeVar("TSchema", bound=BaseModel)


def build_llm(config: Config | None = None, **overrides: Any) -> ChatOpenAI:
    config = config or Config()
    config.require_api_key()
    params: dict[str, Any] = {
        "model": config.model,
        "temperature": config.temperature,
    }
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
