"""Turn a validated SourceConfig into the adapter that can fetch it.

Separate from ``config.py`` so that validating the registry does not require
importing every adapter — a config error should surface as a config error, not
as an ImportError from a module the operator has never heard of.
"""

from __future__ import annotations

from shared.config import Config

from .base import SourceAdapter
from .config import SourceConfig
from .dubizzle import DubizzleAdapter
from .el7far import El7farAdapter
from .telegram import TelegramAdapter


def build_adapter(
    source: SourceConfig,
    *,
    config: Config | None = None,
    is_known_unchanged=None,
) -> SourceAdapter:
    config = config or Config()

    if source.source_type == "blogger_feed":
        return El7farAdapter(
            name=source.name,
            source_group=source.source_group,
            base_url=source.base_url,
            config=config,
            is_known_unchanged=is_known_unchanged,
        )

    if source.source_type == "telegram":
        return TelegramAdapter(
            name=source.name,
            handle=source.handle,
            source_group=source.source_group,
            config=config,
            is_known_unchanged=is_known_unchanged,
        )

    if source.source_type == "html_scrape":
        return DubizzleAdapter(
            name=source.name,
            source_group=source.source_group,
            base_url=source.base_url,
            config=config,
            is_known_unchanged=is_known_unchanged,
        )

    # Raising beats a placeholder that returns an empty result, which would read
    # in the run log as "this source published nothing" — indistinguishable from
    # a real empty day.
    raise NotImplementedError(
        f"{source.name}: no adapter for source_type {source.source_type!r}"
    )
