"""Shared, agent-neutral scraping infrastructure.

Promoted out of Agent B when Agent D (course ingestion) became a second
consumer — the same "promote to shared/ when a second consumer appears" move
that ``grounding.py`` and ``embeddings.py`` went through. The politeness client
and the fail-closed robots policy are generic: neither knows anything about jobs
or courses, only about talking to the internet without being rude.

Item dataclasses and adapters stay per-agent (a ``RawPosting`` and a
``RawCourse`` have almost no fields in common), so this package deliberately
holds only the transport and robots layers.
"""

from .http import Blocked, PoliteClient, ResponseTooLarge, SourcePolicy, host_of
from .robots import RobotsDecision, RobotsPolicy

__all__ = [
    "Blocked",
    "PoliteClient",
    "ResponseTooLarge",
    "RobotsDecision",
    "RobotsPolicy",
    "SourcePolicy",
    "host_of",
]
