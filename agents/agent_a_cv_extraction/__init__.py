"""Agent A — CV and transcript extraction.

Produces a ``shared.contracts.CandidateProfile`` envelope. Downstream agents
should import that contract, never this package's internals.
"""

from .graph import build_graph

__all__ = ["build_graph"]
