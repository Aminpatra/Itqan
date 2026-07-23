"""Graph state for Agent C.

Linear, deterministic, no interrupt — so no checkpointer, no serialization
between supersteps, and state can carry live pydantic objects
(``CandidateProfile``, ``JobPostingExport``) directly. The only reducer is on
``warnings``, which several nodes may append to.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Optional, TypedDict

STATE_VERSION = "itqan.agent_c_state/1.0"


class GapState(TypedDict, total=False):
    # ---- inputs ----------------------------------------------------------
    profile_path: str
    user_id: str
    top_k: int
    sector_override: Optional[str]     # --sector; overrides modal inference
    output_dir: str
    run_id: str

    # ---- build_query_embedding -------------------------------------------
    profile: Any                       # CandidateProfile
    candidate_skills: list[str]        # accepted skill names, order preserved
    essence: str
    query_embedding: list[float]

    # ---- retrieve_postings ----------------------------------------------
    postings: list[Any]                # all retrieved JobPostingExport rows
    stats: list[Any]                   # latest-window SkillDemandStatRow rows
    usable_postings: list[Any]         # similarity >= agent_c_match_threshold
    used_fallback: bool
    inferred_sector: Optional[str]

    # ---- map_candidate_skills -------------------------------------------
    candidate_mappings: list[Any]      # SkillMapping per accepted skill

    # ---- gap_analysis ----------------------------------------------------
    matched_jobs: list[dict[str, Any]]
    fallback_sector_gap: Optional[dict[str, Any]]
    aggregate: dict[str, Any]

    # ---- persist ---------------------------------------------------------
    output_path: str

    warnings: Annotated[list[str], operator.add]
