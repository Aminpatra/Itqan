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

    # ---- what the candidate asked for ------------------------------------
    # Retrieval preferences, collected by the web app while Agent A was reading
    # the documents. They shape WHICH postings this candidate is compared
    # against — never whether a skill counts as satisfied, which is grounded
    # evidence and not a matter of preference.
    preferred_role: Optional[str]        # --preferred-role; takes the title slot
    roles_only: bool                     # --roles-only; the role REPLACES the headline
    preferred_arrangement: Optional[str]  # --preferred-arrangement; text bias only

    # ---- build_query_embedding -------------------------------------------
    profile: Any                       # CandidateProfile
    candidate_skills: list[str]        # accepted skill names, order preserved
    # The accepted-skill dicts as Agent A published them, carrying quality /
    # evidence_type / origin. Used to stop a weakly-evidenced claim closing a gap.
    candidate_skill_records: list[dict[str, Any]]
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
