"""Graph state for Agent E.

Linear and deterministic, so no checkpointer and state carries live objects
(``CourseCandidate``) directly between nodes. The only reducer is on
``warnings``, which several nodes may append to.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

STATE_VERSION = "itqan.agent_e_state/1.0"


class RecommendState(TypedDict, total=False):
    # ---- inputs ----------------------------------------------------------
    gap_path: str
    user_id: str
    output_dir: str
    run_id: str

    # ---- load_missing_skills --------------------------------------------
    # Each: {"skill": str, "esco_code": str|None, "priority_score": float}
    missing: list[dict[str, Any]]
    used_fallback: bool

    # ---- retrieve_candidate_courses -------------------------------------
    courses_by_id: dict[str, Any]            # course_id -> CourseCandidate
    course_covers: dict[str, list[str]]      # course_id -> skills (this set) it teaches
    skill_candidates: dict[str, list[str]]   # skill -> candidate course_ids (ordered)

    # ---- greedy_cover_assign --------------------------------------------
    # course_id -> the skills it was assigned to cover (>=1); deterministic
    assigned: dict[str, list[str]]
    no_course_found_skills: list[str]

    # ---- attach_flags ----------------------------------------------------
    recommendations: list[dict[str, Any]]    # persist shape; rationale filled later

    # ---- persist ---------------------------------------------------------
    output_path: str

    warnings: Annotated[list[str], operator.add]
