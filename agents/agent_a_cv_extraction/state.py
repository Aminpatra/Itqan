"""Graph state — the shared blackboard every node reads and writes.

Reducers are on exactly three fields (``errors``, ``warnings``, ``artifacts``,
``human_validation_notes``): those are append-only, and any node may contribute.
Everything else is last-write-wins, which is correct because the graph is linear
apart from one bounded HITL loop.

Extraction results are stored as plain ``dict``, never Pydantic instances. The
checkpointer serializes state on every superstep; dicts avoid msgpack round-trip
surprises and keep the ``__interrupt__`` payload directly JSON-printable.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, Optional, TypedDict

STATE_VERSION = "itqan.agent_a_state/1.0"

DocKind = Literal["pdf_text", "pdf_scanned", "image", "text", "unsupported"]


class DocRecord(TypedDict, total=False):
    path: str
    role: Literal["cv", "transcript"]
    kind: DocKind
    text: str
    ocr_json_path: Optional[str]
    mean_confidence: Optional[float]
    pages: int
    # Per-block OCR output, kept in state so gap detection can map a weakly
    # recognised region back to the field it produced.
    blocks: list[dict[str, Any]]


class Gap(TypedDict, total=False):
    field_path: str
    reason: str
    prompt_hint: str
    current_value: Optional[str]
    ocr_confidence: Optional[float]


class AgentState(TypedDict, total=False):
    # ---- inputs ----------------------------------------------------------
    cv_path: str
    transcript_path: Optional[str]
    run_id: str
    output_dir: str
    interactive: bool          # False under --no-hitl: gaps are recorded, not asked

    # ---- ingestion -------------------------------------------------------
    cv_doc: DocRecord
    transcript_doc: Optional[DocRecord]

    # ---- extraction ------------------------------------------------------
    cv_extraction: dict[str, Any]
    transcript_extraction: Optional[dict[str, Any]]

    # ---- verification ----------------------------------------------------
    grounding_report: dict[str, Any]     # field_path -> provenance dict
    dropped_fields: Annotated[list[str], operator.add]

    # ---- human in the loop ----------------------------------------------
    gaps: list[Gap]
    human_input: dict[str, str]
    human_supplied_fields: Annotated[list[str], operator.add]
    review_rounds: int
    human_validation_notes: Annotated[list[str], operator.add]

    # ---- judging and output ---------------------------------------------
    # What the candidate's courses/certificates typically teach. Background
    # knowledge about syllabi, used to corroborate claimed skills — never a
    # source of new skills. See nodes/curriculum.py.
    curriculum: list[dict[str, Any]]
    accepted_skills: list[dict[str, Any]]
    rejected_skills: list[dict[str, Any]]
    summary: dict[str, Any]
    final_output_path: str

    # ---- cross-cutting ---------------------------------------------------
    warnings: Annotated[list[str], operator.add]
    errors: Annotated[list[str], operator.add]
    artifacts: Annotated[list[str], operator.add]
    trace: Annotated[list[str], operator.add]
