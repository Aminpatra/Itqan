"""Prompt templates. One module per LLM call the graph makes."""

from .curriculum import CURRICULUM_RESEARCH_PROMPT
from .extraction import (
    CV_EXTRACTION_PROMPT,
    TRANSCRIPT_EXTRACTION_PROMPT,
    quality_note,
)
from .human_validation import HUMAN_INPUT_VALIDATION_PROMPT
from .skills import SKILL_JUDGE_PROMPT
from .summary import SUMMARY_PROMPT
from .verification import GROUNDING_ADJUDICATION_PROMPT

__all__ = [
    "CV_EXTRACTION_PROMPT",
    "TRANSCRIPT_EXTRACTION_PROMPT",
    "quality_note",
    "GROUNDING_ADJUDICATION_PROMPT",
    "CURRICULUM_RESEARCH_PROMPT",
    "SKILL_JUDGE_PROMPT",
    "HUMAN_INPUT_VALIDATION_PROMPT",
    "SUMMARY_PROMPT",
]
