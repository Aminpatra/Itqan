"""Prompts for Agent C.

There is exactly one, and its existence is a deliberate exception: every other
part of this agent is arithmetic. See prompts/skill_match.py for why string
mathematics could not answer the question it answers.
"""

from .skill_match import SKILL_MATCH_PROMPT

__all__ = ["SKILL_MATCH_PROMPT"]
