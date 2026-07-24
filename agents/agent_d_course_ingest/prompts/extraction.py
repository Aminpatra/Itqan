"""Course extraction prompt: course text in, taught skills out.

Deliberately parallel to Agent B's job-extraction prompt, because the two skill
lists must aggregate against each other on the SAME vocabulary — a demand skill
and a supply skill for the same concept have to come out as the same string, or
the demand-vs-supply join is meaningless. So the same "short canonical English
name, recruiter search box" discipline applies, with the one difference that
here the skills are what the course TEACHES rather than what a job requires.
"""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

SYSTEM = """You extract the skills a course TEACHES, for a labour-market database \
that compares skill supply (courses) against skill demand (job postings). A \
fabricated skill becomes a fabricated statistic, so use ONLY what the course text \
states or plainly covers. If it is unclear, return less — a short honest list \
beats a padded one.

- taught_skills: the concrete skills a learner gains, each as a SHORT CANONICAL \
NAME of 1-4 words — the form a recruiter would type in a search box (e.g. \
"machine learning", "python", "financial analysis"). Name the skill, do not copy \
a whole sentence. One skill per entry. Keep proper nouns exactly (a tool, \
language, or framework name is already canonical). Do NOT add skills the course \
does not actually teach, and omit generic filler ("critical thinking", \
"career growth"). Name every skill in ENGLISH so it aggregates with the demand \
side.

- level: 'beginner', 'intermediate', or 'advanced' ONLY if the course states or \
clearly implies it (e.g. "no prior experience needed" -> beginner). Null otherwise.

- subject: one broad domain phrase if clear (e.g. "data science", "web \
development"), else null."""

HUMAN = """COURSE: {name}
PROVIDER: {provider}

DESCRIPTION:
{body}

Extract the skills this course teaches. Return null for level and subject if the \
course does not make them clear."""

EXTRACTION_PROMPT = ChatPromptTemplate.from_messages([("system", SYSTEM), ("human", HUMAN)])
