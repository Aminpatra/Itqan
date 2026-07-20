"""Curriculum research — runs immediately before skill judging.

Courses and certificates are evidence, but only once unpacked. "Introduction to
Database" corroborates a claimed SQL skill; "Machine Learning Specialization"
corroborates scikit-learn and NumPy. Left as bare titles, both read to the judge
as no evidence at all, and genuinely-backed skills get scored as bare claims.

The node is skipped entirely when there is nothing to research, so a CV with no
coursework and no certificates costs nothing extra.
"""

from __future__ import annotations

import json
from typing import Any

from shared.config import Config
from shared.llm import as_dict, structured

from ..prompts import CURRICULUM_RESEARCH_PROMPT
from ..schemas import CurriculumResearch
from ..state import AgentState

# Titles below this length are almost always OCR debris rather than a credential.
MIN_TITLE_LEN = 4

# Cap the research call. A full degree transcript can carry 40+ rows and the
# prompt gets unwieldy well before the marginal course earns its place.
MAX_CREDENTIALS = 40

def _priority(credential: dict[str, str], claimed: set[str]) -> int:
    """Sort key for the research budget: lower is researched first.

    Ordering matters only when there are more credentials than the cap allows.
    It matters *then* because a transcript arrives in chronological order, which
    puts the advanced, most-corroborating courses last — exactly where a naive
    truncation removes them.

    Deliberately free of any opinion about which subjects are worth researching.
    An earlier version carried a stoplist of "low-yield" title words, which
    encoded one culture's and one discipline's assumptions: it would have
    deprioritised the language, humanities and professional-practice courses that
    are the whole point of the transcript for a translator, a lawyer or a nurse.
    The only signals used here come from the candidate's own documents.
    """
    # Certifications are few, deliberately obtained, and usually the strongest
    # evidence available. Never let a long transcript crowd them out.
    if credential["credential_kind"] == "certification":
        return 0
    # A course whose title shares a word with something the candidate claimed is
    # the most likely to corroborate it — "Database Systems" for a claimed SQL,
    # "Pharmacology" for a claimed pharmacokinetics.
    title_words = {w for w in credential["credential_name"].casefold().split() if len(w) > 3}
    if claimed and title_words & claimed:
        return 1
    return 2


def collect_credentials(state: AgentState) -> list[dict[str, str]]:
    """Every named course and certificate available, de-duplicated."""
    extraction = state.get("cv_extraction") or {}
    transcript = state.get("transcript_extraction") or {}

    credentials: list[dict[str, str]] = []
    seen: set[str] = set()

    def add(name: Any, kind: str) -> None:
        if not isinstance(name, str):
            return
        title = name.strip()
        if len(title) < MIN_TITLE_LEN or title.casefold() in seen:
            return
        seen.add(title.casefold())
        credentials.append({"credential_name": title, "credential_kind": kind})

    for cert in extraction.get("certifications") or []:
        if isinstance(cert, dict):
            add(cert.get("name"), "certification")

    for course in list(extraction.get("courses") or []) + list(transcript.get("courses") or []):
        if isinstance(course, dict):
            add(course.get("title"), "course")

    # Words from the claimed skills, used to float likely corroborators to the top.
    claimed = {
        word
        for skill in extraction.get("skills") or []
        if isinstance(skill, dict) and isinstance(skill.get("name"), str)
        for word in skill["name"].casefold().split()
        if len(word) > 3
    }

    # Stable sort: within a priority band the document's own order is preserved.
    credentials.sort(key=lambda c: _priority(c, claimed))
    return credentials[:MAX_CREDENTIALS]


def make_research_curriculum_node(llm: Any, config: Config):
    chain = CURRICULUM_RESEARCH_PROMPT | structured(llm, CurriculumResearch)

    def research_curriculum(state: AgentState) -> dict[str, Any]:
        credentials = collect_credentials(state)
        if not credentials:
            return {"curriculum": [], "trace": ["research_curriculum(none)"]}

        # Nothing to corroborate means nothing to research.
        if not (state.get("cv_extraction") or {}).get("skills"):
            return {"curriculum": [], "trace": ["research_curriculum(no skills)"]}

        try:
            result = as_dict(
                chain.invoke(
                    {"credentials": json.dumps(credentials, indent=2, ensure_ascii=False)}
                )
            )
        except Exception as exc:
            # Enrichment, not a requirement: judging still works on titles alone.
            return {
                "curriculum": [],
                "warnings": [f"Curriculum research failed ({exc}); judging on titles alone."],
                "trace": ["research_curriculum(failed)"],
            }

        known_titles = {c["credential_name"].casefold() for c in credentials}
        entries = [
            entry
            for entry in result.get("credentials", [])
            # Drop anything the model invented rather than was asked about, and
            # anything it admitted not knowing — an unrecognised credential
            # carries no curriculum and must not reach the judge as if it did.
            if entry.get("credential_name", "").casefold() in known_titles
            and entry.get("recognized")
            and (entry.get("typical_skills") or entry.get("typical_concepts"))
        ]

        unrecognised = len(credentials) - len(entries)
        return {
            "curriculum": entries,
            "trace": [
                f"research_curriculum({len(entries)} known / {unrecognised} unknown)"
            ],
        }

    return research_curriculum
