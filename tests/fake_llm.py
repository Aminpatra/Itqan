"""An offline stand-in for ChatOpenAI.

Returns canned instances keyed by the schema it was asked to produce, which is
enough to exercise the entire graph — routing, reducers, the interrupt/resume
cycle, envelope validation — without a network call or an API key.

It deliberately returns one hallucinated field (a fabricated phone number that
appears nowhere in the fixture) so the grounding pass has something real to catch.
"""

from __future__ import annotations

from typing import Any

from agents.agent_a_cv_extraction.schemas import (
    CandidateSummary,
    CredentialCurriculum,
    CurriculumResearch,
    CVExtraction,
    GroundingReport,
    HumanInputValidation,
    SkillJudgement,
    SkillVerdict,
    TranscriptExtraction,
    ValidatedField,
)


class FakeStructuredLLM:
    def __init__(self, **overrides: Any) -> None:
        self.calls: list[str] = []
        self.overrides = overrides

    def with_structured_output(self, schema: type, **kwargs: Any) -> Any:
        """Return a real Runnable.

        The graph composes chains as ``prompt | structured(llm, Schema)``, and
        LCEL's ``__or__`` only accepts a Runnable, a plain callable, or a dict —
        a bespoke object with ``__ror__`` is rejected before ``__ror__`` is ever
        consulted. RunnableLambda is the supported way in.
        """
        from langchain_core.runnables import RunnableLambda

        def _invoke(payload: Any) -> Any:
            self.calls.append(schema.__name__)
            return self.respond(schema, payload)

        return RunnableLambda(_invoke)

    def respond(self, schema: type, payload: Any) -> Any:
        if schema.__name__ in self.overrides:
            return self.overrides[schema.__name__]

        if schema is CVExtraction:
            return CVExtraction(
                full_name="Sara Al-Balushi",
                contact={
                    "email": "sara.b@squ.edu.om",
                    # Not in the fixture. The grounder must drop this.
                    "phone": "+968 9999 0000",
                    "location": "Muscat, Oman",
                },
                skills=[
                    {"name": "Python", "source_span": "Python", "category": "technical"},
                    {"name": "PyTorch", "source_span": "PyTorch", "category": "technical"},
                    {"name": "Teamwork", "source_span": "Teamwork", "category": "soft"},
                ],
                education=[
                    {
                        "institution": "Sultan Qaboos University",
                        "degree": "BSc Computer Science",
                        "start_date": "2020",
                        "end_date": "2024",
                        "gpa": "3.71",
                    }
                ],
                projects=[
                    {
                        "name": "ArabicNER",
                        "description": "NER for Arabic news",
                        "technologies": ["Python", "PyTorch"],
                    }
                ],
            )

        if schema is TranscriptExtraction:
            return TranscriptExtraction(
                student_name="Sara Al-Balushi",
                institution="Sultan Qaboos University",
                cgpa="3.71",
                courses=[{"title": "Machine Learning", "grade": "A-"}],
            )

        if schema is GroundingReport:
            return GroundingReport(verdicts=[])

        if schema is CurriculumResearch:
            # One recognised credential, one deliberately unrecognised, and one
            # the caller never asked about — the node must keep only the first.
            return CurriculumResearch(
                credentials=[
                    CredentialCurriculum(
                        credential_name="Machine Learning",
                        credential_kind="course",
                        recognized=True,
                        typical_skills=["scikit-learn", "NumPy", "PyTorch"],
                        typical_concepts=["supervised learning", "gradient descent"],
                    ),
                    CredentialCurriculum(
                        credential_name="Campus Coding Marathon 2024",
                        credential_kind="certification",
                        recognized=False,
                        typical_skills=[],
                        typical_concepts=[],
                    ),
                    CredentialCurriculum(
                        credential_name="Fabricated Course Nobody Asked About",
                        credential_kind="course",
                        recognized=True,
                        typical_skills=["Kubernetes"],
                        typical_concepts=["orchestration"],
                    ),
                ]
            )

        if schema is SkillJudgement:
            return SkillJudgement(
                verdicts=[
                    SkillVerdict(
                        name="Python",
                        keep=True,
                        quality="high",
                        category="technical",
                        evidence_type="project",
                        evidence_quote="ArabicNER",
                        rationale="used in a named project",
                    ),
                    SkillVerdict(
                        name="PyTorch",
                        keep=True,
                        quality="high",
                        category="technical",
                        evidence_type="project",
                        evidence_quote="ArabicNER",
                        rationale="used in a named project",
                    ),
                    SkillVerdict(
                        name="Teamwork",
                        keep=False,
                        quality="low",
                        category="generic",
                        evidence_type="claim_only",
                        rationale="generic self-assessed trait, no matching signal",
                    ),
                ]
            )

        if schema is HumanInputValidation:
            # Echo whatever was asked, accepting everything except obvious junk.
            import json
            import re

            raw = {}
            text = payload.text if hasattr(payload, "text") else str(payload)
            match = re.search(r"USER ANSWERS:\s*(\{.*)", text, re.S)
            if match:
                try:
                    raw = json.loads(match.group(1))
                except json.JSONDecodeError:
                    raw = {}
            return HumanInputValidation(
                fields=[
                    ValidatedField(
                        field_path=path,
                        normalized_value=str(value).strip(),
                        accepted=str(value).strip().casefold()
                        not in {"asdf", "n/a", "idk", "none", "-"},
                        issue=None,
                    )
                    for path, value in raw.items()
                ]
            )

        if schema is CandidateSummary:
            return CandidateSummary(
                headline="Computer Science graduate with applied NLP experience",
                profile="Sara Al-Balushi holds a BSc in Computer Science from Sultan "
                "Qaboos University with a 3.71 GPA.",
                core_strengths=["Python", "PyTorch"],
                education_summary="BSc Computer Science, Sultan Qaboos University, 2020-2024.",
                experience_summary="No work experience recorded in the source documents.",
                academic_performance="GPA 3.71",
                gaps_or_unknowns=["No professional experience recorded"],
            )

        raise AssertionError(f"FakeStructuredLLM has no canned response for {schema!r}")
