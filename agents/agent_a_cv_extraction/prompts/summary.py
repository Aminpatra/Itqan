"""Few-shot summarization — the final node before persistence.

Summarization is where a careful extraction pipeline usually goes wrong: every
prior stage is constrained by a schema, and then the last step is handed a blank
page and asked to write prose. Rules alone do not hold here, because "be factual"
is exactly what a model believes it is doing while padding a thin CV into a
flattering paragraph.

So the technique is few-shot, and the examples are chosen to demonstrate restraint
rather than fluency:

  Example 1 (rich)   - the normal case; shows the expected register and density.
  Example 2 (sparse) - the important one. A two-line CV whose gold output stays two
                       lines: null academic_performance, "No work experience recorded",
                       core_strengths of length 2, four honest gaps. This is what stops
                       the model reaching for "eager to grow" filler.
  Example 3 (OCR)    - a profile with a human-corrected field and a low-confidence one;
                       shows uncertainty surfacing in gaps_or_unknowns instead of being
                       smoothed over.

Across all three, no gold output contains a fact absent from its input profile.

The examples span different fields on purpose. Few-shot examples teach domain as
well as form: three computing CVs would leave the model treating software work as
the default shape of a career, and quietly reaching for technical framing when
handed a nurse, an accountant or a teacher. The behaviour being taught here —
restraint, honest gaps, no invented detail — is not specific to any profession,
so the examples should not be either.
"""

from __future__ import annotations

from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate

PREFIX = """You write factual candidate summaries for a downstream hiring agent.

RULES
- Every sentence must be supported by the structured profile provided.
- Do not add praise adjectives that are not evidenced. No "passionate", "highly skilled",
  "excellent communicator", "eager to learn".
- Do not infer seniority, years of experience, or proficiency levels that are not stated.
- Do not invent context: if the profile does not say why someone left a role or what a
  company does, say nothing about it.
- If information is absent, list it under gaps_or_unknowns rather than quietly omitting
  it or filling it in. A short summary with honest gaps is more useful to the downstream
  agent than a polished one with invented detail.
- core_strengths must be drawn only from the accepted skills given. Do not pad the list
  to a target length.
- Write in the third person. Use the candidate's name once, then pronouns.
- PRONOUNS: use they/them unless the documents explicitly state otherwise. Never infer
  pronouns from a name. A name does not tell you someone's gender, and guessing wrong
  misgenders a real person in a way the neutral default never does. This applies no
  matter how strongly a name reads as gendered to you.

Below are examples of the expected output."""

EXAMPLES = [
    {
        "input_profile": """name: Sara Al-Balushi
education: BSc Computer Science, Sultan Qaboos University, 2020-2024, GPA 3.71/4.0
courses: Data Structures (A), Machine Learning (A-), Database Systems (B+), Computer Networks (A)
experience: Data Analyst Intern, Omantel, Jun-Sep 2023 - built Power BI dashboards for churn
  reporting; automated monthly ETL in Python, cutting manual prep from 2 days to 3 hours
projects: "ArabicNER" - NER for Arabic news, tech: Python, PyTorch, HuggingFace Transformers
accepted_skills: Python (high/project), PyTorch (high/project), SQL (medium/experience),
  Power BI (high/experience), Arabic NLP (high/project)""",
        "output_summary": """headline: Computer Science graduate with applied NLP and data analytics experience
profile: Sara Al-Balushi holds a BSc in Computer Science from Sultan Qaboos University
  (2024) with a 3.71/4.0 GPA. Their work centres on Python-based data engineering and
  Arabic NLP. During a 2023 internship at Omantel they built Power BI churn dashboards and
  automated a monthly ETL pipeline, reducing manual preparation from two days to three
  hours. Their ArabicNER project applied PyTorch and HuggingFace Transformers to
  named-entity recognition on Arabic news text.
core_strengths: ["Python data engineering", "Arabic NLP with PyTorch/Transformers",
  "BI reporting and ETL automation", "SQL and database fundamentals"]
education_summary: BSc Computer Science, Sultan Qaboos University, 2020-2024, GPA 3.71/4.0.
  Strongest coursework in Data Structures, Machine Learning and Computer Networks (A/A-).
experience_summary: One 4-month data analytics internship at a national telecom operator.
academic_performance: GPA 3.71/4.0; A or A- in all core CS and ML coursework.
gaps_or_unknowns: ["No full-time professional experience",
  "No stated cloud or deployment experience"]""",
    },
    {
        "input_profile": """name: Ahmed Rashid
education: Diploma in Information Technology, Higher College of Technology, end 2022,
  GPA not stated
courses: (none extracted)
experience: (none)
projects: "Portfolio website" - tech: HTML, CSS
accepted_skills: HTML (medium/project), CSS (medium/project)
rejected_skills: teamwork (generic), communication (generic), Microsoft Word (generic)""",
        "output_summary": """headline: IT diploma holder with introductory front-end project work
profile: Ahmed Rashid completed a Diploma in Information Technology at the Higher College
  of Technology in 2022. The available documents show one personal project, a portfolio
  website built with HTML and CSS. No professional experience is recorded in the source
  documents.
core_strengths: ["HTML", "CSS"]
education_summary: Diploma in Information Technology, Higher College of Technology,
  completed 2022.
experience_summary: No work experience recorded in the source documents.
academic_performance: null
gaps_or_unknowns: ["GPA not stated", "No transcript provided - coursework unknown",
  "No professional experience recorded",
  "No programming languages beyond markup/styling"]""",
    },
    {
        "input_profile": """name: Mariam Haddad
contact: email m.haddad@example.com [human-supplied - OCR confidence was 0.41]
education: BSc Nursing, University of Jordan, 2017-2021
transcript: BSc Nursing, University of Jordan, CGPA 3.15/4.0
courses: Pharmacology (B+), Medical-Surgical Nursing (A-), Community Health Nursing (B)
experience: Staff Nurse, Al-Bashir Hospital, 2021-Present - medical-surgical ward;
  medication administration, patient assessment, wound care
accepted_skills: medication administration (high/experience), patient assessment
  (high/experience), wound care (high/experience), pharmacology (medium/course),
  BLS certification (high/certification)
low_confidence_fields: ["contact.phone"]""",
        "output_summary": """headline: Registered nurse with four years on a medical-surgical ward
profile: Mariam Haddad holds a BSc in Nursing from the University of Jordan (2021, CGPA
  3.15/4.0). They have worked as a Staff Nurse at Al-Bashir Hospital since 2021 on a
  medical-surgical ward, where their responsibilities cover medication administration,
  patient assessment and wound care. They hold a current BLS certification.
core_strengths: ["Medication administration", "Patient assessment", "Wound care",
  "Pharmacology", "BLS-certified"]
education_summary: BSc Nursing, University of Jordan, 2017-2021, CGPA 3.15/4.0.
  Strongest coursework in Medical-Surgical Nursing (A-).
experience_summary: Four years as a Staff Nurse on a medical-surgical ward at a single
  hospital; exact start month not stated.
academic_performance: CGPA 3.15/4.0.
gaps_or_unknowns: ["Phone number was low-confidence in OCR and not confirmed",
  "Exact start month of current role not stated",
  "No stated specialisation or ICU/critical care experience"]""",
    },
]

EXAMPLE_PROMPT = PromptTemplate(
    input_variables=["input_profile", "output_summary"],
    template="PROFILE:\n{input_profile}\n\nSUMMARY:\n{output_summary}",
)

SUMMARY_PROMPT = FewShotPromptTemplate(
    examples=EXAMPLES,
    example_prompt=EXAMPLE_PROMPT,
    prefix=PREFIX,
    suffix="PROFILE:\n{input_profile}\n\nSUMMARY:",
    input_variables=["input_profile"],
    example_separator="\n\n---\n\n",
)
