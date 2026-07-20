"""Skill quality judging.

Two jobs, and they are separate: filtering out low-signal claims, and checking the
surviving ones against evidence elsewhere in the documents.

Rejected skills are kept in the output with their rationale rather than deleted.
A downstream agent may reasonably disagree with a call, and silent filtering is
indistinguishable from a bug when the numbers look wrong later.
"""

from __future__ import annotations

from langchain_core.prompts import PromptTemplate

SKILL_JUDGE_PROMPT = PromptTemplate(
    input_variables=["skills_json", "evidence_context"],
    template="""You assess whether claimed skills are worth passing to a job-matching system.

REJECT (keep = false, category = "generic") — self-assessed traits and universal
baseline competencies. These are unfalsifiable, everyone claims them, and they carry
no matching signal:
  communication, teamwork, team player, hard worker, fast learner, self-motivated,
  leadership, time management, problem solving, critical thinking, attention to detail,
  flexibility, adaptability, work under pressure, Microsoft Word, Microsoft Office,
  "computer skills", "internet", "email", typing

KEEP (keep = true) — skills that are specific and verifiable:
  named technologies, languages, frameworks, tools (Python, PyTorch, Siemens TIA Portal)
  named methods or domains (finite element analysis, Arabic NLP, IFRS reporting)
  certifications and licensed competencies (CCNA, PMP, forklift licence)
  spoken languages (category "language")

EVIDENCE CROSS-CHECK — set `evidence_type` from where the skill is actually corroborated
in EVIDENCE CONTEXT, and quote it in `evidence_quote`:
  "project"       - used in a named project
  "experience"    - used in a role's responsibilities
  "course"        - taught in a completed course, or in a certification's curriculum
  "certification" - the certification itself IS the skill (e.g. claimed skill "CCNA")
  "claim_only"    - appears only in a skills list, nothing corroborates it

EVIDENCE PRECEDENCE — check the strongest source first, and stop there:
  1. Used in a named PROJECT or a JOB  -> evidence_type "project"/"experience", quality "high"
  2. Otherwise, taught in a COURSE or a credential's CURRICULUM -> "course", quality "medium"
  3. Otherwise -> "claim_only", quality "low"

A skill demonstrated in a project stays "high" even when a course also happens to teach
it. Curriculum evidence can only ever RAISE a rating, never lower one. Do not let a
matching syllabus entry pull a skill down from project evidence you already found —
scan the projects and experience section first, and only reach for curriculum when
nothing there covers the skill.

CURRICULUM ENTRIES are background about what a credential normally teaches. They are
NOT statements from the candidate's documents. Use them like this:
- A claimed skill that appears in the curriculum of a course or certification the
  candidate completed IS corroborated. Set evidence_type = "course", set
  `corroborating_credential` to that credential's name, and say so in the rationale.
  Example: claimed "SQL" + curriculum of "Introduction to Database" lists SQL
  -> keep, evidence_type "course", quality "medium".
- Curriculum corroboration caps quality at "medium". Only a project, a job, or the
  certification being the skill itself can justify "high". Having studied a tool is
  real evidence, and it is weaker than having built something with it — preserve that
  distinction rather than collapsing it.
- A curriculum entry can never introduce a skill. If a syllabus covers something the
  candidate did not claim, ignore it completely. You are ruling on claims, not
  inferring what the candidate probably knows.
- Credentials absent from CURRICULUM are ones whose content is unknown. Their titles
  alone corroborate nothing.

QUALITY:
  "high"   - used in a project or a job, or the certification is itself the skill
  "medium" - taught in a completed course or a credential's curriculum
  "low"    - claim_only, or too vague to match on ("programming", "databases", "software")

TWO REJECTION REASONS, AND ONLY TWO:
  (a) generic self-assessed trait — the list above.
  (b) the skill NAME is too vague to match a job requirement against:
      "programming", "databases", "software", "engineering", "IT", "design".

"Too vague" is a judgement about the WORDING OF THE NAME, never about whether
evidence was found. A named language, product, standard or tool is never vague:
MATLAB, Siemens TIA Portal, SCADA, AutoCAD, PyTorch, IFRS and CCNA are all
specific, and all of them are KEEP even with no corroborating evidence at all.

MISSING EVIDENCE IS NOT A REJECTION REASON. Many CVs list real skills without
describing where they were used, and a skills section is often all a short CV has.
When nothing corroborates a specific, named skill, the correct output is:
    keep = true, evidence_type = "claim_only", quality = "low"
That records the weakness accurately and lets the downstream system weigh it.
Setting keep = false there discards true information and is a mistake.

If EVIDENCE CONTEXT is empty, expect to keep nearly every specific skill, each as
claim_only. An empty evidence section means the CV was thin, not that the
candidate's skills are fabricated.

`rationale` should be one short clause explaining the call.

Judge only the skills listed. Never add a skill, however strongly the evidence
suggests one — extraction already happened, and this stage is not allowed to
introduce new claims.

CLAIMED SKILLS:
{skills_json}

EVIDENCE CONTEXT (projects, experience, courses, certifications):
{evidence_context}
""",
)
