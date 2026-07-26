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

KEEP (keep = true) — skills that are specific and verifiable, in ANY profession:
  named tools, technologies, instruments, equipment or software
  named methods, procedures, techniques, assays or analyses
  named standards, frameworks, regulations or bodies of practice
  certifications, licences and registrations
  spoken languages (category "language")

EVIDENCE CROSS-CHECK — set `evidence_type` from where the skill is actually corroborated
in EVIDENCE CONTEXT, and quote it in `evidence_quote`:
  "project"       - used in a named project
  "experience"    - used in a role's responsibilities
  "course"        - taught in a completed course, or in a certification's curriculum
  "certification" - the claimed skill IS a certification the candidate holds
  "adjacent"      - not directly corroborated, but standard equipment of a domain the
                    candidate HAS demonstrated evidence in (see ADJACENT EVIDENCE below)
  "claim_only"    - appears only in a skills list, and is not standard equipment of any
                    domain the candidate has demonstrated

EVIDENCE PRECEDENCE — check the strongest source first, and stop there:
  1. Used in a named PROJECT or a JOB  -> evidence_type "project"/"experience", quality "high"
  2. Otherwise, taught in a COURSE or a credential's CURRICULUM -> "course", quality "medium"
  3. Otherwise, standard equipment of a domain the candidate DEMONSTRATED elsewhere
     -> "adjacent", quality "medium" (see ADJACENT EVIDENCE below)
  4. Otherwise -> "claim_only", quality "low"

A skill demonstrated in a project stays "high" even when a course also happens to teach
it. Curriculum evidence can only ever RAISE a rating, never lower one. Do not let a
matching syllabus entry pull a skill down from project evidence you already found —
scan the projects and experience section first, and only reach for curriculum when
nothing there covers the skill.

CURRICULUM ENTRIES are background about what a credential normally teaches. They are
NOT statements from the candidate's documents. Use them like this:
- A "CORROBORATES THESE CLAIMED SKILLS" line has already been checked for you. Every
  skill it names IS corroborated by that credential, which the candidate completed.
  Set evidence_type = "course", quality = "medium", and `corroborating_credential` to
  that credential's name. Do this even when the skill's exact wording does not appear in
  the "usually teaches" list — the semantic match was the point of that line, and
  re-litigating it by looking for the literal string undoes the work.
- A skill named there must NOT be left as claim_only/low. Completing a course that
  teaches something is real evidence, and a transcript is the strongest documentary
  evidence a student has. Ignoring it discards the main reason the transcript was asked
  for at all.
- Where a grade is shown, the candidate completed the course to that standard. Weigh it
  as you would reading the transcript yourself; do not assume any particular scale.
- Curriculum corroboration caps quality at "medium". Only a project, a job, or the
  certification being the skill itself can justify "high". Having studied a tool is
  real evidence, and it is weaker than having built something with it — preserve that
  distinction rather than collapsing it.
- A curriculum entry can never introduce a skill. If a syllabus covers something the
  candidate did not claim, ignore it completely. You are ruling on claims, not
  inferring what the candidate probably knows.
- Credentials absent from CURRICULUM are ones whose content is unknown. Their titles
  alone corroborate nothing.

ADJACENT EVIDENCE — DO NOT default a specific named tool to claim_only when the candidate has
clearly demonstrated its domain. That is the single most common illogical rating, and this step
exists to stop it: a proven programmer marked claim_only on "VS Code", a seven-year accountant
marked claim_only on "Excel". Fix it as follows.

For every skill you are about to mark claim_only, run this check FIRST:
  1. What domain(s) has this candidate DEMONSTRATED? (their high-rated project/experience skills.)
  2. Is this skill a near-universal STAPLE of that demonstrated domain — a tool essentially every
     practitioner of it uses day to day?
  If yes -> evidence_type "adjacent", quality "medium", and set `evidence_quote` to the project or
  role that demonstrates the domain. Only if no -> claim_only.

Lift these CONFIDENTLY (do not hedge — a demonstrated practitioner obviously uses their trade's
staples). The pattern is identical in every field; only the vocabulary changes:
  - demonstrated software developer -> VS Code / their IDE, Git / version control, the Linux shell
  - demonstrated accountant -> Microsoft Excel, and the ledger package their work runs on
  - demonstrated nurse -> the everyday clinical instruments and bedside procedures of that ward
  - demonstrated lawyer -> the standard drafting and court-filing tools of that practice
  - demonstrated translator -> a CAT tool of that language pair
  - demonstrated designer -> the industry-standard design software of that craft
Judge by the candidate's OWN demonstrated domain, never by whether the skill is software.

Bounds, so this lifts the credible and nothing else:
  - The anchor MUST be demonstrated (project / experience / credential) — a skill can never ride on
    another claim_only skill, and generic self-assessed traits are NEVER adjacent (they reject).
  - Be cautious ONLY with several COMPETING alternatives for the same job (e.g. three rival ledger
    packages or design suites): lift the genuine staple, leave the interchangeable rest claim_only.
    Do not chain outward ("uses Python" -> every Python library ever).
  - "adjacent" NEVER reaches "high" — presumed use is weaker than the demonstrated use that earns high.

QUALITY:
  "high"   - used in a project or a job, or the certification is itself the skill
  "medium" - taught in a completed course or a credential's curriculum, OR standard equipment of a
             domain the candidate demonstrated ("adjacent")
  "low"    - claim_only, or too vague to match on ("programming", "databases", "software")

TWO REJECTION REASONS, AND ONLY TWO:
  (a) generic self-assessed trait — the list above.
  (b) the skill NAME is too vague to match a job requirement against:
      "programming", "databases", "software", "engineering", "IT", "design".

"Too vague" is a judgement about the WORDING OF THE NAME, never about whether
evidence was found.

The test: could a recruiter search for this term and get meaningfully filtered
results? A named product, tool, language, standard, procedure, statute or clinical
technique passes and is never vague. A whole field or department name fails.
Judge every profession by that same test — a clinical procedure, a named legal
instrument, a laboratory assay, a teaching methodology and a machine-tool
operation are all as specific as any software library, and are all KEEP even with
no corroborating evidence at all.

MISSING EVIDENCE IS NOT A REJECTION REASON. Many CVs list real skills without
describing where they were used, and a skills section is often all a short CV has.
When a specific, named skill has neither direct evidence NOR an adjacent-domain
anchor (see ADJACENT EVIDENCE), the correct output is:
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
