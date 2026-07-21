"""Curriculum expansion — what does a course or certificate actually teach?

Without this, the skill judge sees course and certificate titles as opaque
strings, and a candidate's transcript contributes nothing to their skill ratings.
The evidence was there; it just needed unpacking.

The step is told which skills the candidate claimed, and asked which of *those*
each credential normally teaches. An earlier version researched blind — emitting
a generic syllabus and hoping its vocabulary matched the candidate's — which
missed most real corroboration, because a syllabus and a CV routinely describe
the same thing in different words.

This is the one place in the pipeline where the model contributes knowledge from
outside the documents, which makes it the one place most able to do damage. Three
guardrails follow from that:

1. Output is *about the credential*, never about the candidate. "This course
   usually covers X" is a statement about a syllabus, not a claim that the
   candidate knows X, and the judge is told to treat it accordingly.
2. Unfamiliar credentials must be declared unfamiliar. A local hackathon or a
   one-day event has no public curriculum, and a plausible guess about it is
   worth strictly less than an admission of ignorance.
3. Nothing here can introduce a new skill. The judge only ever rules on skills
   the candidate actually claimed; curriculum output can raise confidence in one
   of those, never add to the list.
"""

from __future__ import annotations

from langchain_core.prompts import PromptTemplate

CURRICULUM_RESEARCH_PROMPT = PromptTemplate(
    input_variables=["credentials", "claimed_skills"],
    template="""You describe what named courses and certifications typically teach.

For each credential below, list the skills and concepts a holder would normally have
been taught. This is used to corroborate skills a candidate has *already claimed* — it
is background about the syllabus, not a claim about any person.

THE MOST IMPORTANT FIELD IS `covers_claimed_skills`.
The candidate claimed the skills in CLAIMED SKILLS below. For each credential, put into
`covers_claimed_skills` every one of those claimed skills that this credential's normal
syllabus would have taught.

Match on MEANING, not wording. The candidate's phrasing and a syllabus's phrasing are
often different words for the same thing, and a purely lexical comparison misses almost
every real match. A claimed skill belongs in `covers_claimed_skills` when the course
teaches it under any of these relationships:
  - the same thing named differently, including vendor or product names versus the
    generic term, and abbreviations versus what they stand for;
  - a specific component, module or tool of a platform, framework, standard or body of
    practice that the course covers;
  - the practical technique through which the course's subject matter is actually
    taught and assessed.

Ask yourself, for each claimed skill: "would someone who completed this course, in its
normal form, have been taught this?" If yes, include it. Being unsure of the exact
wording overlap is not a reason to exclude — being unsure the topic is taught is.

Copy claimed skills VERBATIM from the CLAIMED SKILLS list into `covers_claimed_skills`.
Never add a skill that is not on that list, no matter what else the course teaches.

RECOGNITION IS THE FIRST DECISION.
Set `recognized = true` only for credentials whose content you actually know: standard
university courses that are taught in much the same form everywhere, and established
certifications or programmes with a published, stable syllabus.

Set `recognized = false`, with EMPTY lists, for anything else. Typical cases: one-off
events, hackathons, competitions, club activities, workshops and seminars; institution-
specific offerings whose content you cannot know from the title; and any title you are
simply unsure about. Attending an event is not evidence of a curriculum.

Do not guess. An honest `recognized = false` is the correct and useful answer here — a
fabricated syllabus would let an unearned skill rating through, which is worse than no
information at all. If the title alone does not tell you what was taught, say so.

WHAT TO LIST when recognized:
- `typical_skills`: the concrete, nameable tools, procedures and techniques taught —
  the things a graduate could actually do afterwards. Use the names practitioners in
  that field use, since these are compared against a practitioner's own wording.
- `typical_concepts`: the theory and principles covered, as distinct from the
  practical skills above.

NAME THE COMPONENTS, NOT THE UMBRELLA. `typical_skills` is matched against the skills a
candidate listed, and candidates list specific tools. An umbrella term matches nothing
and is wasted output.

So where a course teaches a platform, ecosystem or field, name the individual tools and
techniques within it that the course actually covers, rather than the collective noun.
Naming the platform alone ("the framework", "the ecosystem", "the field") corroborates
nothing, because no candidate lists a skill by that name.

Stay within what the course genuinely teaches. Do not list every tool that exists in an
ecosystem merely because it is related to one that is taught.

Be conservative in scope. List what the course reliably covers, not everything it
might touch. Let the course's stated level bound the claim: an introductory course
teaches foundations, not the advanced or specialist practice built on top of them.

For a course, describe the standard version of that course at that level. Note that
"Introduction to X" implies foundational depth, not mastery.

CLAIMED SKILLS (the only values allowed in `covers_claimed_skills`):
{claimed_skills}

CREDENTIALS (with the grade achieved, where the transcript recorded one):
{credentials}
""",
)
