"""Curriculum expansion — what does a course or certificate actually teach?

Without this, the skill judge sees "Introduction to Database" and "Machine
Learning Specialization" as bare titles, and scores a candidate's SQL and
scikit-learn as unevidenced claims. The evidence was there; it just needed
unpacking.

This is the one place in the pipeline where the model contributes knowledge from
outside the documents, which makes it the one place most able to do damage. Three
guardrails follow from that:

1. Output is *about the credential*, never about the candidate. "This course
   usually covers SQL" is a statement about a syllabus. It is not a claim that
   the candidate knows SQL, and the judge is told to treat it accordingly.
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
    input_variables=["credentials"],
    template="""You describe what named courses and certifications typically teach.

For each credential below, list the skills and concepts a holder would normally have
been taught. This is used to corroborate skills a candidate has *already claimed* — it
is background about the syllabus, not a claim about any person.

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
- `typical_skills`: concrete, nameable tools and techniques taught — SQL, subnetting,
  scikit-learn, normalization, packet tracing. These are matched against the
  candidate's claimed skills, so use the names practitioners use.
- `typical_concepts`: the theory covered — relational algebra, the OSI model,
  supervised learning, gradient descent.

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
might touch. An introductory database course teaches SQL and normalization; it does
not teach query optimisation at scale or database administration.

For a course, describe the standard version of that course at that level. Note that
"Introduction to X" implies foundational depth, not mastery.

CREDENTIALS:
{credentials}
""",
)
