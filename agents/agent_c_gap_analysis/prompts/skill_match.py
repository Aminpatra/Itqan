"""Does the candidate's skill set satisfy this job requirement?

The one model call in Agent C, and it exists because string mathematics provably
cannot answer this question. Cosine similarity is symmetric and topical; skill
matching needs an ASYMMETRIC, hierarchical relation — "TensorFlow is an instance
of machine learning" is true, while "machine learning is an instance of
TensorFlow" is not, and one number cannot encode both.

The cost of not having it was measured: a candidate holding TensorFlow, PyTorch,
scikit-learn and completed deep-learning coursework was told a Senior Machine
Learning Engineer role was a 100% gap, because "machine learning" scored 0.5982
against their nearest skill — below the 0.60 floor.

Few-shot rather than description, because every example below is a REAL pair from
that run, and the boundary between them is precisely what a rule cannot state:
`JavaScript`/`Java` (0.6118) and `data analytics`/`data analytics engineering`
(0.8259) are both "similar", and the right answers are opposite. The examples
teach the distinction that matters — subsumption, not resemblance — and pin the
two failure directions the fence exists to prevent: inventing a candidate skill,
and resolving genuine uncertainty in the candidate's favour.
"""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

SYSTEM = """You decide whether a job requirement is already satisfied by a \
candidate's existing skills. Your answers feed a career-gap report, so a wrong \
"satisfied" tells someone they are ready for a job they are not, and a wrong \
"not_satisfied" sends them to relearn something they already know.

THE QUESTION IS SUBSUMPTION, NOT RESEMBLANCE. Ask: "does holding this candidate \
skill mean the person can meet this requirement?" Two skills can be closely \
related and still not satisfy each other.

  satisfied      - a candidate skill IS the requirement, or is a specific
                   instance/implementation of it, or clearly contains it.
  not_satisfied  - no candidate skill covers it. Related-but-different tools,
                   sibling technologies and similar-sounding names go here.
  uncertain      - you cannot tell from the names alone.

ABSOLUTE RULES
- Judge ONLY from the two lists you are given. Never invent, assume or infer a \
skill the candidate did not list — if it is not in CANDIDATE SKILLS, it does not \
exist.
- `satisfied_by` MUST be copied EXACTLY from CANDIDATE SKILLS. A name that is not \
on that list voids your verdict entirely.
- A specific product is NOT implied by the general skill: knowing SQL does not \
mean knowing MySQL, SQL Server or Oracle. The general IS implied by the specific.
- Skills marked [weak evidence] were never demonstrated — the candidate merely \
listed them. They can support "uncertain" but NEVER "satisfied".
- When you are unsure, answer "uncertain". It is always the safe answer: the \
system keeps its own deterministic verdict, so guessing gains nothing and an \
over-confident "satisfied" is the one error that cannot be caught downstream.
- Answer every requirement given, exactly once."""

# Real pairs from the live corpus. Each example is chosen because a rule stated
# in prose would get it wrong, or because it is a boundary the model must not
# cross.
EXAMPLES = """CANDIDATE SKILLS: Python, TensorFlow, PyTorch, scikit-learn, NumPy, \
Pandas, SQL, JavaScript, C++, Full-Stack Web Development, HDFS [weak evidence]

REQUIREMENTS: machine learning; Java; MySQL; software development; C#; \
deep learning frameworks; Hadoop

{{"verdicts": [
  {{"requirement": "machine learning", "decision": "satisfied",
   "satisfied_by": "TensorFlow",
   "reason": "TensorFlow, PyTorch and scikit-learn are machine-learning frameworks; \
using them is doing machine learning"}},
  {{"requirement": "Java", "decision": "not_satisfied", "satisfied_by": null,
   "reason": "the candidate has JavaScript, a different language despite the name"}},
  {{"requirement": "MySQL", "decision": "not_satisfied", "satisfied_by": null,
   "reason": "SQL is the general language; it does not imply this specific product"}},
  {{"requirement": "software development", "decision": "satisfied",
   "satisfied_by": "Full-Stack Web Development",
   "reason": "full-stack web development is software development"}},
  {{"requirement": "C#", "decision": "not_satisfied", "satisfied_by": null,
   "reason": "C++ is a separate language, not an instance of C#"}},
  {{"requirement": "deep learning frameworks", "decision": "satisfied",
   "satisfied_by": "PyTorch",
   "reason": "PyTorch and TensorFlow are deep-learning frameworks"}},
  {{"requirement": "Hadoop", "decision": "uncertain", "satisfied_by": null,
   "reason": "HDFS is a Hadoop component but is marked weak evidence, so it cannot \
confirm the broader skill"}}
]}}"""

HUMAN = """CANDIDATE SKILLS (the only skills this person has; \
[weak evidence] means listed but never demonstrated):
{candidate_skills}

REQUIREMENTS to judge:
{requirements}

Answer every requirement exactly once. Copy each requirement verbatim, and copy \
`satisfied_by` verbatim from CANDIDATE SKILLS."""

SKILL_MATCH_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM + "\n\nWORKED EXAMPLE\n" + EXAMPLES),
    ("human", HUMAN),
])
