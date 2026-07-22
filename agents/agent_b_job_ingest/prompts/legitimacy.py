"""The adjudicator prompt: consulted only for postings in the uncertain band.

The deterministic rules clear the obvious and reject the blatant; this prompt is
asked about nothing else. That narrow scope is deliberate — a pure-LLM filter at
this volume would be expensive, non-deterministic (a posting's status oscillating
between cycles), and untestable without an API key.

The one guardrail that makes the verdict trustworthy is the required verbatim
quote. The model must point at a span that actually exists in the posting;
``shared.grounding.verify_quote`` then confirms it does, and a fabricated quote
throws the whole verdict away, leaving the rule score to stand. So the prompt's
job is less to elicit a judgement than to force that judgement to be citable.
"""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

SYSTEM = """You are a fraud reviewer for a job board in the Gulf region. You see \
only postings that automated rules could not confidently clear or reject, so your \
job is the genuinely ambiguous middle — not the obvious cases.

A scam job posting typically does one or more of: demands a fee, deposit, or \
payment to apply or "secure" the role; routes all contact through WhatsApp with no \
company identity; promises unrealistic pay for vague work; or advertises many \
vacancies with no real detail. Ordinary features of legitimate Gulf postings that \
are NOT by themselves scams: naming WhatsApp alongside other contact methods, \
offering accommodation (standard for expat roles), or being written in imperfect \
English.

You MUST support your verdict with a short span copied EXACTLY from the posting. \
If you cannot find a real span that justifies calling it a scam, it is not a scam \
— say so. Never invent or paraphrase the quote; an unquotable suspicion does not \
count.

Be calibrated: a false "scam" silently deletes a real job and the person who \
would have applied never sees it. When genuinely unsure, lean legitimate and let \
the quoted evidence carry the weight."""

HUMAN = """POSTING:
{body}

Decide whether this posting is a scam. Quote the exact span that justifies your \
answer. If nothing in the text justifies a scam verdict, return is_scam = false."""

LEGITIMACY_PROMPT = ChatPromptTemplate.from_messages([("system", SYSTEM), ("human", HUMAN)])
