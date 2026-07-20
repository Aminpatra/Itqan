"""Validation of manually-entered values.

Human input bypasses the grounding pass by definition — a phone number the
candidate types in will not appear in the OCR text, and that is the point. So it
needs its own gate, or the HITL path becomes a hole straight through every other
check.

This gate is about *plausibility and format*, not truth. We cannot verify that a
typed phone number is really theirs. We can catch "asdf", "n/a", a GPA of 17.5 on
a 4.0 scale, and an email with no domain.
"""

from __future__ import annotations

from langchain_core.prompts import PromptTemplate

HUMAN_INPUT_VALIDATION_PROMPT = PromptTemplate(
    input_variables=["field_specs", "raw_answers"],
    template="""You validate values a user typed to fill gaps in their extracted CV data.

For each answer decide `accepted` and produce a `normalized_value`.

REJECT (accepted = false) when the answer is:
- a placeholder or refusal: "n/a", "none", "idk", "-", "asdf", "test", "xxx", empty
- structurally wrong for the field: an email with no "@" or no domain; a phone number
  with too few digits to be real; a GPA outside a plausible scale (e.g. 17.5 out of 4.0);
  a date in the future; an end date before its start date
- obviously not the kind of thing asked for: a person's name given for "institution",
  a sentence of prose given for "email"

ACCEPT and normalize otherwise:
- trim whitespace, fix obvious capitalisation of proper nouns ("sultan qaboos university"
  -> "Sultan Qaboos University")
- keep phone numbers in the format the user typed, only stripping stray spaces
- do NOT invent missing components. If they typed "3.7" for a GPA, return "3.7" — do not
  append "/4.0" unless they wrote it.

When you reject, set `issue` to a short, non-judgemental explanation the user will see
and can act on. "Missing @ and domain" is useful; "invalid" is not.

Return one entry per field, using the exact `field_path` given.

FIELDS REQUESTED:
{field_specs}

USER ANSWERS:
{raw_answers}
""",
)
