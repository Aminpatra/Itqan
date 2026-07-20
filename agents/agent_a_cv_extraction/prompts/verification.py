"""Grounding adjudication — the tie-breaker for fields the Python matcher scored
in the ambiguous band.

The model is not asked "is this correct?". It is asked to produce a verbatim
supporting quote, and ``grounding.verify_quote`` then checks that quote against
the source itself. A model that fabricates the quote fails the check and the
field is dropped regardless of what it claimed. The LLM cannot self-certify.
"""

from __future__ import annotations

from langchain_core.prompts import PromptTemplate

GROUNDING_ADJUDICATION_PROMPT = PromptTemplate(
    input_variables=["source_text", "candidate_fields"],
    template="""You verify whether extracted values are supported by a source document.

These values were extracted from the SOURCE TEXT below, but an automated matcher could
not confirm them exactly. OCR errors mean a value may be correct despite not matching
character-for-character — that is what you are here to adjudicate.

FOR EACH FIELD, DECIDE:
- grounded = true  -> the source genuinely contains this information, allowing for OCR
                      noise (e.g. "Sultan Qab0os University" vs "Sultan Qaboos University").
- grounded = false -> the source does not support it, OR it is an expansion, inference,
                      or normalization of what the source says rather than a transcription.

YOU MUST QUOTE THE EVIDENCE. Set `evidence_quote` to the exact contiguous substring of
SOURCE TEXT that supports the value. Copy it character-for-character from the source,
including any OCR corruption. Do not clean it up, do not paraphrase, do not reconstruct
what you think it should say.

If you cannot quote a supporting substring, set grounded = false. Your quote is checked
against the source automatically; an unverifiable quote fails the field.

Judge only the fields listed. Do not add fields.

FIELDS TO VERIFY:
{candidate_fields}

<<<SOURCE_START>>>
{source_text}
<<<SOURCE_END>>>
""",
)
