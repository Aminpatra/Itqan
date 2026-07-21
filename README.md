# Itqan

A LangGraph agent that turns a candidate's **CV** (required) and **transcript** (optional) — PDF or image —
into a verified, provenance-tagged JSON profile for a downstream agent to consume.

Built with LangChain + LangGraph, PaddleOCR for scanned documents, and OpenAI for extraction.

---

## Quick start

```bash
pip install -r requirements.txt
cp .env.example .env          # then put your OPENAI_API_KEY in it

python main.py agent-a --cv path/to/cv.pdf
```

With a transcript, and skipping the interactive prompts:

```bash
python main.py agent-a --cv cv.pdf --transcript transcript.pdf
python main.py agent-a --cv cv.pdf --no-hitl
```

**Multi-page documents.** A CV or transcript photographed a page at a time is several files but one
document. Pass them all — they are read in the order given, so list them in page order (filenames are not
sorted for you, because they are not reliably page-ordered):

```bash
python main.py agent-a --cv cv1.png cv2.png --transcript t1.png t2.png
```

Pages may mix types freely (`--cv page1.pdf page2.png`). Each file is classified, read and confidence-scored
independently, then concatenated; every one of them appears separately in `source_documents` with its own
OCR confidence, so you can tell which page was the badly-lit one. An unreadable page is skipped with a
warning rather than failing the run — only losing *every* CV page is fatal.

Try it with no API key and no documents at all:

```bash
python main.py agent-a --cv tests/fixtures/sample_cv.txt --fake-llm --no-hitl
```

### Options

| Flag | Meaning |
|---|---|
| `--cv PATH...` | **Required.** CV as PDF or image. Accepts several files for a multi-page document, read in the order given |
| `--transcript PATH...` | Optional transcript; supplies courses, grades and CGPA. Also accepts several files |
| `--no-hitl` | Never prompt. Gaps are recorded in the output instead. For scripted runs |
| `--fake-llm` | Canned offline LLM — no API key, no network |
| `--model NAME` | OpenAI chat model (default `gpt-4o-mini`) |
| `--ocr-lang CODE` | PaddleOCR language, default `en`. Use `ar`, `fr`, `es`, `ch`, `japan`, … for CVs in another script — the English models will misread them rather than fail |
| `--output-dir DIR` | Where artifacts go (default `./output`) |
| `--run-id ID` | Reuse a specific run id. Advanced; see the note on `thread_id` below |

---

## What it does

```
START
  -> ingest            classify each file from magic bytes, not the extension
  -> extract_text      PDF text layer, or PaddleOCR for images and scanned PDFs
  -> llm_extract_cv
       |--[transcript]--> llm_extract_transcript --|
       |--[none]-----------------------------------+--> verify_grounding
  -> assess_gaps
       |--[gaps found]--> human_review     <== pauses here for your input
       |                    -> validate_human_input
       |                         -> assess_gaps    (bounded, max 2 rounds)
       |--[no gaps]-----> research_curriculum
  -> research_curriculum   what the courses/certificates typically teach
  -> judge_skills          claims cross-checked against that evidence
  -> summarize             few-shot prompting
  -> persist               writes candidate_profile.json
  -> END
```

### Anti-hallucination: four independent layers

An LLM asked to extract from a thin or OCR-mangled CV will invent an email domain, expand "BSc CS" into a
full degree name, or add skills nobody claimed. Prompt instructions alone do not prevent this, so the
prompt is only the first of four layers:

1. **Prompt constraints** — "extract only what is literally present; null is a correct answer", with
   anti-patterns spelled out by example. Source text is fenced and declared to be data, so a CV containing
   "ignore previous instructions" is treated as content, not as a command.
2. **Structured output** — `with_structured_output(CVExtraction)`; the model cannot return prose.
3. **Python grounding** (`grounding.py`, no LLM) — every extracted string is matched against the source
   text. Exact substring scores 1.0; otherwise a windowed `difflib` ratio. Above 0.92 accepted, 0.75–0.92
   escalated, below 0.75 dropped. This is the load-bearing check, and it is deterministic — a check
   implemented by the same class of system that produces the errors is not much of a check.
4. **LLM adjudication with a Python backstop** — borderline fields go to a second model call that must
   quote verbatim supporting text. That quote is then verified against the source *in Python*. A fabricated
   quote fails and the field is dropped regardless of what the model claimed. The LLM cannot self-certify.

Worked example from the test suite — the extractor was fed a CV reading `BSc Computer Science, Sultan
Qaboos University … Skills: Python, PyTorch, SQL`:

```
1.000 exact    OK  full_name = 'Sara Al-Balushi'
1.000 exact    OK  contact.email = 'sara.b@squ.edu.om'
0.200 dropped  NO  contact.phone = '+968 9123 4567'          <- invented
0.588 dropped  NO  education[0].degree = 'Bachelor of Science in Computer Science'  <- expanded
0.364 dropped  NO  skills[1].name = 'Kubernetes'             <- invented
```

### Curriculum research

Before judging, the agent expands each course and certificate into what it typically teaches, so that
credentials count as evidence instead of being ignored as bare titles. Without it, a candidate's transcript
contributed nothing to their skill ratings — the evidence was there, it just needed unpacking.

The step is given **the candidate's claimed skills** and asked, per credential, which of *those specific
skills* its normal syllabus covers. That direct question replaced an earlier design that emitted a generic
syllabus and hoped its wording overlapped the candidate's. It usually didn't: a syllabus saying "data
ingestion tools" and a CV saying "Apache Sqoop" describe the same thing and share no words, so real
corroboration was missed. Matching is on meaning — product names against generic terms, components against
the platform that teaches them, techniques against the subject assessed through them — which is why it works
the same for a nurse's clinical skills or a lawyer's coursework as for an engineer's.

Grades ride along where the transcript recorded one, passed through **verbatim and never interpreted in
code**. Grading scales differ too much between countries and institutions — letters, 4.0, 5.0, ten-point,
percentages, honours classes — for any threshold here to be right everywhere.

This is the one stage where the model contributes knowledge from outside the documents, so it is fenced in:

- **It cannot introduce a skill.** Curriculum only ever corroborates a skill the candidate already claimed.
  The corroboration list is intersected in code with what was actually claimed, so a syllabus covering
  something they never listed is dropped rather than added.
- **Unrecognised credentials are discarded.** The model must declare when it does not know a credential —
  local hackathons, one-day events, club competitions — and those are dropped rather than passed on with a
  guessed syllabus. Attending an event is not evidence of a curriculum.
- **Credentials it was never asked about are discarded**, so an invented course cannot corroborate anything.
- **It caps at `medium`.** Only a project, a job, or the certification being the skill itself reaches
  `high`. Having studied a tool is real evidence, and weaker than having built something with it.
- **It can only raise a rating, never lower one**, and every use is published in
  `provenance.curriculum_researched` so a consumer can see exactly which ratings rested on inference.

### Skill quality judging

Generic claims are filtered out with a reason, not silently deleted — a downstream agent may reasonably
disagree, and silent filtering is indistinguishable from a bug when the numbers look wrong later.

```
Skills kept (2):
  + Python                       high    project
  + PyTorch                      high    project
Skills filtered out (1):
  - Teamwork                     Generic self-assessed trait.
```

Every kept skill is tagged with where it was corroborated: `project`, `experience`, `course`,
`certification`, or `claim_only`. A specific, named skill with no corroboration is **kept** as
`claim_only`/`low`, not rejected — a thin CV is not a fabricated one.

Two invariants are enforced in code rather than asked for in the prompt, because a model under a long
prompt drifts and a rating that silently changes meaning is worse than no rating:

- **Quality is clamped to what the evidence supports** — `claim_only` cannot exceed `low`, `course` cannot
  exceed `medium`, and only `project`/`experience`/`certification` reach `high`. Downgrades only; the clamp
  never invents confidence the model withheld.
- **A skill the judge fails to return is not lost.** It survives as `claim_only`/`low` with a warning,
  rather than disappearing because the model returned a short list.

### Human-in-the-loop

When a required field is missing, a field fails verification, or OCR read something at low confidence, the
graph pauses via LangGraph's `interrupt()` and asks:

```
--------------------------------------------------------------------
  MISSING OR UNCERTAIN - round 1
  2 item(s) need your input. Press Enter to skip any of them.
--------------------------------------------------------------------

  [1/2] contact.phone
      why: Phone number could not be verified against the document
      e.g. +968 9123 4567
      >
```

Manual entries are **also** sent through the LLM, which checks format plausibility and rejects placeholders
("asdf", "n/a", a GPA of 17.5 on a 4.0 scale). Human input bypasses grounding by design — a phone number you
type will not appear in the OCR text — so this validation gate is what stops the HITL path from being a hole
through every other check.

---

## The output

Everything lands in `output/<run_id>/`:

| File | Contents |
|---|---|
| `candidate_profile.json` | **The deliverable.** The envelope defined in `shared/contracts.py` |
| `ocr_cv.json` | Raw OCR: every block with text, confidence and bbox, plus reconstructed text. One file per document, covering every page of every part |
| `ocr_transcript.json` | Same, for the transcript |
| `cv_part00_pages/`, … | PNGs rendered from scanned PDFs, one directory per input file |

The envelope:

```json
{
  "schema_version": "itqan.candidate_profile/1.0",
  "run_id": "20260720-105733-d16b6da6",
  "generated_at": "2026-07-20T10:57:41+00:00",
  "source_documents": [{"path": "...", "kind": "image", "role": "cv", "mean_confidence": 0.9848}],
  "candidate": {
    "full_name": "Sara Al-Balushi",
    "contact": {"email": "sara.b@squ.edu.om", "phone": "+968 9123 4567"},
    "education": [...], "experience": [...], "projects": [...],
    "courses": [{"code": "COMP3202", "title": "Machine Learning", "grade": "A-"}],
    "academic_record": {"institution": "Sultan Qaboos University", "cgpa": "3.71"}
  },
  "skills": {
    "accepted": [{"name": "Python", "quality": "high", "evidence_type": "project", ...}],
    "rejected": [{"name": "Teamwork", "rationale": "Generic self-assessed trait.", ...}]
  },
  "summary": { "headline": "...", "profile": "...", "gaps_or_unknowns": [...] },
  "provenance": {
    "grounding": {"full_name": {"grounded": true, "score": 1.0, "method": "exact"}},
    "human_supplied_fields": ["contact.phone"],
    "dropped_fields": ["contact.phone"],
    "review_rounds": 1
  },
  "confidence": {"overall": 0.94, "per_section": {"contact": 0.76, "education": 1.0}}
}
```

### How the next agent should read it

- **`provenance.grounding[field].method`** tells you where a fact came from: `exact` and `fuzzy` are
  document-derived, `llm` survived adjudication, `human` was typed in by the candidate. Do not treat a
  `human` field and an `exact` field as equally verified — they are verified against different things.
- **`provenance.dropped_fields`** is what the pipeline refused to publish. A field appearing in *both*
  `dropped_fields` and `human_supplied_fields` was hallucinated, caught, and then supplied by the user.
- **`skills.rejected`** is not garbage — it is the filter's audit trail, with a `rationale` per skill. Read
  it if a candidate seems to be missing skills you expected.
- **`skills.accepted[].corroborating_credential`** marks a rating that came from a credential's *typical*
  curriculum rather than from CV text. Cross-reference `provenance.curriculum_researched` to see the
  assumed syllabus, and discount the rating if you disagree with it. `evidence_type: "project"` never
  rests on inference.
- **`summary.gaps_or_unknowns`** is deliberately populated rather than smoothed over. It is the honest
  account of what the documents did not say.
- **`confidence.overall`** is scaled by the *fraction* of fields that could be grounded, so an extraction
  where most fields were dropped scores low even if the survivors scored well.

Read it with the published contract rather than by hand:

```python
from shared.contracts import load_profile

profile = load_profile("output/<run_id>/candidate_profile.json")
for skill in profile.skills["accepted"]:
    print(skill["name"], skill["quality"], skill["evidence_type"])
```

---

## Layout

```
Itqan/
  main.py                    dispatcher: `python main.py <agent> ...`
  shared/                    cross-agent: config, LLM factory, contract, artifacts
    contracts.py             <- the inter-agent interface
  agents/
    agent_a_cv_extraction/
      graph.py  state.py  schemas.py  grounding.py  cli.py
      prompts/   extraction, verification, curriculum, skills, human_validation, summary
      ingestion/ detect, pdf_text, ocr
      nodes/     one module per graph node
  tests/
```

### Adding Agent B

1. Create `agents/agent_b_job_match/` with a `cli.py` exposing `main(argv)`.
2. Import `shared.contracts` and `shared.llm`. **Never import `agents.agent_a_cv_extraction.*`** — the
   envelope JSON is the only interface between agents, which is what lets Agent A be rewritten (or have its
   OCR stack swapped) without touching anything downstream.
3. Register it in `AGENTS` in `main.py`.

```python
from shared.contracts import load_profile
from shared.llm import build_llm

profile = load_profile(path_from_agent_a)
```

---

## Testing

```bash
python -m pytest tests/ -q            # 58 tests, no network, no API key, no Paddle
```

The suite runs the entire graph — routing, reducers, the interrupt/resume cycle, envelope validation —
against a canned LLM, then attacks the grounding matcher with deliberately hallucinated extractions.

To check OCR in isolation before trusting it inside the graph:

```bash
python -m agents.agent_a_cv_extraction.ingestion.ocr some_image.png
```

---

## Notes and gotchas

**PaddleOCR 3.x, not 2.x.** `predict()` (not the deprecated `ocr()`), and `use_textline_orientation` (the
old `use_angle_cls` no longer exists). Any tutorial or generated code you compare against will use the 2.x
API and be wrong. `paddleocr==3.4.0` is pinned deliberately.

**Never resume with an empty dict.** On langgraph 1.2.9, `Command(resume={})` does *not* resume — an empty
mapping reads as a resume-map with no entries, so the node re-runs and interrupts again, forever, with no
error. (`Command(resume=None)` is worse: `UnboundLocalError` inside the Pregel loop.) A user pressing Enter
through every prompt would hang the CLI, so `nodes/human_review.py` defines `SKIP_SENTINEL` for that case.
`test_empty_resume_would_hang_so_sentinel_is_required` pins the behaviour.

**The interrupt node replays.** On resume LangGraph re-executes `human_review` from the top; `interrupt()`
does not resume mid-function. Anything above that call runs twice, which is why the node does nothing but
build a payload, and why HITL is split into two nodes with validation in the second.

**`thread_id` must be fresh per run.** The CLI generates one automatically. Passing `--run-id` with a value
you have used before resumes that run's checkpoint instead of starting clean.

**Model downloads.** `PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK` is set in `shared/config.py` before paddle is
imported; without it every import makes a network round-trip that makes the CLI feel hung. Models are read
from `~/.paddlex/official_models`.

**`.env` is inside a git repo.** `git rev-parse --show-toplevel` here resolves to `C:\Users\Aminpatra` — your
whole home directory is an uncommitted repo. `.gitignore` covers `.env` and `output/`; check it is in effect
before committing anything from this tree.
