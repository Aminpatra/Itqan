# Itqan

Two independent LangGraph agents that share nothing but `shared/`:

- **Agent A** turns a candidate's **CV** (required) and **transcript** (optional) — PDF or image —
  into a verified, provenance-tagged JSON profile.
- **Agent B** ingests **public job postings** every 12 hours into two Postgres tables — a searchable
  posting store and an aggregated skill-demand table — filtering scams and never fabricating a count.
- **Agent C** reads both and computes the candidate's **skill gap** against live postings, falling
  back to sector demand statistics when retrieval is thin. Fully deterministic — zero LLM calls.
- **Agent D** is Agent B for **courses**: it ingests online courses (Coursera + freeCodeCamp),
  extracts the skills each one *teaches*, and aggregates a `skill_supply_stats` table. Joined with
  Agent B's demand on `esco_code`, it answers *which in-demand skills have few courses*.

Built with LangChain + LangGraph, PaddleOCR for scanned documents, OpenAI for extraction and
embeddings, and Postgres + pgvector for the job-market store, with the EU's ESCO taxonomy as the
shared skill vocabulary.

**How they connect** — the agents never import each other; the joins are a file contract and a
database read surface, which is what lets each be run, tested, or rewritten alone:

```
                        one command:  python main.py pipeline --cv cv.pdf
                        ┌──────────────────────────────────────────────┐
CV/transcript ──▶ Agent A ── candidate_profile.json ──▶ Agent C ◀── job_postings +
                     (or agent-c --watch picks it up)      │        skill_demand_stats
                                                           ▼            ▲
                                                     skill_gap.json     │ live Postgres reads,
                                                                        │ every C run
web sources ──▶ Agent B (12h cycle) ────────────────────────────────────┘
```

Agent A is documented first; **[Agent B](#agent-b--scheduled-job-ingestion)** and
**[Agent C](#agent-c--skill-gap-analysis)** have their own sections below.

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

## Agent B — scheduled job ingestion

A second, fully independent agent. Every 12 hours it ingests **public, logged-out** job postings,
filters out scams, extracts structured skills, deduplicates, and maintains an aggregated
skill-demand table. Its only outputs are two Postgres tables that Agent C (not built) will read —
it never calls Agent A or any user-facing code.

### Setup

Agent B needs Postgres with the **pgvector** extension. A local container is enough:

```bash
docker run -d --name itqan-pg -p 5432:5432 \
  -e POSTGRES_PASSWORD=itqan_dev -e POSTGRES_DB=itqan \
  -v itqan_pgdata:/var/lib/postgresql/data pgvector/pgvector:pg16

# in .env:
ITQAN_DATABASE_URL=postgresql://postgres:itqan_dev@localhost:5432/itqan
ITQAN_USER_AGENT=ItqanJobBot/0.1 (+you@example.com)   # a real contact; sent on every request
```

The user agent is **required before any live run** — a scraper that does not identify itself gives
a site operator no way to reach you, so the code refuses to start with the placeholder.

```bash
python main.py agent-b --migrate          # create/upgrade the schema (idempotent)
python main.py agent-b --check            # connectivity, schema, row counts
python main.py agent-b --dry-run --sources el7far --limit 5   # fetch + score, write nothing
python main.py agent-b --once             # one real cycle
python main.py agent-b --once --fake-llm  # a full cycle with no API key and no spend
```

### The cycle

```
plan_sources  choose which sources run (config-validated)
  -> Send(scrape)   ONE concurrent branch per source — fetch and parse only, no DB
  -> ingest         change-detect -> legitimacy gate -> extract -> link-dedup -> embed
                    -> near-dup -> upsert, per source batch, each in its own transaction
  -> staleness      age by CYCLES not clock; stale at 3 missed, hard-delete at 60 days
  -> aggregate      recompute skill_demand_stats for the current window
  -> runlog         record per-source health, write output/<run_id>/ingest_cycle.json
```

The ordering exists to make a warm cycle cheap: an **unchanged** posting costs one `UPDATE` and no
LLM or embedding; a **rejected** one costs no extraction and no embedding; a Telegram repost that
**links** to its blog original is resolved by URL and never embedded. Run the same cycle twice and
the second does zero LLM and zero embedding work.

### Not fabricating the numbers

The demand table is a claim about the labour market someone may plan a career around, so the whole
agent is built so a number is never invented:

- **Legitimacy is deterministic first.** Rules (bilingual Arabic/English) score each posting with
  **noisy-OR**, not a sum — five weak signals reach 0.67, not certainty. Only the genuinely
  uncertain band (0.40–0.70 risk) costs an LLM call, and that call must quote verbatim text which is
  re-checked in Python; a fabricated quote is discarded and the rule score stands. Rejected postings
  are kept for audit, never deleted, and excluded from every statistic.
- **Country is not a scam signal.** A regional feed legitimately carries postings for neighbouring
  countries; scoring "not Oman" as fraud would silently delete real jobs. Scope is a *separate* axis,
  filtered only at aggregation.
- **Only real vacancies aggregate.** A classifieds site is full of job *seekers* advertising
  themselves — counting those measures supply and reports it as demand. A posting contributes to
  `skill_demand_stats` only if it is `active`, not a duplicate, in scope, `listing_intent='vacancy'`
  **and** `poster_type='company'`. Everything else is stored and retrievable, just not counted.
- **Trends are conservative.** Below 5 postings a skill's trend is pinned to `stable`, because 1→2 is
  +100% noise, not a rising trend, and a sector with fewer than 10 postings flags `low_confidence` on
  every row.
- **A blocked source never ages its inventory.** A partial fetch (a 429 mid-cycle) still ingests what
  it got, but its un-fetched postings are left alone rather than aged toward deletion.

### The two tables (the Agent C contract)

`job_postings` (retrievable postings, with a 1536-d embedding) and `skill_demand_stats` (aggregated
fallback). Everything else — `source_health`, `schema_migrations` — is internal bookkeeping, not part
of the contract. **Agent C must filter `skill_demand_stats` to the latest window**
(`WHERE window_end = (SELECT max(window_end) …)`); a sector absent from that window has no current
demand data, which is a normal state, not an error.

### Sources

`oman.el7far.com` (a Blogger Atom feed) and `@omanjob1` (a Telegram channel, read via its public
`t.me/s/` preview) are live. A Telegram channel stays disabled until a human sets `terms_reviewed=True`
for it in `sources/config.py` — confirming a channel renders is not the same as having reviewed its
terms. Dubizzle (`html_scrape`) is built but shipped disabled: it exposes no poster identity, so
nothing it produces can be classified `company` and therefore nothing it produces can aggregate yet.

### The ESCO layer — one vocabulary for skills

Free-text skills fragment ("prioritization", "priority organization", "task organization" — one
concept, three counts). The ESCO layer maps each raw skill key to a concept in the EU's
[ESCO classification](https://esco.ec.europa.eu) (~13,900 skills), and aggregation fills the
`esco_code` column from that map. Raw rows stay as the audit trail; a consumer groups by
`esco_code` to get canonical counts. Mapping is deterministic and auditable — exact label match,
then alt-label match, then nearest-label embedding above `esco_map_threshold`, else honestly
`unmapped` with the near-miss score stored as tuning evidence. No LLM is involved.

One-time setup: download the ESCO skills CSV bundle (English) from the
[ESCO download page](https://esco.ec.europa.eu/en/use-esco/download) into `ESCO/`, then:

```bash
python main.py agent-b --esco-sync            # load + embed the taxonomy (one-off, a few cents)
```

Each cycle then maps only genuinely new skill phrases. Bump `--esco-version` when loading a newer
release; previously-unmapped skills are retried against it.

*This project uses the ESCO classification of the European Commission.*

### Scheduling (Windows Task Scheduler)

`--once` is crash- and reboot-safe and reports failure through its exit code (non-zero on a partial
fetch), so a scheduler running `--once` beats the in-process `--loop` (demos only). An advisory lock
means a second cycle that overlaps the first exits cleanly rather than double-counting.

```powershell
# every 12 hours; adjust the path and python
schtasks /create /tn "ItqanAgentB" /sc hourly /mo 12 ^
  /tr "cmd /c cd /d C:\path\to\Itqan && python main.py agent-b --once >> output\cron.log 2>&1"
```

Remember `docker start itqan-pg` after a reboot — the container does not auto-start.

### Operator commands

| Flag | Meaning |
|---|---|
| `--dry-run` | Fetch, parse and score; write nothing. For vetting a source before trusting it |
| `--fake-llm` | Run a real cycle with offline doubles — no API key, no spend |
| `--no-embed` | Skip embedding (disables similarity-based near-dup; link dedup still runs) |
| `--sources A,B` | Run only these sources |
| `--limit N` | Cap postings per source |
| `--label-sample N` | Export N ingested postings (gitignored) for a human to label, to measure the reject filter's precision — the one number the rules cannot self-report |
| `--purge-source S` | Delete a decommissioned source's postings. Never automatic |

---

## Agent C — skill gap analysis

```bash
python main.py agent-c --profile output/<run_id>/candidate_profile.json
python main.py agent-c --profile ... --sector 2 --top-k 15 --user-id someone
python main.py agent-c --watch          # react to Agent A automatically (see below)
```

**Watch mode** is the hands-free pipeline: leave `agent-c --watch` running, and every time Agent A
finishes a profile, the gap analysis follows automatically — `skill_gap.json` is written into the
same `output/<run_id>/` folder, so everything about one candidate lives in one place and the file's
presence marks the run processed (nothing is ever analysed twice). The agents stay fully decoupled:
the handshake is the filesystem, not an import. Profiles that already exist when the watcher starts
are ignored unless `--backfill`; a profile that fails is set aside rather than retried every poll.

Reads Agent A's profile and Agent B's tables (through the published read surface,
`shared/job_market.py` — Agent C never touches Agent B's internals), and writes
`output/<run_id>/skill_gap.json`.

**Deterministic by design — zero LLM calls.** Matching and scoring are arithmetic over data two
other systems already verified: cosine similarity in the same embedding space Agent B uses, and
ESCO concept identity from the same tables. An LLM here would add nothing but non-determinism.

```
build_query_embedding   profile -> posting-shaped essence (headline / location / skills) -> vector
retrieve_postings       nearest eligible postings; < 5 above the similarity threshold => fallback
map_candidate_skills    the candidate's skills through the ESCO tables — READ-ONLY, same tiers
                        and threshold as Agent B, so both sides speak one vocabulary
gap_analysis            per job: matched / missing / possible_match per required skill
persist                 skill_gap.json
```

The honesty rules the arithmetic follows:

- **`possible_match` ([0.6, 0.8) similarity) is never auto-resolved.** It counts in the
  `gap_score` denominator (it is a real requirement) but never in the numerator (we do not know
  it is missing). `gap_score = Σweight(missing) / Σweight(matched+missing+possible)`.
- **Weights are demand counts** — `frequency_count` from the latest stats window (by `esco_code`,
  else by raw key), floor 1 so an un-aggregated skill still exists in the score. This system has no
  essential/optional tags and does not invent them.
- **The fallback sector is never guessed**: modal sector of the retrieved postings, or `--sector`,
  or the sector-level analysis is skipped with a warning in the output.
- **ESCO identity beats phrase fuzz**: a job skill sharing the candidate's concept is matched even
  when the raw phrasings are embedding-distant.
- The output flags `used_fallback` so a consumer knows whether it is reading per-job specificity or
  sector-level aggregates, embeds the gap-score formula, and carries the candidate's per-skill ESCO
  mappings (including near-miss scores for unmapped skills) as tuning evidence.

The retrieval threshold (`agent_c_match_threshold`) started at the spec's 0.80 and was moved to
**0.43 on the first live measurement**: a real CS-graduate profile scored 0.41–0.48 against
postings a human judges relevant (technology roles, data specialist, Oracle DBA) — cross-type
similarity compresses, because a candidate essence aggregates many skills while a posting names
few. That was one profile; the CLI prints the distribution every run, and the threshold moves on
that evidence. The stats fallback likewise ignores skills aggregated only once
(`agent_c_fallback_min_freq`) — without the floor it compared against 463 sector phrases, ~87%
freq-1 noise, and the gap saturated at a meaningless ~1.0.

---

## Agent D — course ingestion (the supply side)

Agent B ingests jobs → skill **demand**; Agent D ingests courses → skill **supply**. Same
architecture (the scraping layer is shared in `shared/scraping/`), same ESCO vocabulary, same store
patterns — on a **3-day cycle** (courses change far slower than jobs).

```bash
python main.py agent-d --migrate
python main.py agent-d --once --dry-run --sources coursera --limit 5   # live, no writes
python main.py agent-d --once --limit 20                               # a real cycle
python main.py agent-d --check
```

Runtime dependency: the shared ESCO taxonomy, synced once by `agent-b --esco-sync`.

**Sources.** Two consent models, deliberately different:
- **Coursera** (`source_type='api'`) — the public Catalog API (23K courses). robots.txt disallows
  `/api/` for **crawlers**; a documented public API consumed within its rate limits is governed by
  its **terms**, gated by a human `terms_reviewed=True` (same discipline as the Telegram gate). API
  sources do not robots-check; that override is recorded, never silent.
- **freeCodeCamp** (`source_type='html_scrape'`) — a web scrape, so it **does** honor robots
  (fully open). The 11 free certifications; curriculum is **CC-BY-SA-4.0**, so every row carries
  `license` and `attribution`. We catalog skill facts + links, never redistribute content.

**The pipeline** mirrors Agent B stage-for-stage, with two differences that fall out of "courses
aren't scam-prone postings": there is **no legitimacy filter** (a **quality gate** rejects a course
that yields no extractable skill instead), and **no link dedup** (courses have no cross-source
links; near-dup by essence embedding still runs). Skills taught are extracted with the same
recruiter-canonical-name discipline as job requirements, so the two sides aggregate on one
vocabulary, and mapped into Agent D's own `course_esco_map` (never Agent B's `skill_esco_map`).

**The payoff** — demand meets supply on `esco_code`:

```sql
SELECT es.preferred_label AS skill, d.frequency_count AS demand_jobs,
       COALESCE(s.course_count, 0) AS supply_courses
FROM skill_demand_stats d
LEFT JOIN skill_supply_stats s USING (esco_code)
JOIN esco_skills es ON es.esco_uri = d.esco_code
WHERE d.window_end = (SELECT max(window_end) FROM skill_demand_stats)
ORDER BY d.frequency_count DESC;
```

A high `demand_jobs` with a low `supply_courses` is a real market gap — an in-demand skill with few
courses teaching it.

---

## Layout

```
Itqan/
  main.py                    dispatcher: `python main.py <agent> ...`
  shared/                    cross-agent: config, LLM + embedding factories, contract, grounding
    contracts.py             <- the inter-agent interface
  agents/
    agent_a_cv_extraction/
      graph.py  state.py  schemas.py  cli.py
      prompts/   extraction, verification, curriculum, skills, human_validation, summary
      ingestion/ detect, pdf_text, ocr
      nodes/     one module per graph node
    agent_b_job_ingest/
      graph.py  state.py  nodes.py  runner.py  cli.py
      pipeline.py            the ingest tail: dedup -> legitimacy -> extract -> embed -> neardup -> upsert
      legitimacy.py  aggregate.py  hashing.py  schemas.py  records.py
      prompts/   extraction, legitimacy
      sources/   base, http, robots, config, factory, el7far, telegram, dubizzle
      db/        store.py (all SQL), migrate.py, migrations/*.sql
  tests/
```

### Adding an agent

1. Create `agents/<agent_name>/` with a `cli.py` exposing `main(argv) -> int`.
2. Import from `shared/` only. **Never import `agents.agent_a_cv_extraction.*`** — that boundary is what
   lets Agent A be rewritten (or have its OCR stack swapped) without touching anything downstream. If you
   need something that currently lives inside an agent, move it into `shared/` first;
   `shared/grounding.py` is the precedent.
3. Register it in `AGENTS` in `main.py` — one dict entry, no base class or plugin protocol.

The agents in play:

| Agent | Reads | Writes |
|---|---|---|
| **A** — `agent_a_cv_extraction` | CV + transcript files | `candidate_profile.json` |
| **B** — `agent_b_job_ingest` | Public job sources (feed, Telegram, HTML) | `job_postings`, `skill_demand_stats` (Postgres) |
| **C** — `agent_c_gap_analysis` | Agent A's profile **and** Agent B's tables (via `shared/job_market.py`) | `skill_gap.json` |
| **D** — `agent_d_course_ingest` | Coursera API + freeCodeCamp | `courses`, `skill_supply_stats` (Postgres) |

Agent B does **not** read `CandidateProfile`; that is Agent C's job. Agent B and Agent A share nothing but
`shared/`, and neither knows the other exists.

```python
from shared.contracts import load_profile   # Agent C
from shared.embeddings import build_embedder
from shared.llm import build_llm
```

---

## Testing

```bash
python -m pytest tests/ -q            # ~290 tests, no network, no API key, no Paddle, no Postgres
```

The suite runs Agent A's entire graph — routing, reducers, the interrupt/resume cycle, envelope
validation — against a canned LLM, attacks the grounding matcher with deliberately hallucinated
extractions, and exercises Agent B's adapters (against saved fixtures), legitimacy rules, and the
whole ingest pipeline (against an in-memory store). Nothing hits a live site or a database.

Agent B's SQL — the FK-ordered upsert, the pgvector cosine search, and above all the aggregation
query whose counts are the fabrication-prone core — can only be verified against real Postgres, so
those tests are opt-in and skip cleanly when no database is configured:

```bash
docker exec itqan-pg psql -U postgres -c "CREATE DATABASE itqan_test;"
ITQAN_TEST_DATABASE_URL=postgresql://postgres:itqan_dev@localhost:5432/itqan_test \
  python -m pytest tests/ -q          # ~365 tests including the database suite
```

The database tests use a **separate** database from `ITQAN_DATABASE_URL` — they truncate between
cases, and pointing them at the working database would destroy real ingested postings.

To check OCR in isolation before trusting it inside Agent A's graph:

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
