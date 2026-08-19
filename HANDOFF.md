# Handoff

For a person picking this up on another machine, or an assistant starting a fresh session.

**This file holds what the code cannot say about itself:** what is live, the rules that are not
negotiable, the hazards that cost real time, and what was mid-flight. For how the agents work read
[README.md](README.md); for the box and the deploys read [DEPLOY.md](DEPLOY.md). Nothing here
repeats either — where something is mechanical, this points rather than paraphrases.

---

## The shape of it

Two repositories, both inside the same OneDrive folder, so both already travel between machines:

| | |
|---|---|
| `Itqan/` | six agents, the FastAPI service, the migrations. Deploys from `main` |
| `itqan_web/` | Astro marketing site + React app. **Deploys from `main` only** — work happens on `amin-dev` |

Live at **tryitqan.com** — one OVH box, three containers (api, db, caddy), ~$6/month.

*(README's status section says "five agents". There are six: Agent S shipped after it was written.)*

## Working on another machine

**Docker is not required to work on this project.** It runs exactly one thing here — a Postgres
container with pgvector. It is not a build system and not a runtime: `python main.py`, every agent,
the API and both front ends run natively.

Measured with the database switched off:

```bash
unset ITQAN_TEST_DATABASE_URL && python -m pytest tests/ -q   # 845 passed, 417 skipped
python -m pytest tests/ -q                                    # 1,262 with a database
```

So a laptop with only Python and Node installed can write code, run the agents' logic, build both
front ends, and pass two thirds of the suite. What the other 417 need — `tests/api/` and the three
`tests/*/db/` suites — is **a Postgres with the pgvector extension**. Two ways to get one:

* the container README documents under "Agent B needs Postgres with the **pgvector** extension",
  plus its `CREATE DATABASE itqan_test` line;
* **any hosted Postgres that offers pgvector.** The migrations build the schema on first connect and
  the suites seed their own fixtures, so nothing has to be copied for tests to run. The choice is
  about the extension, not about Docker.

> **Never point `ITQAN_TEST_DATABASE_URL` at the VPS.** `tests/api/conftest.py` `TRUNCATE`s every
> `app_*` table between tests. Aimed at production it would delete every account, every uploaded CV
> and every stored profile — the same deletion that was once done deliberately, done here by
> accident, in the time it takes one test to run. A throwaway database, always.

**A fresh database is empty, and that surprises people.** Tests do not care. Running the agents or
the API against real data does: the corpus is roughly **1.7 GB**, of which `esco_labels` is 1,706 MB
(DEPLOY.md §4). That is a `pg_dump` to move between machines, not something a free tier will hold —
and without it Agent C retrieves nothing, so the dashboard is empty rather than broken.

## State, measured rather than remembered

Numbers rot. These two commands print the current ones:

```bash
python main.py status                     # corpus sizes, window freshness, ESCO coverage, source health
curl -s https://tryitqan.com/api/health   # API, and the mail hand-off counters
```

**Deliberately off, and why:**

- **edX** — `enabled=False` in `agents/agent_d_course_ingest/sources/config.py`, and
  `terms_reviewed=False`. Waiting on API credentials.
- **`browser_sources`** — empty in `shared/config.py`. Chromium was measured against the sources
  that motivated it and did not add a single rating; the measurement is recorded beside the setting
  so nobody re-litigates it from instinct.
- **Rerun from chat** — on, but only behind an explicit confirm.

## The rules

Each of these was learned by paying for it. Each is enforced somewhere, because an intention in a
document is not a control.

**Instructions are not controls; verification is.** Measured three times: `work_arrangement`
fabricated on 19 of 19 postings against a prompt forbidding it in as many words; Agent E calling an
unpriced course "free" once in 25 draws with the warning present; Agent A paraphrasing quotes it was
told to copy verbatim. Every agent that writes prose has a code-side fence — `shared/grounding.py`,
`verify_claims`, `verify_answer` — and the fence, not the prompt, is what holds.

**GulfTalent is not touched.** It is the majority of the job corpus and is crawled under one specific
exception in its terms. That exception's three conditions are behaviour, pinned by
`tests/agent_b/test_attribution_compliance.py`: minimal snippets only, the source named as
"GulfTalent", and the apply link pointing back to GulfTalent rather than the employer.

**A challenge or a 403 is a refusal.** We leave and record it. The techniques that would get past one
are deliberately unbuilt and AST-pinned in `tests/agent_b/test_browser_client.py`. Three aggregators
are written up as refusals in `agents/agent_b_job_ingest/sources/config.py` so nobody researches them
a fourth time.

**`terms_reviewed` is set by a human who has read the terms.** Never in code, never defaulted true,
never a command-line flag.

**robots.txt is fetched over plain HTTP, never rendered.** A browser hands it back wrapped in
`<html><body><pre>`, which parses as an *empty* robots file, which permits everything.

**Null is not zero.** Fixed six times: `gap_score`, price, hours, salary, arrangement, and a `NaN`
that reached a live card. A value nobody measured renders as nothing at all — never as 0, never as a
midpoint of a range, never as a guess.

**Attach, never describe.** In Hud's chat a posting arrives as a whole card carrying its own `why`,
`source` and `retrievedAt`. The model never emits a card: it names a handle from the fact sheet and
`resolve_refs` in `agents/agent_s_assistant/facts.py` decides whether that handle points at anything
real. This separation is the entire reason the mascot is allowed on a screen the brand otherwise
fences him away from.

**The model proposes, the user disposes, code executes.** Nothing a model returns spends a credit,
starts a run, or reaches another user's data.

**Never send mail to a reserved TLD from production.** `.test`, `.invalid`, `.example` and
`.localhost` can never resolve, so each message is a guaranteed hard bounce charged against sender
reputation. Enforced by `is_undeliverable` in `api/email.py` — added after roughly seven such bounces
from test accounts preceded a delivery outage.

## Hazards that cost real time

**psycopg opens a transaction on the first statement.** A bare read before the first
`with conn.transaction()` means every later one nests as a savepoint, and releasing a savepoint
commits nothing. This silently discarded an entire scrape cycle while the run log said it had written
25 rows. Both stores now connect with `autocommit=True`.

**`tsc -b`, never `tsc --noEmit`.** The root tsconfig is `{files: [], references: []}`, so `--noEmit`
type-checks zero files and exits 0 — proven by planting a deliberate error and still getting a pass.

**i18n parity is CI-enforced** in both front ends. A key added to one language fails the build.

**A new table without a foreign key to `app_users` must be named in the API suite's TRUNCATE list**
(`_APP_TABLES` in `tests/api/conftest.py`). `CASCADE` does not reach it, its counters accumulate
across tests, and the symptom is fifteen unrelated assertions failing.

**Deploy the serving side first.** The API began redirecting new signups to a page the site had not
shipped yet, and every signup dead-ended on a 404 for about fifteen minutes. When one side redirects
to something the other must serve, the server of that thing goes first.

**The deploy does not apply agent migrations.** Only `api/` self-migrates on boot; Agent B's and
Agent D's run from their CLIs (`python main.py agent-d --migrate`). Restoring a dump can leave the
schema ahead of the tracking table, which then fails on a later `--migrate`.

**Check that `git fetch` actually succeeded** before reasoning about divergence. A silent failure left
a stale `origin/main`, and a fast-forward then swept in an unrelated tree.

## Open threads

- **SPF is broken.** `tryitqan.com` publishes `v=spf1 -all`, which authorises *nobody* to send as the
  domain. DKIM (`brevo1` / `brevo2` selectors) and the Brevo verification token are present, so the
  fix is one record — Brevo documents `include:spf.brevo.com`. Confirm the exact value on Brevo's
  domain-authentication page before changing it.
- **Nothing knows whether an email was delivered.** The counters stop at the relay's 250, which means
  *queued*. The answer is Brevo's delivery webhook: a public endpoint, signature verification, a
  table. Named in `api/email.py` rather than left as an oversight.
- **edX** — build the catalog adapter when credentials arrive.
- **`is_known_unchanged` does not reach Agent D's adapters in production**, the same gap Agent B
  closed for itself. It starts to matter once edX is per-request.
- **Agent S's two quota numbers** (30 messages a day, 1 rerun a week) are decisions with no data
  behind them. Worth re-measuring after real use: how often anyone reaches the cap, and how often a
  rerun changed anything.

## Where the reasoning lives

The full decision history — every change with the measurements that justified it, including the ones
that overturned an assumption — is in the plan file, on the machine it was written on:

```
~/.claude/plans/using-langchain-and-langgraph-prancy-goose.md
```

**It is not in OneDrive and does not sync.** Copy it deliberately if you want it. Conversation
transcripts live beside it under `~/.claude/projects/<project>/` and are local to that machine too —
a Claude account carries auth and billing, not CLI history.
