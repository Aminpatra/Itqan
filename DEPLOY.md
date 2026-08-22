# Deploying Itqan

**One box serves everything.** After the first setup, every push to either repo
deploys itself.

```
   https://<your-domain>
        │
      caddy ── /          the marketing site   (static, from ./web)
        │   ── /app/*     the onboarding app   (static + SPA fallback)
        │   ── /api/*     FastAPI (OCR) ──▶ postgres (pgvector)
        │
      TLS obtained and renewed automatically
```

**Why one host, and not two.** `src/api/http.ts` sends
`credentials: 'same-origin'` and the session cookie is set by the marketing
site's own login form — so if the app or the API sat on a second domain, the
browser would silently withhold that cookie and every authenticated call would
401 with nothing in the console to explain it. Serving all three from one Caddy
makes that structurally impossible rather than correctly configured.

Consequences worth knowing: **never set `VITE_API_BASE_URL`** (unset it defaults
to `/api`, which is the point), and `SITE_ORIGIN` in `Onboarding/src/lib/site.ts`
stays empty so links to the site are plain paths.

## What it costs

| | |
|---|---|
| OVH VPS-1 (2 vCore, 4 GB, 40 GB NVMe, Singapore) | $5.75/mo |
| OVH snapshot backup | $0.40/mo |
| Domain | ~$1/mo |
| **Total** | **~$7/mo** |

Against ~$44/mo for the managed equivalent (Render Standard 2 GB + Neon Launch),
and unlike a free tier this one runs OCR, holds the full 1.7 GB corpus, keeps its
disk, never sleeps, and can schedule the ingestion agents.

**What is given up by not using a CDN:** static assets come from Singapore rather
than an edge near the user, and there are no per-PR preview URLs. The first is
mitigated by cache headers (hashed assets are `immutable` for a year, so only
HTML crosses the network on a return visit); the second is a real workflow loss,
which is why the deploy job is gated on the full Playwright suite.

---

## 1. The domain

Buy one, then point an **A record** at the VPS IP. Do this first: DNS propagation
and the Let's Encrypt certificate both wait on it, and it is the only step with a
delay you cannot shorten.

```
@      A     <vps-ip>
www    A     <vps-ip>        # Caddy redirects www to the apex
```

## 2. The box

Order the VPS with the **Docker (Debian 12)** image — everything runs in
containers, so the host is only a Docker host, and Debian matches the
`python:3.13-slim` base the API is built on.

```bash
ssh debian@<vps-host>
curl -fsSL https://raw.githubusercontent.com/Aminpatra/Itqan/main/scripts/vps-bootstrap.sh | sudo bash
```

That creates the `itqan` deploy user, locks the firewall to 22/80/443, enables
unattended security upgrades, and **adds a 2 GB swap file** — the last one is not
optional, see *Memory* below.

## 3. The stack

```bash
sudo -u itqan -i
git clone https://github.com/Aminpatra/Itqan.git /opt/itqan && cd /opt/itqan
cp .env.example .env
```

Fill in `.env`:

| Variable | |
|---|---|
| `ITQAN_DOMAIN` | your domain, bare (`itqan.om`, not `https://itqan.om`) — what Caddy requests the certificate for |
| `POSTGRES_PASSWORD` | anything long and random |
| `ITQAN_SESSION_SECRET` | `python -c "import secrets;print(secrets.token_hex(32))"` |
| `OPENAI_API_KEY` | your key |
| `ITQAN_UNLIMITED_EMAILS` | comma-separated developer addresses, exempt from the assistant quotas |

`ITQAN_UNLIMITED_EMAILS` is optional and easy to forget, and forgetting it is
silent: the developers are simply rationed like everyone else — 30 assistant
messages a day, one re-run a week — with nothing anywhere saying why. The API
logs how many accounts are exempt at boot, so `0` in the deploy log is the tell.
It lives in the environment rather than the repository because this repository is
public and committed personal addresses stay scrapeable, history included.

`ITQAN_SESSION_SECRET` is not optional: session cookies are
`user_id.HMAC(secret, user_id)` and the development fallback is public in this
repository, so without a real value every account is forgeable. The app refuses
to boot rather than allow it, and compose refuses to start without the others.

```bash
docker compose up -d
docker compose ps          # three services, api healthy
```

## 4. The corpus

From your laptop, with the local Postgres running:

```bash
python scripts/seed_remote_db.py --target "postgresql://itqan:<pw>@<vps-host>:5432/itqan"
```

This copies **everything** — 2,099 courses, 400 postings, the stats tables and all
100,350 ESCO labels **with their embeddings** (1,706 MB of the 40 GB disk). The
embedding tier is what maps skills the exact and alt-label tiers miss — two thirds
of all successful mappings — and step 4 depends on it.

> Postgres is not published to the internet (`expose`, not `ports`), so open 5432
> to your IP for the duration of the seed, or run the script through an SSH
> tunnel: `ssh -L 5432:localhost:5432 itqan@<vps-host>`.

## 5. Keep it fresh

```bash
sudo cp /opt/itqan/scripts/itqan-cron /etc/cron.d/itqan && sudo chmod 644 /etc/cron.d/itqan
```

Agent B every 12h (job postings), Agent D every 3 days (courses). Job postings
expire — a product about live vacancies showing a frozen snapshot ages visibly.
This is the capability a free tier could not have at all.

## 6. Backend deploys on push

On GitHub → repo → Settings → Secrets → Actions:

| Secret | |
|---|---|
| `VPS_HOST` | the VPS hostname or IP |
| `VPS_USER` | `itqan` |
| `VPS_SSH_KEY` | the private half of a key whose public half is in `/home/itqan/.ssh/authorized_keys` |

**The box must be able to pull from GHCR.** Images published from a private repo
are private, so `docker compose pull` fails with `denied` until the VPS has a
read token. Once, as the `itqan` user:

```bash
# GitHub -> Settings -> Developer settings -> Personal access tokens (classic)
# one scope: read:packages
echo "<token>" | docker login ghcr.io -u <your-github-username> --password-stdin
```

(Or make the package public: GitHub -> the repo -> Packages -> the image ->
Package settings -> Change visibility. Then no login is needed, but anyone can
pull your image — which contains no secrets, but does contain your source.)

Then push. `.github/workflows/deploy.yml` runs the 808 tests, builds the image on
**GitHub's** runners, pushes it to GHCR, and SSHes in to pull and restart —
waiting on the healthcheck rather than declaring victory when the container is
created.

Building on GitHub rather than on the box is deliberate: the OCR image is ~2 GB
and installing paddle would compete for the 4 GB the live service is using.

## 7. Frontend deploys on push

In `abujamal3221-eng/itqan`, set the same three secrets, plus one **variable**:

| | |
|---|---|
| secret `VPS_HOST` / `VPS_USER` / `VPS_SSH_KEY` | as above |
| variable `ITQAN_SITE_URL` | `https://<your-domain>` — **with** the scheme |

`ITQAN_SITE_URL` is baked in at build time: Astro uses it for canonical URLs,
`hreflang` and the sitemap, so it cannot be read at serve time. It lives in one
place and flows into `astro.config.mjs`, `src/config.ts` and `Base.astro`.

Then push. The workflow typechecks, checks i18n parity across both languages,
runs Playwright, builds both apps into one tree, and `rsync`s the 7.5 MB to
`/opt/itqan/web` — `--delay-updates`, so no request is ever served a half-copied
site. It then curls `/`, `/app/` and `/api/health` and fails if any is not 200.

## 8. Check it

```bash
curl -s https://<your-domain>/api/health       # {"ok":true}  <- TLS is the gate
curl -sI https://<your-domain>/app/confirm | head -1   # 200: the SPA fallback works
```

Then in a browser:

1. sign up on the site → land in the app **signed in**;
2. upload a CV with selectable text → `reading` → `awaiting_confirmation` with your
   details filled in;
3. confirm → `matching` → `done`, dashboard fills in by itself;
4. upload a **scanned** CV → it is read. This is the capability the free tier could
   not have, and the only real proof `WITH_OCR=1` took;
5. `free -h` during that OCR run — the number that decides whether 4 GB was enough.

Verified locally against this exact Caddyfile and a real build: `/`, `/ar/`,
`/en/…`, `/app/` and `/app/confirm` all 200, `/nonsense` serves Astro's own 404
page, hashed assets come back `immutable`, HTML comes back `no-cache`, and
responses are gzipped.

---

## Memory

4 GB, and the peaks overlap:

| | |
|---|---|
| Postgres | ~700 MB (pinned in `docker-compose.yml`, not left to grow) |
| API idle | 145 MB, measured |
| **OCR while reading a scanned page** | **~1.2 GB peak, measured** |
| OS + Docker | ~300 MB |

That is ~2.3 GB of 4 GB with everything busy at once — comfortable, and better
than the 3.5 GB first estimated, because two settings were needed to make OCR run
in a container at all and both cut memory hard:

- **`PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT=False`** (`shared/config.py`). Without it
  every `predict()` raises `NotImplementedError ... onednn_instruction.cc:116`.
  The models load and the engine constructs, so nothing catches it until a real
  scanned page arrives.
- **`PP-OCRv5_mobile_det`** rather than PaddleOCR's server default
  (`Config.ocr_detection_model`). The server detector needed more than 3 GB for
  one CV page and was OOM-killed in a 2 GB container; the mobile one reads the
  same page correctly at 1,228 MB.

The swap file stays as the backstop, and `shared_buffers` is pinned rather than
left to grow. If step 7.5 says otherwise, the answer is to move Postgres to a
managed host — not to shrink the corpus.

## If the certificate does not issue

`docker compose logs caddy` says why, plainly. The usual causes are DNS not yet
pointing at the box, or port 80 blocked — Let's Encrypt's HTTP-01 challenge lands
there, which is why `vps-bootstrap.sh` opens it. Using your own domain avoids the
rate-limit trap that OVH's shared `vps.ovh.net` hostname carries.

## Turning OCR off

If the box is ever too small: `WITH_OCR: "0"` in `docker-compose.yml`. The image
drops to ~670 MB, `ocr_available()` returns False, and a scanned CV is refused by
name with the manual-entry route offered. Nothing else changes.
