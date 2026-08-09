# The Itqan backend: the FastAPI app at /api and the five agents behind it.
#
# The VPS deployment builds this with OCR ON:
#
#     docker build --build-arg WITH_OCR=1 .   # ~2 GB, scanned CVs work
#     docker build .                          # ~670 MB, scanned CVs refused by name
#
# The flag stays because it is the difference between fitting a 512 MB free tier
# and not, and that option is worth keeping open. Nothing is removed for the small
# build: `ocr_available()` returns False and a scanned document is refused with a
# sentence written for the user (`ingestion/detect._NO_OCR`).
FROM python:3.13-slim

# libgomp1: numpy and paddle both link OpenMP. curl: the HEALTHCHECK below.
# Cleanup in the same layer so the apt lists never reach the image.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 curl \
 && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Requirements first, so editing source does not re-resolve the dependency tree —
# and, with OCR on, does not re-download 1.2 GB of paddle.
COPY requirements.txt requirements-ocr.txt ./

ARG WITH_OCR=0
RUN pip install --no-cache-dir -r requirements.txt \
 && if [ "$WITH_OCR" = "1" ]; then \
        echo "building WITH OCR (~1.2 GB of paddle)"; \
        # PaddleOCR pulls in OpenCV, which links libGL and glib even for
        # headless use. `python:3.13-slim` has neither, and the failure is a
        # runtime `ImportError: libGL.so.1` on the first scanned page rather
        # than anything visible at install time — found by the pre-warm step
        # below, which is most of why it exists.
        apt-get update && apt-get install -y --no-install-recommends \
            libgl1 libglib2.0-0 \
         && rm -rf /var/lib/apt/lists/*; \
        pip install --no-cache-dir -r requirements-ocr.txt; \
    else \
        echo "building WITHOUT OCR — scanned documents will be refused by name"; \
    fi

# Chromium for the crawl transport, behind its own flag.
#
# `playwright install --with-deps` pulls the browser binary (~450 MB) plus the
# shared libraries a headless Chromium needs. Only the ingestion agents use it;
# the API serving user requests never launches a browser, so a build that only
# serves does not need to carry one.
ARG WITH_BROWSER=0
RUN if [ "$WITH_BROWSER" = "1" ]; then         echo "building WITH the browser transport (~450 MB of Chromium)";         playwright install --with-deps chromium;     else         echo "building WITHOUT the browser transport";     fi

COPY . .

# Pre-warm the OCR model weights INTO the image.
#
# PaddleOCR fetches them on first use, so without this the first user to submit a
# scanned CV after every deploy pays ~40 seconds of download inside their own
# request. Baking them in turns that into a build-time constant and makes the
# image self-contained — it no longer reaches PaddlePaddle's CDN at runtime.
#
# This FAILS THE BUILD if the download fails, deliberately: an unwarmed image
# looks identical and quietly breaks the promise this layer exists to make, and
# re-running a build is the cheaper of the two problems.
RUN if [ "$WITH_OCR" = "1" ]; then \
        echo "pre-warming PaddleOCR weights"; \
        python -c "from agents.agent_a_cv_extraction.ingestion.ocr import get_engine; get_engine('en')"; \
    fi

# Uploads and per-run artifacts. Bind-mounted to a host directory by
# docker-compose so they survive `up -d --build`; the profile Agent C needs is
# ALSO in `app_runs.profile` and rehydrated from there if this is ever empty.
RUN mkdir -p /data/output
ENV ITQAN_OUTPUT_DIR=/data/output

ENV PORT=8000
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD curl -fsS "http://127.0.0.1:${PORT}/api/health" || exit 1

# The factory form. `api.main:app` starts, prints "startup complete", and then
# 500s every request — the module has no `app`, deliberately, because building it
# connects to Postgres and applies migrations.
#
# ONE worker, and that is load-bearing: a run is a background thread holding
# in-process state, so a second worker would answer polls for a job it is not
# running. Concurrency comes from threads inside this process, not from forks.
CMD ["sh", "-c", "uvicorn api.main:get_app --factory --host 0.0.0.0 --port ${PORT} --workers 1"]
