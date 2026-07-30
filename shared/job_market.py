"""The job-market read surface — how Agent C sees Agent B's tables.

Read-only by construction: nothing in this module writes a row, and nothing in
it calls an LLM. It exists because the architecture forbids Agent C from
importing ``agents.agent_b_job_ingest.*`` (agents share nothing but ``shared/``),
so the two output tables plus the ESCO vocabulary need a published surface here.

The SQL in this module is the *contract* surface, not Agent B pipeline SQL — the
"every Agent B SQL string lives in db/store.py" rule governs the pipeline's own
modules and is untouched. Where a predicate must be shared with the pipeline it
is defined HERE and imported by the pipeline (``aggregate.py``), because
agent → shared is the only import direction the architecture allows.

Vector-space contract: query embeddings MUST come from
``shared.embeddings.build_embedder`` over candidate essence text shaped like a
posting essence (title / scalars / skills lines). Same model, same dimensions,
same space — or every similarity number here is meaningless.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator, Optional

from pydantic import BaseModel

from .config import Config
from .contracts import JobPostingExport, SkillDemandStatRow

# ---------------------------------------------------------------------------
# The eligibility predicate — single source of truth.
#
# This is migration 0006's job_postings_aggregable_idx predicate: the four
# conditions under which a posting is real demand. Aggregation imports it from
# here; retrieval applies it below. A posting failing any clause is stored for
# audit but must never be served to Agent C — a duplicate double-counts one job,
# a seeker's advert is supply mislabeled as demand, an unknown poster is a
# classification that has not happened yet.
# ---------------------------------------------------------------------------
AGGREGABLE_POSTING_PREDICATE = (
    "status = 'active' AND duplicate_of IS NULL "
    "AND listing_intent = 'vacancy' AND poster_type = 'company'"
)


# ---------------------------------------------------------------------------
# connection plumbing (local: shared/ cannot import Agent B's JobStore)
# ---------------------------------------------------------------------------
@contextmanager
def _connection(config: Config) -> Iterator[Any]:
    import psycopg
    from psycopg.rows import dict_row

    conn = psycopg.connect(config.require_database_url(), row_factory=dict_row)
    try:
        try:
            from pgvector.psycopg import register_vector

            register_vector(conn)
        except Exception:
            # Tolerated exactly as JobStore tolerates it: a virgin database
            # without the extension should fail on the query, not the import.
            pass
        yield conn
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------
def export_for_agent_c(
    query_embedding: list[float],
    top_k: int = 15,
    isco08_sector: Optional[str] = None,
    *,
    config: Config | None = None,
) -> dict[str, list[Any]]:
    """Retrieve the job market for one candidate query.

    Returns ``{"job_postings": [JobPostingExport...], "skill_demand_stats":
    [SkillDemandStatRow...]}`` — nearest eligible postings by essence similarity,
    plus the aggregated demand rows (for fallback when retrieval is thin).

    Stats are LATEST WINDOW ONLY, per the consumer contract stated on the
    skill_demand_stats schema: history rows exist for audit, and serving an old
    window as current demand would be exactly the stale-statistic mistake the
    contract warns about.
    """
    config = config or Config()

    if len(query_embedding) != config.embedding_dims:
        raise ValueError(
            f"query_embedding has {len(query_embedding)} dimensions; this corpus "
            f"is embedded at {config.embedding_dims}. Use shared.embeddings."
            "build_embedder — a vector from any other model is not comparable."
        )
    if isco08_sector is not None and isco08_sector not in set("0123456789"):
        raise ValueError(
            f"isco08_sector must be a single ISCO-08 major-group digit '0'-'9', "
            f"got {isco08_sector!r}"
        )

    with _connection(config) as conn:
        with conn.transaction():
            with conn.cursor() as cur:
                # Same three rules as the pipeline's near-dup search, verbatim:
                # ef_search raised so a filtered HNSW scan cannot silently return
                # fewer rows than asked; similarity is 1 - distance (<=> is
                # cosine DISTANCE); the filter implies the partial index
                # predicate (duplicate_of IS NULL AND status <> 'rejected').
                cur.execute("SET LOCAL hnsw.ef_search = 100")
                cur.execute(
                    f"""
                    SELECT posting_id, source, source_group, source_type, source_url,
                           title, raw_description, sector, required_skills,
                           company, seniority_level, location, country, posted_date,
                           legitimacy_score, listing_intent, poster_type,
                           first_seen_at, last_seen_at,
                           1 - (embedding <=> %(emb)s::vector) AS similarity
                      FROM job_postings
                     WHERE {AGGREGABLE_POSTING_PREDICATE}
                       AND country = ANY(%(countries)s)
                       AND embedding IS NOT NULL
                       -- The SAME window the demand stats use, and the same
                       -- COALESCE expression (aggregate.py's _ELIGIBLE_CTE).
                       -- Without it the two halves of one export were drawn from
                       -- different time populations: stats windowed to 30 days,
                       -- postings unbounded. Nothing is currently older than the
                       -- window, but only as a side effect of the scraper's
                       -- lookback and cycle cadence — an index page that keeps
                       -- re-listing a dead vacancy would keep it 'active' forever
                       -- and Agent C would serve it as current demand while Agent
                       -- B had already stopped counting it.
                       AND COALESCE(posted_date, first_seen_at::date)
                           > current_date - %(window_days)s::int
                     ORDER BY embedding <=> %(emb)s::vector
                     LIMIT %(k)s
                    """,
                    {
                        "emb": query_embedding,
                        "countries": list(config.in_scope_countries),
                        "k": top_k,
                        "window_days": config.window_days,
                    },
                )
                postings = [_posting_row(dict(r)) for r in cur.fetchall()]

        with conn.cursor() as cur:
            sector_clause = "AND sector = %(sector)s" if isco08_sector is not None else ""
            cur.execute(
                f"""
                SELECT sector, skill, skill_key, esco_code, window_start, window_end,
                       frequency_count, prior_frequency_count, trend,
                       co_occurring_skills, sample_postings, low_confidence, computed_at,
                       sector_volume, distinct_posts
                  FROM skill_demand_stats
                 WHERE window_end = (SELECT max(window_end) FROM skill_demand_stats)
                       {sector_clause}
                 ORDER BY frequency_count DESC, skill_key
                """,
                {"sector": isco08_sector},
            )
            stats = [_stat_row(dict(r)) for r in cur.fetchall()]

    return {"job_postings": postings, "skill_demand_stats": stats}


def _posting_row(row: dict[str, Any]) -> JobPostingExport:
    for key in ("posted_date", "first_seen_at", "last_seen_at"):
        if row.get(key) is not None:
            row[key] = row[key].isoformat()
    if row.get("legitimacy_score") is not None:
        row["legitimacy_score"] = float(row["legitimacy_score"])
    row["similarity"] = float(row["similarity"])
    return JobPostingExport(**row)


def _stat_row(row: dict[str, Any]) -> SkillDemandStatRow:
    for key in ("window_start", "window_end", "computed_at"):
        row[key] = row[key].isoformat()
    return SkillDemandStatRow(**row)


# ---------------------------------------------------------------------------
# ESCO mapping, read-only — Agent C's side of the shared vocabulary
# ---------------------------------------------------------------------------
class SkillMapping(BaseModel):
    """One free-text skill resolved (or honestly not) against ESCO.

    Same tiers, same tables and same threshold as Agent B's mapper, so both
    sides of a match speak one vocabulary. READ-ONLY by design decision:
    candidate phrases never enter ``skill_esco_map`` — that table records the
    POSTING corpus and is written only by Agent B; a candidate's near-miss
    scores belong in Agent C's own output as tuning evidence.
    """

    skill: str
    skill_key: str
    esco_uri: Optional[str] = None
    esco_label: Optional[str] = None
    method: str = "unmapped"        # exact | alt_label | embedding | unmapped | deferred
    similarity: Optional[float] = None


def map_skills_to_esco(
    skills: list[str],
    *,
    embedder: Any,
    config: Config | None = None,
) -> list[SkillMapping]:
    """Resolve free-text skills against the ESCO tables, without writing.

    Tier precedence mirrors ``agents.agent_b_job_ingest.esco_map.map_new_skills``
    exactly — exact preferred label, exact alt label, nearest-label embedding at
    ``config.esco_map_threshold``, else unmapped with the near-miss score. With
    no embedder the lexical misses come back ``method='deferred'``: "we did not
    look" and "we looked and found nothing" are different answers here too.
    """
    from .embeddings import embed_texts

    config = config or Config()
    if not skills:
        return []

    keyed = [(s, s.strip().lower()) for s in skills if s and s.strip()]
    keys = [k for _, k in keyed]

    with _connection(config) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT DISTINCT ON (l.label_key)
                       l.label_key, l.esco_uri, l.is_preferred, s.preferred_label
                  FROM esco_labels l
                  JOIN esco_skills s ON s.esco_uri = l.esco_uri
                 WHERE l.label_key = ANY(%s)
                 ORDER BY l.label_key, l.is_preferred DESC, l.esco_uri
                """,
                (keys,),
            )
            lexical = {r["label_key"]: dict(r) for r in cur.fetchall()}

        results: list[SkillMapping] = []
        remaining: list[tuple[str, str]] = []
        for skill, key in keyed:
            hit = lexical.get(key)
            if hit:
                results.append(SkillMapping(
                    skill=skill, skill_key=key,
                    esco_uri=hit["esco_uri"], esco_label=hit["preferred_label"],
                    method="exact" if hit["is_preferred"] else "alt_label",
                ))
            else:
                remaining.append((skill, key))

        if remaining and embedder is not None:
            vectors = embed_texts(embedder, [k for _, k in remaining])
            threshold = config.esco_map_threshold
            with conn.transaction():
                with conn.cursor() as cur:
                    cur.execute("SET LOCAL hnsw.ef_search = 100")
                    for (skill, key), vector in zip(remaining, vectors):
                        cur.execute(
                            """
                            SELECT l.esco_uri, s.preferred_label,
                                   1 - (l.embedding <=> %(emb)s::vector) AS similarity
                              FROM esco_labels l
                              JOIN esco_skills s ON s.esco_uri = l.esco_uri
                             WHERE l.embedding IS NOT NULL
                             ORDER BY l.embedding <=> %(emb)s::vector
                             LIMIT 1
                            """,
                            {"emb": vector},
                        )
                        best = cur.fetchone()
                        if best and float(best["similarity"]) >= threshold:
                            results.append(SkillMapping(
                                skill=skill, skill_key=key,
                                esco_uri=best["esco_uri"],
                                esco_label=best["preferred_label"],
                                method="embedding",
                                similarity=round(float(best["similarity"]), 4),
                            ))
                        else:
                            results.append(SkillMapping(
                                skill=skill, skill_key=key, method="unmapped",
                                similarity=round(float(best["similarity"]), 4) if best else None,
                            ))
        else:
            for skill, key in remaining:
                results.append(SkillMapping(skill=skill, skill_key=key, method="deferred"))

    return results
