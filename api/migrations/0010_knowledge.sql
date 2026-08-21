-- The knowledge base: what Itqan is, in Itqan's own words.
--
-- Agent S could only ever answer questions about the person asking. Asked what
-- Itqan actually IS, it declined -- correctly, because the fact sheet is one
-- user's rows and nothing else, and answering from general knowledge about
-- career sites is exactly the fabrication every fence in this system exists to
-- prevent.
--
-- These rows are the second thing it may draw on. They are written by us, in
-- `docs/knowledge/`, reviewed as source, and ingested here -- NOT scraped, not
-- user-supplied, and not the model's own recollection. That is what makes them
-- safe to quote: `verify_answer` treats a figure in a retrieved passage exactly
-- as it treats a figure in the fact sheet, because both are things we actually
-- showed the model.
--
-- IN api/migrations DELIBERATELY. `api/db.apply_migrations` runs on boot; agent
-- migrations are applied by hand and the deploy does not run them (HANDOFF.md).
-- A table the assistant needs, living where nothing applies it, would mean the
-- first request after a deploy raises instead of answering.
--
-- GLOBAL, with no user_id and no foreign key to app_users. There is nothing
-- per-person here, so the isolation question does not arise: this table cannot
-- leak one user's data to another because it never held any.
CREATE TABLE IF NOT EXISTS app_knowledge_chunks (
    chunk_id     text        PRIMARY KEY,
    -- The document this came from, e.g. 'what-is-itqan'. One slug spans both
    -- languages; `locale` separates them.
    doc_slug     text        NOT NULL,
    -- The document's own heading, stored per chunk and prepended to the text on
    -- ingest. A passage retrieved on its own must still say what it is about --
    -- "Deleting removes the file itself" means little without "What Itqan
    -- stores" above it.
    title        text        NOT NULL,
    locale       text        NOT NULL CHECK (locale IN ('en', 'ar')),
    -- Position within the document, so a passage can be shown in reading order
    -- and so re-ingesting is a deterministic replace rather than an append.
    ordinal      integer     NOT NULL CHECK (ordinal >= 0),
    text         text        NOT NULL,
    -- sha256 of the chunk text. The warm-cycle gate Agents B and D already use:
    -- re-ingesting an unchanged document costs zero embedding calls, which is
    -- what makes running the ingest after every edit cheap enough to actually do.
    content_hash text        NOT NULL,
    embedding    vector(1536),
    ingested_at  timestamptz NOT NULL DEFAULT now(),
    UNIQUE (doc_slug, locale, ordinal)
);

-- Retrieval is always scoped to a language first, so the index carries it.
CREATE INDEX IF NOT EXISTS app_knowledge_chunks_locale_idx
    ON app_knowledge_chunks (locale);

-- Same shape as job_postings and esco_labels: cosine, because the embeddings are
-- normalised and every other vector query in this system is `1 - (a <=> b)`.
-- Partial on a populated embedding so a chunk awaiting its vector -- which is the
-- state between INSERT and the embedding call -- never enters the index.
CREATE INDEX IF NOT EXISTS app_knowledge_chunks_embedding_hnsw
    ON app_knowledge_chunks USING hnsw (embedding vector_cosine_ops)
    WHERE embedding IS NOT NULL;
