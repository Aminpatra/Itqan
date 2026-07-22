-- pgvector. Required before any table declares a `vector` column.
--
-- Kept in its own migration because CREATE EXTENSION is the one statement here
-- that can need superuser, so a failure at this step has a completely different
-- remedy from a failure in a CREATE TABLE and should be isolated.

CREATE EXTENSION IF NOT EXISTS vector;
