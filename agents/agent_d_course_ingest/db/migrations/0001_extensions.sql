-- pgvector, for course essence embeddings. IF NOT EXISTS because Agent D shares
-- the itqan database with Agent B, which may have created it already; this makes
-- Agent D's migration set self-contained against a virgin database too.
CREATE EXTENSION IF NOT EXISTS vector;
