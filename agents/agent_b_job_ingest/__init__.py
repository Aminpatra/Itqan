"""Agent B — scheduled job-posting ingestion.

Outputs are two Postgres tables, ``job_postings`` and ``skill_demand_stats``.
Agent B never calls Agent C or any user-facing code, and never reads
``CandidateProfile`` — that is Agent C's input, not Agent B's.
"""
