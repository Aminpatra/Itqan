"""The web app's backend: FastAPI joining the agent pipeline to the frontend.

An orchestrator, not an agent. It may import ``shared/`` and drive the agents
through their public CLIs — the same latitude ``agents/pipeline.py`` and
``agents/status.py`` have — but it never reaches into an agent's internals, and it
reads the pipeline's tables only through ``shared/job_market.py`` and
``shared/course_market.py`` so the eligibility predicates stay single-sourced.
"""
