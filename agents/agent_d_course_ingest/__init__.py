"""Agent D: scheduled course ingestion — the supply side of the skill market.

Agent B ingests job postings (skill demand); Agent D ingests courses (skill
supply). Same architecture, same ESCO vocabulary, same store patterns. Its
outputs are the ``courses`` and ``skill_supply_stats`` tables; joined with Agent
B's ``skill_demand_stats`` on ``esco_code`` they answer "in-demand skills with
few courses". Imports from ``shared/`` only — never from another agent.
"""
