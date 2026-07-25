"""Agent E: course recommender — closes the loop A -> C -> E.

Recommends exactly one course per confirmed missing skill. Selection is
DETERMINISTIC (a weighted greedy set-cover, no LLM); the ONLY LLM call is a short
grounded rationale per recommendation, fenced to the already-finalized record so
it can decorate the decision but never change it. Imports from ``shared/`` only.
"""
