"""Graph assembly for Agent S.

    START
      -> answer     the one LLM call, fenced to a fact sheet built in code
      -> verify     every figure must be in that fact sheet, or it is replaced
      -> END

No checkpointer and no interrupt: a question in, an answer out. The graph is
here for uniformity with Agents A-E rather than for control flow it needs.

There is deliberately no node that reads a database, spends a credit, or starts
a run. Agent S is handed everything it may know and returns text; the API layer
around it does the rest. That is what makes "the model cannot act" a fact about
the code rather than a promise in a prompt.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from .nodes import Deps, make_answer, make_verify
from .state import AssistantState


def build_assistant_graph(deps: Deps):
    builder = StateGraph(AssistantState)

    builder.add_node("answer", make_answer(deps))
    builder.add_node("verify", make_verify(deps))

    builder.add_edge(START, "answer")
    builder.add_edge("answer", "verify")
    builder.add_edge("verify", END)

    return builder.compile()
