"""Graph construction.

This module is intentionally import-safe. It imports LangGraph only inside the builder so unit tests
that check schema/metrics can run even if students are still debugging graph wiring.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .state import AgentState


def build_graph(checkpointer: Any | None = None) -> Any:
    """Build and compile the LangGraph workflow.

    TODO(student): Build the complete graph with this architecture:

    START → intake → classify → [conditional: route_after_classify]
      simple       → answer → finalize → END
      tool         → tool → evaluate → [conditional: route_after_evaluate]
                                          success → answer → finalize → END
                                          needs_retry → retry → [conditional: route_after_retry]
                                                                  tool (retry)
                                                                  dead_letter → finalize → END
      missing_info → clarify → finalize → END
      risky        → risky_action → approval → [conditional: route_after_approval]
                                                  approved → tool → evaluate → ...
                                                  rejected → clarify → finalize → END
      error        → retry → [conditional: route_after_retry] → ...

    Steps:
    1. Import StateGraph, START, END from langgraph.graph
    2. Create StateGraph(AgentState)
    3. Import and add all nodes from nodes.py (11 nodes total)
    4. Import and use routing functions from routing.py for conditional edges
    5. Add fixed edges (e.g., START→intake, intake→classify, tool→evaluate, etc.)
    6. Add conditional edges using add_conditional_edges()
    7. Compile with checkpointer: graph.compile(checkpointer=checkpointer)

    Reference: https://langchain-ai.github.io/langgraph/how-tos/create-react-agent/
    """
    from langgraph.graph import END, START, StateGraph

    from .nodes import (
        answer_node,
        approval_node,
        ask_clarification_node,
        classify_node,
        dead_letter_node,
        evaluate_node,
        finalize_node,
        intake_node,
        retry_or_fallback_node,
        risky_action_node,
        tool_node,
    )
    from .routing import (
        route_after_approval,
        route_after_classify,
        route_after_evaluate,
        route_after_retry,
    )

    workflow = StateGraph(AgentState)
    workflow.add_node("intake", intake_node)
    workflow.add_node("classify", classify_node)
    workflow.add_node("tool", tool_node)
    workflow.add_node("evaluate", evaluate_node)
    workflow.add_node("answer", answer_node)
    workflow.add_node("clarify", ask_clarification_node)
    workflow.add_node("risky_action", risky_action_node)
    workflow.add_node("approval", approval_node)
    workflow.add_node("retry", retry_or_fallback_node)
    workflow.add_node("dead_letter", dead_letter_node)
    workflow.add_node("finalize", finalize_node)

    workflow.add_edge(START, "intake")
    workflow.add_edge("intake", "classify")
    workflow.add_conditional_edges("classify", route_after_classify)
    workflow.add_edge("tool", "evaluate")
    workflow.add_conditional_edges("evaluate", route_after_evaluate)
    workflow.add_conditional_edges("retry", route_after_retry)
    workflow.add_edge("risky_action", "approval")
    workflow.add_conditional_edges("approval", route_after_approval)

    for node in ("answer", "clarify", "dead_letter"):
        workflow.add_edge(node, "finalize")
    workflow.add_edge("finalize", END)
    return workflow.compile(checkpointer=checkpointer)


def export_mermaid(graph: Any, output_path: str | Path) -> Path:
    """Export the compiled graph's actual Mermaid diagram to a file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(graph.get_graph().draw_mermaid(), encoding="utf-8")
    return path
