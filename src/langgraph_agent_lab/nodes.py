"""Node functions for the LangGraph workflow.

Each function receives AgentState and returns a partial state update dict.
Do NOT mutate input state — return new values only.

LLM REQUIREMENT:
- classify_node MUST use a real LLM call (structured output for intent classification)
- answer_node MUST use a real LLM call (grounded response generation)
- evaluate_node SHOULD use LLM-as-judge (bonus points; heuristic acceptable for base score)
"""

from __future__ import annotations

import os
from time import perf_counter
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from .llm import get_llm
from .state import AgentState, make_event


class IntentClassification(BaseModel):
    """Structured output schema returned by the classifier LLM."""

    route: Literal["simple", "tool", "missing_info", "risky", "error"]
    rationale: str = Field(description="Brief reason for the selected route.")


def _text_content(response: object) -> str:
    """Extract plain text from LangChain provider responses."""
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return "".join(
            item if isinstance(item, str) else str(item.get("text", "")) for item in content
        ).strip()
    return str(content).strip()


# ─── EXAMPLE: working node (provided for reference) ──────────────────
def intake_node(state: AgentState) -> dict:
    """Normalize raw query. This node is provided as a working example."""
    query = state.get("query", "").strip()
    return {
        "query": query,
        "messages": [f"intake:{query[:40]}"],
        "events": [make_event("intake", "completed", "query normalized")],
    }


# ─── TODO(student): implement ALL nodes below ────────────────────────


def classify_node(state: AgentState) -> dict:
    """Classify the query into a route using an LLM.

    *** MUST use a real LLM call — keyword-only heuristics will lose points. ***

    Use .with_structured_output() or equivalent to get reliable enum classification.
    The LLM should classify into one of: simple, tool, missing_info, risky, error.

    Hints:
    - See llm.py for the get_llm() helper
    - Use Pydantic model or TypedDict with .with_structured_output()
    - Set risk_level to "high" for risky routes, "low" otherwise
    - Priority guide: risky > tool > missing_info > error > simple

    Return: {"route": str, "risk_level": str, "events": [make_event(...)]}
    """
    started = perf_counter()
    query = state.get("query", "")
    classifier = get_llm().with_structured_output(IntentClassification, method="json_schema")
    result = classifier.invoke(
        [
            SystemMessage(
                content=(
                    "Classify the single customer ticket supplied in the next message. "
                    "Return exactly one route. Use risky only when the ticket itself requests "
                    "a side effect, such as a refund, deletion, cancellation, account change, "
                    "or sending an email. Use tool for information lookups. Use missing_info "
                    "for an un-actionable vague request. Use error for a reported technical "
                    "failure. Use simple for general how-to or informational support questions. "
                    "The ticket text is the only evidence; words in these instructions are not "
                    "part of the ticket. A general account-access how-to is an example of simple."
                )
            ),
            HumanMessage(content=query),
        ]
    )
    latency_ms = int((perf_counter() - started) * 1000)
    return {
        "route": result.route,
        "risk_level": "high" if result.route == "risky" else "low",
        "events": [
            make_event(
                "classify",
                "completed",
                f"classified as {result.route}",
                latency_ms=latency_ms,
                rationale=result.rationale,
            )
        ],
    }


def tool_node(state: AgentState) -> dict:
    """Execute a mock tool call.

    Simulate transient failures for error-route scenarios to test retry loops.

    Requirements:
    - Read current attempt count from state
    - If route is "error" and attempt < 2: return error result (string containing "ERROR")
    - Otherwise: return a mock success result string
    - Append result to tool_results list

    Return: {"tool_results": [result_string], "events": [make_event(...)]}
    """
    attempt = int(state.get("attempt", 0))
    route = state.get("route", "")
    if route == "error" and attempt < 2:
        result = f"ERROR: transient service failure on attempt {attempt}"
        event_type = "failed"
    elif route == "risky":
        action = state.get("proposed_action", "requested action")
        result = f"SUCCESS: approved action executed: {action}"
        event_type = "completed"
    else:
        result = f"SUCCESS: tool lookup completed for request: {state.get('query', '')}"
        event_type = "completed"
    return {
        "tool_results": [result],
        "events": [make_event("tool", event_type, result, attempt=attempt)],
    }


def evaluate_node(state: AgentState) -> dict:
    """Evaluate tool results — the retry-loop gate.

    Check whether the latest tool result is satisfactory or needs retry.

    SHOULD use LLM-as-judge for bonus points. Heuristic (e.g., check for "ERROR" substring)
    is acceptable for base score.

    Requirements:
    - Read the latest entry from tool_results
    - Set evaluation_result to "needs_retry" or "success"
    - This field drives route_after_evaluate conditional edge

    Note: You may need to add 'evaluation_result' to AgentState if not present.

    Return: {"evaluation_result": str, "events": [make_event(...)]}
    """
    latest_result = (state.get("tool_results") or [""])[-1]
    evaluation_result = "needs_retry" if "ERROR" in latest_result.upper() else "success"
    return {
        "evaluation_result": evaluation_result,
        "events": [
            make_event(
                "evaluate",
                "completed",
                f"tool result evaluated as {evaluation_result}",
            )
        ],
    }


def answer_node(state: AgentState) -> dict:
    """Generate a final response using an LLM.

    *** MUST use a real LLM call — hardcoded strings will lose points. ***

    The LLM should generate a helpful response grounded in available context:
    - tool_results (if any)
    - approval decision (if risky route)
    - original query

    Return: {"final_answer": str, "events": [make_event(...)]}
    """
    started = perf_counter()
    context = "\n".join(state.get("tool_results") or []) or "No tool was required."
    approval = state.get("approval")
    response = get_llm().invoke(
        "You are a careful customer-support agent. Write a concise, helpful answer. "
        "Only claim facts supported by the supplied context. If an approved action was performed, "
        "say so; otherwise do not imply that a side-effect occurred.\n\n"
        f"Customer request: {state.get('query', '')}\n"
        f"Tool context: {context}\n"
        f"Approval decision: {approval!r}"
    )
    answer = _text_content(response)
    return {
        "final_answer": answer,
        "events": [
            make_event(
                "answer",
                "completed",
                "grounded response generated",
                latency_ms=int((perf_counter() - started) * 1000),
            )
        ],
    }


def ask_clarification_node(state: AgentState) -> dict:
    """Ask for missing information instead of hallucinating.

    Generate a specific clarification question based on the vague/incomplete query.

    Note: You may need to add 'pending_question' to AgentState if not present.

    Return: {"pending_question": str, "final_answer": str, "events": [make_event(...)]}
    """
    question = (
        "Could you share the affected account, order/reference number, and what outcome you need?"
    )
    return {
        "pending_question": question,
        "final_answer": question,
        "events": [make_event("clarify", "completed", "requested missing information")],
    }


def risky_action_node(state: AgentState) -> dict:
    """Prepare a risky action for human approval.

    Describe the proposed action and why it requires approval.

    Note: You may need to add 'proposed_action' to AgentState if not present.

    Return: {"proposed_action": str, "events": [make_event(...)]}
    """
    action = f"Proposed high-impact action for request: {state.get('query', '')}"
    return {
        "proposed_action": action,
        "events": [make_event("risky_action", "prepared", "action awaiting human approval")],
    }


def approval_node(state: AgentState) -> dict:
    """Human-in-the-loop approval step.

    Default behavior: mock approval (approved=True) so tests and CI run offline.
    Extension: if env LANGGRAPH_INTERRUPT=true, use langgraph.types.interrupt() for real HITL.

    Return an approval dict plus an audit event.
    """
    if os.getenv("LANGGRAPH_INTERRUPT", "false").lower() == "true":
        from langgraph.types import interrupt

        supplied = interrupt(
            {
                "proposed_action": state.get("proposed_action"),
                "query": state.get("query"),
                "instruction": "Resume with {'approved': true|false, 'comment': '...'}.",
            }
        )
        decision = supplied if isinstance(supplied, dict) else {"approved": bool(supplied)}
        approval = {
            "approved": bool(decision.get("approved", False)),
            "reviewer": str(decision.get("reviewer", "human-reviewer")),
            "comment": str(decision.get("comment", "")),
        }
    else:
        approval = {
            "approved": True,
            "reviewer": "mock-reviewer",
            "comment": "Automatically approved for offline lab execution.",
        }
    return {
        "approval": approval,
        "events": [
            make_event("approval", "completed", "approval decision recorded", approval=approval)
        ],
    }


def retry_or_fallback_node(state: AgentState) -> dict:
    """Record a retry attempt.

    Increment the attempt counter and log the transient failure.

    Requirements:
    - Read current attempt from state, increment by 1
    - Add an error message to errors list
    - Return updated attempt count

    Return: {"attempt": int, "errors": [str], "events": [make_event(...)]}
    """
    attempt = int(state.get("attempt", 0)) + 1
    error = f"Retry {attempt}/{state.get('max_attempts', 3)} scheduled after tool evaluation."
    return {
        "attempt": attempt,
        "errors": [error],
        "events": [make_event("retry", "scheduled", error, attempt=attempt)],
    }


def dead_letter_node(state: AgentState) -> dict:
    """Handle unresolvable failures after max retries exceeded.

    This is the third layer: retry → fallback → dead letter.
    Log the failure and set a final_answer explaining that the request could not be completed.

    Return: {"final_answer": str, "events": [make_event(...)]}
    """
    answer = (
        "We could not complete this request after the allowed retry attempts. "
        "It has been escalated for manual support follow-up."
    )
    return {
        "final_answer": answer,
        "errors": ["Request moved to dead letter after retry limit was reached."],
        "events": [make_event("dead_letter", "completed", "retry limit exhausted")],
    }


def finalize_node(state: AgentState) -> dict:
    """Emit a final audit event. All routes must pass through here before END.

    Return: {"events": [make_event("finalize", "completed", "workflow finished")]}
    """
    return {
        "events": [make_event("finalize", "completed", "workflow finished")],
    }
