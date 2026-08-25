"""End-to-end graph checks using a fake LLM, so CI does not need an API key."""

from types import SimpleNamespace

import pytest
from langgraph.types import Command

from langgraph_agent_lab import nodes
from langgraph_agent_lab.graph import build_graph, export_mermaid
from langgraph_agent_lab.persistence import build_checkpointer
from langgraph_agent_lab.state import Route, Scenario, initial_state


class FakeLLM:
    def __init__(self, schema=None):
        self.schema = schema

    def with_structured_output(self, schema, **kwargs):
        return FakeLLM(schema)

    def invoke(self, prompt):
        if self.schema is not None:
            text = str(prompt[-1].content).lower()
            if "refund" in text or "delete" in text:
                route = "risky"
            elif "lookup" in text:
                route = "tool"
            elif "fix it" in text:
                route = "missing_info"
            elif "timeout" in text or "failure" in text:
                route = "error"
            else:
                route = "simple"
            return self.schema(route=route, rationale="fake test classification")
        return SimpleNamespace(content="Grounded fake support response.")


@pytest.fixture
def fake_llm(monkeypatch):
    monkeypatch.setattr(nodes, "get_llm", FakeLLM)


@pytest.mark.parametrize(
    ("query", "expected_route", "max_attempts"),
    [
        ("How do I reset my password?", Route.SIMPLE, 3),
        ("Please lookup order status", Route.TOOL, 3),
        ("Can you fix it?", Route.MISSING_INFO, 3),
        ("Refund this customer", Route.RISKY, 3),
        ("Timeout failure while processing", Route.ERROR, 3),
        ("System failure cannot recover", Route.ERROR, 1),
    ],
)
def test_graph_routes_and_finalizes_offline(fake_llm, query, expected_route, max_attempts):
    graph = build_graph(checkpointer=build_checkpointer("memory"))
    scenario = Scenario(
        id=f"offline-{expected_route.value}-{max_attempts}",
        query=query,
        expected_route=expected_route,
        max_attempts=max_attempts,
    )
    state = initial_state(scenario)
    result = graph.invoke(state, config={"configurable": {"thread_id": state["thread_id"]}})

    assert result["route"] == expected_route.value
    assert result["final_answer"]
    assert any(event["node"] == "finalize" for event in result["events"])
    if expected_route is Route.RISKY:
        assert result["approval"]["approved"] is True
    if max_attempts == 1:
        assert any(event["node"] == "dead_letter" for event in result["events"])


def test_sqlite_checkpointer_records_state_history(fake_llm, tmp_path):
    checkpointer = build_checkpointer("sqlite", str(tmp_path / "checkpoints.sqlite"))
    graph = build_graph(checkpointer=checkpointer)
    scenario = Scenario(
        id="sqlite-history",
        query="How do I reset my password?",
        expected_route=Route.SIMPLE,
    )
    state = initial_state(scenario)
    config = {"configurable": {"thread_id": state["thread_id"]}}

    result = graph.invoke(state, config=config)

    assert result["final_answer"]
    assert len(list(graph.get_state_history(config))) > 1


def test_real_hitl_rejection_resumes_same_thread_to_clarification(fake_llm, monkeypatch):
    monkeypatch.setenv("LANGGRAPH_INTERRUPT", "true")
    graph = build_graph(checkpointer=build_checkpointer("memory"))
    scenario = Scenario(
        id="hitl-rejection",
        query="Delete this customer account",
        expected_route=Route.RISKY,
    )
    state = initial_state(scenario)
    config = {"configurable": {"thread_id": state["thread_id"]}}

    interrupted = graph.invoke(state, config=config)
    assert "__interrupt__" in interrupted

    resumed = graph.invoke(
        Command(resume={"approved": False, "reviewer": "qa", "comment": "Do not proceed."}),
        config=config,
    )
    assert resumed["approval"]["approved"] is False
    assert resumed["pending_question"]
    assert not any(event["node"] == "tool" for event in resumed["events"])
    assert any(event["node"] == "finalize" for event in resumed["events"])


def test_export_mermaid_uses_compiled_graph(tmp_path):
    output = export_mermaid(build_graph(), tmp_path / "workflow.mmd")

    diagram = output.read_text(encoding="utf-8")
    assert "risky_action" in diagram
    assert "dead_letter" in diagram
