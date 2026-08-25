# Day 08 Lab Report

## 1. Team / student

- Name: Bui Van Khoi
- Date: Generated from the latest scenario run

## 2. Architecture

The workflow is `intake -> classify` followed by conditional routing. Simple requests answer directly; tool and approved risky requests run through `tool -> evaluate`; failures use a bounded retry loop and end in dead letter when exhausted. Every branch passes through `finalize` before `END`.

## 3. State schema

Append-only fields are `messages`, `tool_results`, `errors`, and `events` for auditability. The current decision fields (`route`, `attempt`, `evaluation_result`, `pending_question`, `proposed_action`, `approval`, and `final_answer`) overwrite their prior value to keep state compact and serializable.

## 4. Scenario results

| Total scenarios | Success rate | Avg. nodes visited | Total retries | Total approvals | State-history replay |
|---:|---:|---:|---:|---:|---:|
| 7 | 100.00% | 6.43 | 3 | 2 | True |

| Scenario | Expected route | Actual route | Success | Retries | Interrupts |
|---|---|---|---:|---:|---:|
| S01_simple | simple | simple | True | 0 | 0 |
| S02_tool | tool | tool | True | 0 | 0 |
| S03_missing | missing_info | missing_info | True | 0 | 0 |
| S04_risky | risky | risky | True | 0 | 1 |
| S05_error | error | error | True | 2 | 0 |
| S06_delete | risky | risky | True | 0 | 1 |
| S07_dead_letter | error | error | True | 1 | 0 |

## 5. Failure analysis

1. Transient tool errors are detected by `evaluate`, counted by `retry`, and constrained by `max_attempts`; exhausted work is sent to `dead_letter` with a safe escalation answer.
2. High-impact actions are represented as a proposed action and must pass `approval`. A rejected approval is routed to clarification rather than executing the tool.

## 6. Persistence / recovery evidence

Each run uses a distinct `thread_id`. When a checkpointer is configured, the CLI reads state history after execution; the `resume_success` metric records that checkpoint history was available for replay or recovery.

## 7. Extension work

Baseline behavior uses mock approval so automated runs are non-interactive. Set `LANGGRAPH_INTERRUPT=true` to pause at `approval`; resume the same `thread_id` with an explicit reviewer decision. A rejection is routed to clarification and never executes the tool.

The compiled graph is exported as Mermaid when `diagram_path` is configured. This creates a diagram from the actual registered nodes and conditional edges rather than maintaining a hand-drawn copy. SQLite persistence remains available through `SqliteSaver` with WAL mode for durable checkpoints.

## 8. Improvement plan

Next, add domain tools with authenticated data access, LLM-as-judge evaluation, real human approval UI, and tracing around model and tool latency.
