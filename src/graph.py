"""
LangGraph StateGraph definition for the Map-Reduce audit workflow.

Graph topology
--------------

    [START]
       │
  ┌────▼────┐
  │ ingestor │   (sync — populates layers with L1-L4 engine outputs)
  └────┬────┘
       │  fan-out (all 4 run in parallel)
  ┌────┼────────────────────────────┐
  │    │                            │
  ▼    ▼         ▼                  ▼
architect  quantifier  mapmaker  refactor
  │    │         │                  │
  └────┴────┬────┘──────────────────┘
            │  fan-in (all must complete)
       ┌────▼────┐
       │aggregator│  (sync — merges all outputs)
       └────┬────┘
            │
          [END]
"""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from models import AuditState
from nodes import (
    aggregator_node,
    architect_node,
    ingestor_node,
    mapmaker_node,
    quantifier_node,
    refactor_node,
)


def build_graph() -> StateGraph:
    """
    Construct and compile the audit StateGraph.

    Parallel workers (architect, quantifier, mapmaker, refactor) are wired
    with individual edges to aggregator_node, which LangGraph treats as a
    fan-in barrier — all four must complete before aggregator runs.
    """
    builder = StateGraph(AuditState)

    # Register nodes
    builder.add_node("ingestor", ingestor_node)
    builder.add_node("architect", architect_node)
    builder.add_node("quantifier", quantifier_node)
    builder.add_node("mapmaker", mapmaker_node)
    builder.add_node("refactor", refactor_node)
    builder.add_node("aggregator", aggregator_node)

    # Entry: START → ingestor
    builder.add_edge(START, "ingestor")

    # Fan-out: ingestor → all four parallel workers
    builder.add_edge("ingestor", "architect")
    builder.add_edge("ingestor", "quantifier")
    builder.add_edge("ingestor", "mapmaker")
    builder.add_edge("ingestor", "refactor")

    # Fan-in: all four workers → aggregator
    builder.add_edge("architect", "aggregator")
    builder.add_edge("quantifier", "aggregator")
    builder.add_edge("mapmaker", "aggregator")
    builder.add_edge("refactor", "aggregator")

    # Exit: aggregator → END
    builder.add_edge("aggregator", END)

    return builder.compile()


# Compiled graph — imported by orchestrator.py
audit_graph = build_graph()
