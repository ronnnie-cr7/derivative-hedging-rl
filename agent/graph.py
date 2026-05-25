# agent/graph.py
from langgraph.graph import StateGraph, END
from agent.state import HedgeState
from agent.nodes import (
    market_monitor,
    volatility_analyzer,
    hedge_decider,
    report_generator
)

# ── Conditional edge function ──────────────────────────────────────
def should_execute_hedge(state: HedgeState) -> str:
    """
    After volatility analysis:
    - If error occurred → go to report directly
    - If low volatility → skip hedge, go to report
    - Otherwise → call RL agent
    """
    if state.get("error"):
        return "report"
    if not state.get("should_hedge"):
        return "report"
    return "execute"

# ── Build the graph ────────────────────────────────────────────────
def build_hedge_graph():
    workflow = StateGraph(HedgeState)

    # Add all 4 nodes
    workflow.add_node("monitor", market_monitor)
    workflow.add_node("analyze", volatility_analyzer)
    workflow.add_node("execute", hedge_decider)
    workflow.add_node("report", report_generator)

    # Entry point
    workflow.set_entry_point("monitor")

    # Edges
    workflow.add_edge("monitor", "analyze")

    # Conditional — hedge or skip straight to report
    workflow.add_conditional_edges(
        "analyze",
        should_execute_hedge,
        {
            "execute": "execute",
            "report": "report"
        }
    )

    workflow.add_edge("execute", "report")
    workflow.add_edge("report", END)

    return workflow.compile()

# Compiled agent — import this everywhere
hedge_agent = build_hedge_graph()