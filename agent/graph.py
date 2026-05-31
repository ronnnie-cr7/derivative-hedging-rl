# agent/graph.py
from langgraph.graph import StateGraph, END
from agent.state import HedgeState
from agent.nodes import (
    market_monitor,
    volatility_analyzer,
    hedge_decider,
    comparison_node,
    report_generator
)

# ── Conditional edge ───────────────────────────────────────────────
def should_execute_hedge(state: HedgeState) -> str:
    if state.get("error"):
        return "compare"
    if not state.get("should_hedge"):
        return "compare"
    return "execute"

# ── Build graph ────────────────────────────────────────────────────
def build_hedge_graph():
    workflow = StateGraph(HedgeState)

    # Add all 5 nodes
    workflow.add_node("monitor", market_monitor)
    workflow.add_node("analyze", volatility_analyzer)
    workflow.add_node("execute", hedge_decider)
    workflow.add_node("compare", comparison_node)
    workflow.add_node("report", report_generator)

    # Entry point
    workflow.set_entry_point("monitor")

    # Edges
    workflow.add_edge("monitor", "analyze")

    # Conditional — hedge or skip to comparison
    workflow.add_conditional_edges(
        "analyze",
        should_execute_hedge,
        {
            "execute": "execute",
            "compare": "compare"
        }
    )

    workflow.add_edge("execute", "compare")
    workflow.add_edge("compare", "report")
    workflow.add_edge("report", END)

    return workflow.compile()

hedge_agent = build_hedge_graph()