# agent/state.py
from typing import TypedDict, Optional

class HedgeState(TypedDict):
    # Input — market conditions
    stock_price: float
    strike_price: float
    time_to_expiry: float
    volatility: float
    risk_free_rate: float

    # Scenario input (optional)
    scenario_description: Optional[str]   # e.g. "price drops 10%"
    scenario_stock_price: Optional[float] # adjusted price after scenario

    # Node 1 output
    market_data: Optional[dict]

    # Node 2 output
    volatility_regime: Optional[str]
    should_hedge: Optional[bool]

    # Node 3 output — RL decision
    rl_action: Optional[dict]
    hedge_recommendation: Optional[str]

    # Node 4 output — Comparison
    comparison: Optional[dict]   # RL vs Delta vs No Hedge metrics

    # Node 5 output — Report
    risk_report: Optional[str]

    # Decision flow trace
    decision_flow: Optional[list]  # tracks which nodes ran and why

    # Meta
    error: Optional[str]