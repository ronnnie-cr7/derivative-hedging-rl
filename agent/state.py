from typing import TypedDict, Optional

class HedgeState(TypedDict):
    """
    Shared state that flows through all 4 nodes.
    Every node reads this and updates it.
    """
    # Input — market conditions
    stock_price: float
    strike_price: float
    time_to_expiry: float
    volatility: float
    risk_free_rate: float

    # Node 1 output — raw market data fetched
    market_data: Optional[dict]

    # Node 2 output — volatility analysis
    volatility_regime: Optional[str]      # "low" / "medium" / "high"
    should_hedge: Optional[bool]

    # Node 3 output — RL agent decision
    rl_action: Optional[dict]
    hedge_recommendation: Optional[str]

    # Node 4 output — final report
    risk_report: Optional[str]

    # Meta
    error: Optional[str]