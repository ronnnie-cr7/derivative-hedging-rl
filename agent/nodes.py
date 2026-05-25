import os
from dotenv import load_dotenv
load_dotenv()

import httpx
from langchain_groq import ChatGroq
from agent.state import HedgeState

# Initialize LLM
llm = ChatGroq(model="llama-3.3-70b-versatile")

# live HF Spaces API
API_BASE = "https://ronityadav8905-alphahedge.hf.space"

# ── NODE 1: Market Monitor ─────────────────────────────────────────
def market_monitor(state: HedgeState) -> HedgeState:
    """
    Fetches current market state and checks API health.
    """
    print("[Node 1] Monitoring market conditions...")

    try:
        # Check if API is alive
        response = httpx.get(f"{API_BASE}/health", timeout=30)
        health = response.json()

        if not health["model_loaded"]:
            state["error"] = "RL model not loaded on server"
            return state

        # Package market data for next nodes
        state["market_data"] = {
            "stock_price": state["stock_price"],
            "option_price": state["stock_price"] * 0.1,  # rough estimate
            "time_to_expiry": state["time_to_expiry"],
            "current_hedge_position": 0.0,
            "delta": 0.5,
            "volatility": state["volatility"],
            "risk_free_rate": state["risk_free_rate"]
        }

        print(f"  Market data ready: S={state['stock_price']}, "
              f"σ={state['volatility']}")

    except Exception as e:
        state["error"] = f"API unreachable: {str(e)}"

    return state


# ── NODE 2: Volatility Analyzer ────────────────────────────────────
def volatility_analyzer(state: HedgeState) -> HedgeState:
    """
    Classifies volatility regime and decides whether to hedge.
    """
    print("[Node 2] Analyzing volatility regime...")

    vol = state["volatility"]

    if vol < 0.15:
        state["volatility_regime"] = "low"
        state["should_hedge"] = False
    elif vol < 0.30:
        state["volatility_regime"] = "medium"
        state["should_hedge"] = True
    else:
        state["volatility_regime"] = "high"
        state["should_hedge"] = True

    print(f"  Volatility: {vol} → Regime: {state['volatility_regime']} "
          f"→ Hedge: {state['should_hedge']}")

    return state


# ── NODE 3: Hedge Decider (calls your RL API) ──────────────────────
def hedge_decider(state: HedgeState) -> HedgeState:
    """
    Calls your live PPO model via FastAPI to get hedge recommendation.
    """
    print("[Node 3] Calling RL agent for hedge decision...")

    try:
        response = httpx.post(
            f"{API_BASE}/predict",
            json=state["market_data"],
            timeout=30
        )
        rl_action = response.json()
        state["rl_action"] = rl_action

        state["hedge_recommendation"] = (
            f"Strategy: {rl_action['strategy'].upper()} | "
            f"Hedge Position: {rl_action['recommended_hedge_position']} | "
            f"Error vs BS: {rl_action['hedging_error_vs_bs']}"
        )

        print(f"  RL Decision: {state['hedge_recommendation']}")

    except Exception as e:
        state["error"] = f"Prediction failed: {str(e)}"

    return state


# ── NODE 4: Report Generator (LLM) ────────────────────────────────
def report_generator(state: HedgeState) -> HedgeState:
    """
    Uses Groq LLM to generate a structured risk report
    based on market conditions and RL agent decision.
    """
    print("[Node 4] Generating risk report...")

    if state.get("should_hedge") is False:
        prompt = f"""
        You are a quantitative risk analyst.
        
        Market conditions:
        - Stock Price: {state['stock_price']}
        - Volatility: {state['volatility']} (LOW regime)
        - Time to Expiry: {state['time_to_expiry']} years
        
        The volatility is LOW — hedging is NOT recommended.
        
        Write a concise 3-sentence risk report explaining why 
        no hedge is needed and what to monitor.
        """
    else:
        prompt = f"""
        You are a quantitative risk analyst.
        
        Market conditions:
        - Stock Price: {state['stock_price']}
        - Volatility: {state['volatility']} ({state['volatility_regime'].upper()} regime)
        - Time to Expiry: {state['time_to_expiry']} years
        - Risk Free Rate: {state['risk_free_rate']}
        
        RL Agent Decision: {state.get('hedge_recommendation', 'N/A')}
        
        Write a concise 3-sentence professional risk management 
        report covering: current risk exposure, the recommended 
        hedge action, and key risk factors to monitor.
        """

    response = llm.invoke(prompt)
    state["risk_report"] = response.content

    print(f"  Report generated ✓")
    return state