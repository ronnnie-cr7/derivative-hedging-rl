import os
from dotenv import load_dotenv
load_dotenv()

import httpx
from langchain_groq import ChatGroq
from agent.state import HedgeState

# Initialize LLM
llm = ChatGroq(model="llama-3.3-70b-versatile")

# — call locally since agent runs on same server
API_BASE = "http://localhost:7860"

# ── NODE 1: Market Monitor ─────────────────────────────────────────
def market_monitor(state: HedgeState) -> HedgeState:
    """
    Packages market state — no HTTP call needed.
    """
    print("[Node 1] Monitoring market conditions...")

    state["market_data"] = {
        "stock_price": state["stock_price"],
        "option_price": state["stock_price"] * 0.1,
        "time_to_expiry": state["time_to_expiry"],
        "current_hedge_position": 0.0,
        "delta": 0.5,
        "volatility": state["volatility"],
        "risk_free_rate": state["risk_free_rate"]
    }

    print(f"  Market data ready: S={state['stock_price']}, σ={state['volatility']}")
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


# ── NODE 3: Hedge Decider (calls PPO model directly) ──────────────
def hedge_decider(state: HedgeState) -> HedgeState:
    """
    Calls PPO model directly instead of via HTTP API.
    """
    print("[Node 3] Getting hedge decision from RL model...")

    try:
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        from stable_baselines3 import PPO
        import numpy as np

        model = PPO.load("models/ppo_hedging_agent")
        
        obs = np.array([
            state["market_data"]["stock_price"] / 100.0,
            state["market_data"]["option_price"] / 50.0,
            state["market_data"]["time_to_expiry"],
            state["market_data"]["current_hedge_position"],
            state["market_data"]["delta"]
        ], dtype=np.float32)

        obs = np.clip(obs, [0.0, 0.0, 0.0, -1.0, 0.0],
                           [5.0, 5.0, 1.0,  1.0, 1.0])

        action, _ = model.predict(obs, deterministic=True)
        action_value = float(np.clip(action[0], -1.0, 1.0))

        new_position = float(np.clip(
            state["market_data"]["current_hedge_position"] + action_value * 0.1,
            -1.0, 1.0
        ))

        hedging_error = abs(new_position - state["market_data"]["delta"])

        if action_value > 0.05:
            strategy = "increase"
        elif action_value < -0.05:
            strategy = "decrease"
        else:
            strategy = "hold"

        state["rl_action"] = {
            "recommended_hedge_position": round(new_position, 4),
            "hedge_action_delta": round(action_value, 4),
            "hedging_error_vs_bs": round(hedging_error, 4),
            "strategy": strategy
        }

        state["hedge_recommendation"] = (
            f"Strategy: {strategy.upper()} | "
            f"Hedge Position: {round(new_position, 4)} | "
            f"Error vs BS: {round(hedging_error, 4)}"
        )

        print(f"  RL Decision: {state['hedge_recommendation']}")

    except Exception as e:
        state["error"] = f"Model prediction failed: {str(e)}"
        print(f"  Error: {str(e)}")

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