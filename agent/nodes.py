# agent/nodes.py
import os
import sys
import numpy as np
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from agent.state import HedgeState
from scipy.stats import norm

load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile")

# ── Black-Scholes helpers ──────────────────────────────────────────
def bs_delta(S, K, T, r, sigma):
    T = max(T, 1e-10)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return float(norm.cdf(d1))

def bs_call(S, K, T, r, sigma):
    T = max(T, 1e-10)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return float(S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))


# ── NODE 1: Market Monitor ─────────────────────────────────────────
def market_monitor(state: HedgeState) -> HedgeState:
    print("[Node 1] Monitoring market conditions...")

    stock_price = state["stock_price"]
    if state.get("scenario_description") and state.get("scenario_stock_price"):
        stock_price = state["scenario_stock_price"]
        print(f"  Scenario active: {state['scenario_description']}")

    # Calculate actual BS delta and option price from inputs
    delta = bs_delta(
        stock_price, state["strike_price"],
        state["time_to_expiry"], state["risk_free_rate"],
        state["volatility"]
    )
    option_price = bs_call(
        stock_price, state["strike_price"],
        state["time_to_expiry"], state["risk_free_rate"],
        state["volatility"]
    )

    state["market_data"] = {
        "stock_price": stock_price,
        "option_price": round(option_price, 4),
        "time_to_expiry": state["time_to_expiry"],
        "current_hedge_position": 0.0,
        "delta": round(delta, 4),
        "volatility": state["volatility"],
        "risk_free_rate": state["risk_free_rate"]
    }

    state["decision_flow"] = [{
        "node": "MarketMonitor",
        "action": "Packaged market state",
        "detail": (
            f"S={stock_price}, K={state['strike_price']}, "
            f"σ={state['volatility']}, T={state['time_to_expiry']}, "
            f"BS Delta={round(delta, 4)}, Option Price={round(option_price, 4)}"
        )
    }]

    print(f"  S={stock_price}, delta={round(delta,4)}, option={round(option_price,4)}")
    return state


# ── NODE 2: Volatility Analyzer ────────────────────────────────────
def volatility_analyzer(state: HedgeState) -> HedgeState:
    print("[Node 2] Analyzing volatility regime...")

    vol = state["volatility"]

    if vol < 0.15:
        state["volatility_regime"] = "low"
        state["should_hedge"] = False
        reason = "Volatility below 0.15 — hedging cost exceeds benefit"
    elif vol < 0.30:
        state["volatility_regime"] = "medium"
        state["should_hedge"] = True
        reason = "Volatility 0.15–0.30 — moderate hedging recommended"
    else:
        state["volatility_regime"] = "high"
        state["should_hedge"] = True
        reason = "Volatility above 0.30 — aggressive hedging required"

    state["decision_flow"].append({
        "node": "VolatilityAnalyzer",
        "action": f"Classified as {state['volatility_regime'].upper()} regime",
        "detail": reason
    })

    print(f"  Regime: {state['volatility_regime']} → Hedge: {state['should_hedge']}")
    return state


# ── NODE 3: Hedge Decider ──────────────────────────────────────────
def hedge_decider(state: HedgeState) -> HedgeState:
    """
    Calls PPO model directly for single-step hedge recommendation.
    Note: PPO was trained episodically (252 steps). Single-step inference
    shows the agent's first action from position 0 toward BS delta.
    Full performance metrics (96.5% error reduction) are in /simulate endpoint.
    """
    print("[Node 3] Getting hedge decision from RL model...")

    try:
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from stable_baselines3 import PPO

        model = PPO.load("models/ppo_hedging_agent")

        S = state["market_data"]["stock_price"]
        C = state["market_data"]["option_price"]
        T = state["market_data"]["time_to_expiry"]
        delta = state["market_data"]["delta"]

        S_norm = S / 100.0
        C_norm = C / (S * 0.5 + 1e-8)
        T_norm = T
        pos = 0.0

        obs = np.array([S_norm, C_norm, T_norm, pos, delta],
                       dtype=np.float32)
        obs = np.clip(obs, [0.0, 0.0, 0.0, -1.0, 0.0],
                           [5.0, 5.0, 1.0,  1.0, 1.0])

        action, _ = model.predict(obs, deterministic=True)
        action_value = float(np.clip(action[0], -1.0, 1.0))

        new_position = float(np.clip(pos + action_value * 0.1, -1.0, 1.0))
        hedging_error = abs(new_position - delta)

        strategy = "increase" if action_value > 0.05 else \
                   "decrease" if action_value < -0.05 else "hold"

        state["rl_action"] = {
            "recommended_hedge_position": round(new_position, 4),
            "hedge_action_delta": round(action_value, 4),
            "hedging_error_vs_bs": round(hedging_error, 4),
            "strategy": strategy
        }

        state["hedge_recommendation"] = (
            f"Strategy: {strategy.upper()} | "
            f"Recommended Position: {round(new_position, 4)} | "
            f"BS Delta: {round(delta, 4)}"
        )

        state["decision_flow"].append({
            "node": "HedgeDecider",
            "action": f"RL Agent recommends: {strategy.upper()}",
            "detail": (
                f"Recommended position: {round(new_position, 4)}, "
                f"BS Delta (target): {round(delta, 4)}, "
                f"Single-step gap: {round(hedging_error, 4)}"
            )
        })

        print(f"  RL: {state['hedge_recommendation']}")

    except Exception as e:
        state["error"] = f"Model prediction failed: {str(e)}"
        print(f"  Error: {str(e)}")

    return state


# ── NODE 4: Context Node (replaces Comparison) ────────────────────
def comparison_node(state: HedgeState) -> HedgeState:
    """
    Provides market context for the report generator.
    No winner logic — episodic performance is in /simulate endpoint.
    """
    print("[Node 4] Building market context...")

    try:
        delta = state["market_data"]["delta"]
        stock_price = state["market_data"]["stock_price"]
        option_price = state["market_data"]["option_price"]

        # Context for report generator
        state["comparison"] = {
            "bs_delta": round(delta, 4),
            "option_price": option_price,
            "moneyness": "in-the-money" if stock_price > state["strike_price"]
                         else "out-of-the-money" if stock_price < state["strike_price"]
                         else "at-the-money",
            "rl_recommended_position": state.get("rl_action", {}).get(
                "recommended_hedge_position", "N/A"),
            "note": (
                "Single-step inference. Full episodic performance "
                "(96.5% error reduction over 252 steps) shown in /simulate endpoint."
            )
        }

        state["decision_flow"].append({
            "node": "ContextNode",
            "action": "Built market context for report",
            "detail": (
                f"BS Delta: {round(delta, 4)}, "
                f"Moneyness: {state['comparison']['moneyness']}, "
                f"Option Price: {option_price}"
            )
        })

        print(f"  Context: delta={round(delta,4)}, {state['comparison']['moneyness']}")

    except Exception as e:
        print(f"  Context error: {str(e)}")

    return state


# ── NODE 5: Report Generator ───────────────────────────────────────
def report_generator(state: HedgeState) -> HedgeState:
    print("[Node 5] Generating risk report...")

    context = state.get("comparison", {})
    scenario_text = ""
    if state.get("scenario_description"):
        scenario_text = (
            f"\nScenario applied: {state['scenario_description']} "
            f"→ adjusted price: {state.get('scenario_stock_price', 'N/A')}"
        )

    if not state.get("should_hedge"):
        prompt = f"""You are a quantitative risk analyst.
        
        Market conditions:
        - Stock: {state['stock_price']}, Strike: {state['strike_price']}
        - Volatility: {state['volatility']} — LOW regime
        - Time to Expiry: {state['time_to_expiry']} years
        - BS Delta: {context.get('bs_delta', 'N/A')}
        - Option: {context.get('moneyness', 'N/A')}, Price: {context.get('option_price', 'N/A')}
        {scenario_text}
        
        Write a concise 3-sentence risk report explaining:
        1. Why volatility is low and hedging is not cost-effective
        2. Current risk exposure level
        3. What to monitor going forward
        """
    else:
        prompt = f"""You are a quantitative risk analyst.
        
        Market conditions:
        - Stock: {state['stock_price']}, Strike: {state['strike_price']}
        - Volatility: {state['volatility']} — {state['volatility_regime'].upper()} regime
        - Time to Expiry: {state['time_to_expiry']} years
        - Risk-free Rate: {state['risk_free_rate']}
        - BS Delta: {context.get('bs_delta', 'N/A')}
        - Option: {context.get('moneyness', 'N/A')}, Price: {context.get('option_price', 'N/A')}
        {scenario_text}
        
        RL Agent recommendation: {state.get('hedge_recommendation', 'N/A')}
        
        Write a concise 4-sentence professional risk report covering:
        1. Current risk exposure and volatility regime
        2. What the BS delta tells us about option sensitivity
        3. The RL agent's recommended hedge action
        4. Key risk factors to monitor
        
        Do NOT compare RL vs delta hedge accuracy — focus on market conditions
        and the hedging recommendation.
        """

    response = llm.invoke(prompt)
    state["risk_report"] = response.content

    state["decision_flow"].append({
        "node": "ReportGenerator",
        "action": "Generated LLM risk report",
        "detail": "Report based on market conditions and RL hedge recommendation"
    })

    print("  Report generated ✓")
    return state