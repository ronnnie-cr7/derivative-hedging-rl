# run_agent.py
import os
from dotenv import load_dotenv
from agent.graph import hedge_agent

load_dotenv()

# ── Test Case 1: HIGH volatility → should hedge ───────────────────
print("\n" + "="*55)
print("TEST 1: High Volatility Market")
print("="*55)

result = hedge_agent.invoke({
    "stock_price": 105.0,
    "strike_price": 100.0,
    "time_to_expiry": 0.4,
    "volatility": 0.35,        # HIGH → should hedge
    "risk_free_rate": 0.05,
    "market_data": None,
    "volatility_regime": None,
    "should_hedge": None,
    "rl_action": None,
    "hedge_recommendation": None,
    "risk_report": None,
    "error": None
})

print(f"\n✅ Volatility Regime: {result['volatility_regime']}")
print(f"✅ Should Hedge: {result['should_hedge']}")
print(f"✅ RL Decision: {result['hedge_recommendation']}")
print(f"\n📋 Risk Report:\n{result['risk_report']}")

# ── Test Case 2: LOW volatility → skip hedge ─────────────────────
print("\n" + "="*55)
print("TEST 2: Low Volatility Market")
print("="*55)

result2 = hedge_agent.invoke({
    "stock_price": 100.0,
    "strike_price": 100.0,
    "time_to_expiry": 0.8,
    "volatility": 0.10,        # LOW → skip hedge
    "risk_free_rate": 0.05,
    "market_data": None,
    "volatility_regime": None,
    "should_hedge": None,
    "rl_action": None,
    "hedge_recommendation": None,
    "risk_report": None,
    "error": None
})

print(f"\n✅ Volatility Regime: {result2['volatility_regime']}")
print(f"✅ Should Hedge: {result2['should_hedge']}")
print(f"\n📋 Risk Report:\n{result2['risk_report']}")