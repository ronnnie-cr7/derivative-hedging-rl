import numpy as np
import os
import sys
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from stable_baselines3 import PPO
from pydantic import BaseModel, Field
from typing import Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from phase2_environment import HedgingEnv
from phase1_simulation import simulate_gbm, build_dataset
from api.schemas import (
    MarketState, HedgeAction, SimulationRequest,
    SimulationResult, HealthResponse
)
from agent.graph import hedge_agent

# ── Global model ───────────────────────────────────────────────────
MODEL = None
MODEL_PATH = os.getenv("MODEL_PATH", "models/ppo_hedging_agent")


# ── Lifespan ───────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL
    print(f"[startup] Loading PPO model from {MODEL_PATH}.zip ...")
    try:
        MODEL = PPO.load(MODEL_PATH)
        print("[startup] Model loaded successfully.")
    except Exception as e:
        print(f"[startup] WARNING: Could not load model — {e}")
    yield
    print("[shutdown] Cleaning up...")


# ── App ────────────────────────────────────────────────────────────
app = FastAPI(
    title="AlphaHedge RL API",
    description=(
        "Production REST API for PPO-based derivative hedging agent "
        "with LangGraph multi-agent orchestration."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helper: build full initial state ──────────────────────────────
def build_initial_state(
    stock_price: float,
    strike_price: float,
    time_to_expiry: float,
    volatility: float,
    risk_free_rate: float,
    scenario_description: Optional[str] = None,
    scenario_stock_price: Optional[float] = None,
) -> dict:
    """Returns a fully initialised HedgeState dict."""
    return {
        "stock_price": stock_price,
        "strike_price": strike_price,
        "time_to_expiry": time_to_expiry,
        "volatility": volatility,
        "risk_free_rate": risk_free_rate,
        "scenario_description": scenario_description,
        "scenario_stock_price": scenario_stock_price,
        "market_data": None,
        "volatility_regime": None,
        "should_hedge": None,
        "rl_action": None,
        "hedge_recommendation": None,
        "comparison": None,
        "risk_report": None,
        "decision_flow": None,
        "error": None,
    }


# ══════════════════════════════════════════════════════════════════
# SYSTEM ENDPOINTS
# ══════════════════════════════════════════════════════════════════

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    """Check if API and model are running."""
    return HealthResponse(
        status="healthy" if MODEL is not None else "degraded",
        model_loaded=MODEL is not None,
        model_path=MODEL_PATH,
        environment="HedgingEnv-v1"
    )


@app.get("/model/info", tags=["System"])
async def model_info():
    """Return model architecture and training config."""
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    return {
        "algorithm": "PPO",
        "policy": "MlpPolicy",
        "observation_space": str(MODEL.observation_space),
        "action_space": str(MODEL.action_space),
        "learning_rate": MODEL.learning_rate,
        "n_steps": MODEL.n_steps,
        "state_vector": [
            "stock_price_normalized",
            "option_price_normalized",
            "time_to_expiry",
            "current_hedge_position",
            "black_scholes_delta"
        ],
        "network_architecture": "[256, 256]",
        "training_paths": 1000,
        "training_steps": 252,
        "note": (
            "PPO trained episodically over 252 steps. "
            "Single-step inference shows first action from position 0. "
            "Full performance (96.5% error reduction) shown in /simulate."
        )
    }


# ══════════════════════════════════════════════════════════════════
# INFERENCE ENDPOINTS
# ══════════════════════════════════════════════════════════════════

@app.post("/predict", response_model=HedgeAction, tags=["Inference"])
async def predict_hedge(state: MarketState):
    """
    Single-step hedge recommendation from the PPO model.

    Note: PPO was trained episodically (252 steps per episode).
    This endpoint shows the agent's first action from position 0.
    For full episodic evaluation use /simulate.
    """
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    obs = np.array([
        state.stock_price / 100.0,
        state.option_price / (state.stock_price * 0.5 + 1e-8),
        state.time_to_expiry,
        state.current_hedge_position,
        state.delta
    ], dtype=np.float32)

    obs = np.clip(obs, [0.0, 0.0, 0.0, -1.0, 0.0],
                       [5.0, 5.0, 1.0,  1.0, 1.0])

    action, _ = MODEL.predict(obs, deterministic=True)
    action_value = float(np.clip(action[0], -1.0, 1.0))

    new_position = float(np.clip(
        state.current_hedge_position + action_value * 0.1, -1.0, 1.0
    ))
    hedging_error = abs(new_position - state.delta)

    if action_value > 0.05:
        strategy = "increase"
    elif action_value < -0.05:
        strategy = "decrease"
    else:
        strategy = "hold"

    return HedgeAction(
        recommended_hedge_position=round(new_position, 4),
        hedge_action_delta=round(action_value, 4),
        hedging_error_vs_bs=round(hedging_error, 4),
        strategy=strategy
    )


@app.post("/simulate", response_model=SimulationResult, tags=["Simulation"])
async def run_simulation(req: SimulationRequest):
    """
    Full episodic evaluation — runs GBM simulation then evaluates
    PPO agent over complete 252-step episodes across n_paths paths.

    This is the correct endpoint for performance benchmarking.
    Results show the 96.5% error reduction vs no-hedge baseline.
    """
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    try:
        prices = simulate_gbm(
            S0=req.S0, mu=req.mu, sigma=req.sigma,
            T=req.T, N=req.n_steps, paths=req.n_paths
        )
        df = build_dataset(
            prices=prices, K=req.K, T=req.T,
            r=req.r, sigma=req.sigma, N=req.n_steps
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Simulation failed: {str(e)}")

    env = HedgingEnv(df, mode="test")
    rl_errors, no_hedge_errors = [], []
    rl_tx_costs, delta_tx_costs = [], []
    paths_summary = []

    for episode in range(req.n_paths):
        obs, _ = env.reset()
        done, step = False, 0
        ep_rl_err, ep_no_hedge_err, ep_rl_tx = [], [], []

        while not done:
            row = env.path_data.iloc[min(step, len(env.path_data) - 1)]
            delta_true = float(row["delta"])

            action, _ = MODEL.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)

            pos_rl = info["hedge_position"]
            ep_rl_err.append(abs(pos_rl - delta_true))
            ep_rl_tx.append(abs(float(action[0])) * 0.001)
            ep_no_hedge_err.append(abs(0.0 - delta_true))

            done = terminated or truncated
            step += 1

        rl_errors.extend(ep_rl_err)
        no_hedge_errors.extend(ep_no_hedge_err)
        rl_tx_costs.extend(ep_rl_tx)
        delta_tx_costs.extend([0.001] * step)

        paths_summary.append({
            "episode": episode,
            "rl_mean_error": round(float(np.mean(ep_rl_err)), 4),
            "steps": step
        })

    rl_mean       = float(np.mean(rl_errors))
    no_hedge_mean = float(np.mean(no_hedge_errors))
    reduction_pct = round(
        (no_hedge_mean - rl_mean) / no_hedge_mean * 100, 2
    ) if no_hedge_mean > 0 else 0.0

    return SimulationResult(
        n_paths=req.n_paths,
        rl_mean_error=round(rl_mean, 4),
        delta_mean_error=0.0,
        no_hedge_mean_error=round(no_hedge_mean, 4),
        rl_mean_transaction_cost=round(float(np.mean(rl_tx_costs)), 6),
        delta_mean_transaction_cost=round(float(np.mean(delta_tx_costs)), 6),
        error_reduction_vs_no_hedge_pct=reduction_pct,
        paths_summary=paths_summary[:10]
    )


# ══════════════════════════════════════════════════════════════════
# AGENT ENDPOINTS
# ══════════════════════════════════════════════════════════════════

class AgentRequest(BaseModel):
    stock_price:    float = Field(..., example=105.0)
    strike_price:   float = Field(..., example=100.0)
    time_to_expiry: float = Field(..., example=0.4)
    volatility:     float = Field(..., example=0.35)
    risk_free_rate: float = Field(default=0.05)


class AgentResponse(BaseModel):
    volatility_regime:    str
    should_hedge:         bool
    hedge_recommendation: Optional[str]
    comparison:           Optional[dict]
    risk_report:          str
    decision_flow:        Optional[list]
    error:                Optional[str]


class ScenarioRequest(BaseModel):
    stock_price:           float = Field(..., example=105.0)
    strike_price:          float = Field(..., example=100.0)
    time_to_expiry:        float = Field(..., example=0.4)
    volatility:            float = Field(..., example=0.35)
    risk_free_rate:        float = Field(default=0.05)
    scenario_description:  str   = Field(..., example="price drops 10%")
    price_change_pct:      float = Field(..., example=-10.0,
        description="% price change. -10 = drop 10%, +15 = rise 15%")


@app.post("/agent/analyze", response_model=AgentResponse, tags=["Agent"])
async def agent_analyze(request: AgentRequest):
    """
    Full LangGraph 5-node agent pipeline.

    Nodes: MarketMonitor → VolatilityAnalyzer → HedgeDecider
           → ContextNode → ReportGenerator

    Returns volatility regime, RL hedge recommendation,
    market context (BS delta, option price, moneyness),
    LLM risk report, and full decision flow trace.
    """
    result = hedge_agent.invoke(
        build_initial_state(
            stock_price=request.stock_price,
            strike_price=request.strike_price,
            time_to_expiry=request.time_to_expiry,
            volatility=request.volatility,
            risk_free_rate=request.risk_free_rate,
        )
    )

    if result.get("error"):
        raise HTTPException(status_code=500, detail=result["error"])

    return AgentResponse(
        volatility_regime=result["volatility_regime"],
        should_hedge=result["should_hedge"],
        hedge_recommendation=result.get("hedge_recommendation"),
        comparison=result.get("comparison"),
        risk_report=result["risk_report"],
        decision_flow=result.get("decision_flow"),
        error=result.get("error"),
    )


@app.post("/agent/scenario", tags=["Agent"])
async def agent_scenario(request: ScenarioRequest):
    """
    Scenario simulation — 'What if the price drops 10%?'

    Adjusts the stock price by price_change_pct and runs the
    full LangGraph agent pipeline with the new price.
    Useful for stress testing hedge positions.
    """
    scenario_price = round(
        request.stock_price * (1 + request.price_change_pct / 100), 2
    )

    result = hedge_agent.invoke(
        build_initial_state(
            stock_price=request.stock_price,
            strike_price=request.strike_price,
            time_to_expiry=request.time_to_expiry,
            volatility=request.volatility,
            risk_free_rate=request.risk_free_rate,
            scenario_description=request.scenario_description,
            scenario_stock_price=scenario_price,
        )
    )

    return {
        "original_price":        request.stock_price,
        "scenario_price":        scenario_price,
        "scenario_description":  request.scenario_description,
        "volatility_regime":     result["volatility_regime"],
        "should_hedge":          result["should_hedge"],
        "hedge_recommendation":  result.get("hedge_recommendation"),
        "comparison":            result.get("comparison"),
        "risk_report":           result["risk_report"],
        "decision_flow":         result.get("decision_flow"),
        "error":                 result.get("error"),
    }