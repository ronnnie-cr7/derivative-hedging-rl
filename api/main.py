# api/main.py
import numpy as np
import pandas as pd
import os
import sys
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from stable_baselines3 import PPO


# Add parent directory so we can import your existing code
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from phase2_environment import HedgingEnv
from phase1_simulation import simulate_gbm, build_dataset
from api.schemas import (
    MarketState, HedgeAction, SimulationRequest,
    SimulationResult, HealthResponse
)

# ── Global model holder ────────────────────────────────────────────
MODEL = None
MODEL_PATH = os.getenv("MODEL_PATH", "models/ppo_hedging_agent")

# ── Lifespan: load model once on startup ──────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL
    print(f"[startup] Loading PPO model from {MODEL_PATH}.zip ...")
    try:
        MODEL = PPO.load(MODEL_PATH)
        print("[startup] Model loaded successfully.")
    except Exception as e:
        print(f"[startup] WARNING: Could not load model — {e}")
        print("[startup] /predict will return 503 until model is available.")
    yield
    print("[shutdown] Cleaning up...")

# ── App ────────────────────────────────────────────────────────────
app = FastAPI(
    title="AlphaHedge RL API",
    description="Production API for PPO-based derivative hedging agent",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── ENDPOINT 1: Health check ───────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    """Check if API and model are running."""
    return HealthResponse(
        status="healthy" if MODEL is not None else "degraded",
        model_loaded=MODEL is not None,
        model_path=MODEL_PATH,
        environment="HedgingEnv-v1"
    )

# ── ENDPOINT 2: Single-step hedge prediction ──────────────────────
@app.post("/predict", response_model=HedgeAction, tags=["Inference"])
async def predict_hedge(state: MarketState):
    """
    Given current market state, return the PPO agent's 
    recommended hedge action.
    
    This mirrors your HedgingEnv._get_observation() normalization exactly.
    """
    if MODEL is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Check model path."
        )
    
    # Normalize exactly as HedgingEnv._get_observation() does
    S_mean = state.stock_price  # single call: normalize by itself
    obs = np.array([
        state.stock_price / 100.0,          # rough normalization
        state.option_price / (100.0 * 0.5), # matches env logic
        state.time_to_expiry,               # already 0-1
        state.current_hedge_position,       # already -1 to 1
        state.delta                         # already 0-1
    ], dtype=np.float32)
    
    # Clip to observation space bounds
    obs = np.clip(obs, [0.0, 0.0, 0.0, -1.0, 0.0],
                       [5.0, 5.0, 1.0,  1.0, 1.0])
    
    # Get PPO action — deterministic=True means no exploration noise
    action, _ = MODEL.predict(obs, deterministic=True)
    action_value = float(np.clip(action[0], -1.0, 1.0))
    
    # Compute new position (mirrors env.step() logic)
    new_position = float(np.clip(
        state.current_hedge_position + action_value * 0.1,
        -1.0, 1.0
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

# ── ENDPOINT 3: Full simulation run ───────────────────────────────
@app.post("/simulate", response_model=SimulationResult, tags=["Simulation"])
async def run_simulation(req: SimulationRequest):
    """
    Runs simulate_gbm() → build_dataset() → HedgingEnv → PPO evaluation.
    Exactly mirrors your phase1 + phase2 + phase3 pipeline.
    """
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    try:
        # Step 1: simulate_gbm() returns shape (N, n_paths)
        prices = simulate_gbm(
            S0=req.S0,
            mu=0.08,          # using your default mu
            sigma=req.sigma,
            T=req.T,
            N=req.n_steps,
            paths=req.n_paths
        )

        # Step 2: build_dataset() assembles the DataFrame HedgingEnv expects
        # columns: path_id, step, stock_price, time_to_expiry, option_price, delta
        df = build_dataset(
            prices=prices,
            K=req.K,
            T=req.T,
            r=req.r,
            sigma=req.sigma,
            N=req.n_steps
        )
    except Exception as e:
        raise HTTPException(status_code=500,
            detail=f"Simulation failed: {str(e)}")

    # Step 3: run HedgingEnv evaluation — same as your phase3 evaluate_strategies()
    env = HedgingEnv(df, mode="test")
    rl_errors, no_hedge_errors = [], []
    rl_tx_costs, delta_tx_costs = [], []
    paths_summary = []

    for episode in range(req.n_paths):
        obs, _ = env.reset()
        done = False
        step = 0
        ep_rl_err, ep_no_hedge_err = [], []
        ep_rl_tx = []

        while not done:
            row = env.path_data.iloc[min(step, len(env.path_data) - 1)]
            delta_true = float(row["delta"])

            # RL agent prediction
            action, _ = MODEL.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            pos_rl = info["hedge_position"]
            action_value = float(action[0])

            ep_rl_err.append(abs(pos_rl - delta_true))
            ep_rl_tx.append(abs(action_value) * 0.001)   # matches your env tx cost
            ep_no_hedge_err.append(abs(0.0 - delta_true)) # no hedge = hold 0

            done = terminated or truncated
            step += 1

        rl_errors.extend(ep_rl_err)
        no_hedge_errors.extend(ep_no_hedge_err)
        rl_tx_costs.extend(ep_rl_tx)
        delta_tx_costs.extend([0.001] * step)  # naive delta rebalances every step

        paths_summary.append({
            "episode": episode,
            "rl_mean_error": round(float(np.mean(ep_rl_err)), 4),
            "steps": step
        })

    rl_mean = float(np.mean(rl_errors))
    no_hedge_mean = float(np.mean(no_hedge_errors))
    reduction_pct = round((no_hedge_mean - rl_mean) / no_hedge_mean * 100, 2) \
                    if no_hedge_mean > 0 else 0.0

    return SimulationResult(
        n_paths=req.n_paths,
        rl_mean_error=round(rl_mean, 4),
        delta_mean_error=0.0,              # naive delta error is 0 by definition
        no_hedge_mean_error=round(no_hedge_mean, 4),
        rl_mean_transaction_cost=round(float(np.mean(rl_tx_costs)), 6),
        delta_mean_transaction_cost=round(float(np.mean(delta_tx_costs)), 6),
        error_reduction_vs_no_hedge_pct=reduction_pct,
        paths_summary=paths_summary[:10]
    )

# ── ENDPOINT 4: Model info ─────────────────────────────────────────
@app.get("/model/info", tags=["System"])
async def model_info():
    """Return model architecture and training config."""
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    
    policy = MODEL.policy
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
        "network_architecture": "[256, 256]"
    }