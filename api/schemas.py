from pydantic import BaseModel, Field
from typing import Literal

class MarketState(BaseModel):
    """
    The 5-dimensional state vector your HedgingEnv uses.
    Matches phase2_environment.py observation_space exactly.
    """
    stock_price: float = Field(..., gt=0, example=100.0,
        description="Current stock price (unnormalized)")
    option_price: float = Field(..., gt=0, example=10.5,
        description="Current option price (Black-Scholes)")
    time_to_expiry: float = Field(..., ge=0, le=1.0, example=0.5,
        description="Time to expiry normalized 0 to 1")
    current_hedge_position: float = Field(..., ge=-1.0, le=1.0, example=0.0,
        description="Current hedge position agent holds")
    delta: float = Field(..., ge=0.0, le=1.0, example=0.45,
        description="Black-Scholes delta (naive baseline)")

class HedgeAction(BaseModel):
    """What the PPO agent recommends."""
    recommended_hedge_position: float
    hedge_action_delta: float  # change in position
    hedging_error_vs_bs: float  # how far from naive delta
    strategy: Literal["increase", "decrease", "hold"]
    model_version: str = "ppo_hedging_agent_v1"

class SimulationRequest(BaseModel):
    """Run a full GBM simulation + hedging episode."""
    n_paths: int = Field(default=10, ge=1, le=200,
        description="Number of GBM paths to simulate")
    S0: float = Field(default=100.0, gt=0, description="Initial stock price")
    K: float = Field(default=100.0, gt=0, description="Strike price")
    T: float = Field(default=1.0, gt=0, description="Time to expiry in years")
    r: float = Field(default=0.05, description="Risk-free rate")
    mu: float = Field(default=0.08, description="Expected return of the stock")
    sigma: float = Field(default=0.2, gt=0, description="Volatility")
    n_steps: int = Field(default=252, description="Steps per path")

class SimulationResult(BaseModel):
    """Results of a full simulation run."""
    n_paths: int
    rl_mean_error: float
    delta_mean_error: float
    no_hedge_mean_error: float
    rl_mean_transaction_cost: float
    delta_mean_transaction_cost: float
    error_reduction_vs_no_hedge_pct: float
    paths_summary: list

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_path: str
    environment: str