"""
Phase 2: RL Environment (Custom OpenAI Gym)
===========================================
This is the "world" in which our RL agent lives and learns.

Think of it like a video game:
    - STATE  = what the agent sees (stock price, option price, time left, position)
    - ACTION = what the agent does (how many shares to buy/sell)
    - REWARD = points the agent gets (we reward low hedging error)

The agent plays this "game" thousands of times and learns to hedge better.

Key Gym methods:
    reset()  → start a new episode (new stock path)
    step()   → take one action, get next state + reward
    render() → (optional) visualize what's happening
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces


class HedgingEnv(gym.Env):
    """
    Custom Gym environment for Options Delta Hedging.

    The agent manages a portfolio that hedges a short call option.
    Each episode = one simulated stock price path (252 steps).

    ANALOGY:
    Imagine you're a trader who sold an IPL ticket option.
    Each day you decide how many tickets to hold to cover yourself.
    The environment tells you the market conditions, you decide your position,
    and you get penalized for any mismatch (hedging error).
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, data: pd.DataFrame, mode: str = "train"):
        """
        Args:
            data : DataFrame with columns [path_id, step, stock_price,
                   option_price, delta, time_to_expiry]
            mode : 'train' or 'test'
        """
        super().__init__()
        self.data = data
        self.mode = mode

        # Get unique path ids for sampling episodes
        self.path_ids = data["path_id"].unique()
        self.n_steps  = data["step"].nunique()   # 252 steps per episode

        # ── ACTION SPACE ──────────────────────────────────────────────────
        # Continuous: agent outputs a single number in [-1, 1]
        #   -1 = sell 1 share (max sell)
        #   +1 = buy 1 share  (max buy)
        #   0  = hold current position
        # This is a CONTINUOUS action space — more realistic than discrete
        self.action_space = spaces.Box(
            low   = np.array([-1.0], dtype=np.float32),
            high  = np.array([ 1.0], dtype=np.float32),
            dtype = np.float32
        )

        # ── OBSERVATION (STATE) SPACE ─────────────────────────────────────
        # What the agent observes each step:
        #   [0] stock_price_normalized    (0 to ~5)
        #   [1] option_price_normalized   (0 to ~5)
        #   [2] time_to_expiry            (1.0 → 0.0 over the episode)
        #   [3] current_hedge_position    (-1.0 to 1.0)
        #   [4] delta (naive baseline)    (0.0 to 1.0)
        self.observation_space = spaces.Box(
            low   = np.array([0.0, 0.0, 0.0, -1.0, 0.0], dtype=np.float32),
            high  = np.array([5.0, 5.0, 1.0,  1.0, 1.0], dtype=np.float32),
            dtype = np.float32
        )

        # Internal state
        self.current_path    = None
        self.current_step    = 0
        self.hedge_position  = 0.0   # How many shares we currently hold
        self.path_data       = None  # DataFrame rows for current episode
        self.prev_portfolio  = 0.0   # Track portfolio value for P&L reward

        # Normalization constants (computed from data)
        self.S_mean  = data["stock_price"].mean()
        self.S_std   = data["stock_price"].std()
        self.opt_std = data["option_price"].std()

    # ─────────────────────────────────────────────────────────
    # RESET — called at the start of each new episode
    # ─────────────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Pick a random path for this episode
        self.current_path   = self.np_random.choice(self.path_ids)
        self.path_data      = (
            self.data[self.data["path_id"] == self.current_path]
            .sort_values("step")
            .reset_index(drop=True)
        )
        self.current_step   = 0
        self.hedge_position = 0.0
        self.prev_portfolio = 0.0

        obs  = self._get_observation()
        info = {}
        return obs, info

    # ─────────────────────────────────────────────────────────
    # STEP — called each time the agent takes an action
    # ─────────────────────────────────────────────────────────
    def step(self, action):
        """
        Execute one time step.

        Args:
            action: numpy array of shape (1,), value in [-1, 1]
                    Represents the CHANGE in hedge position

        Returns:
            obs      : next state observation
            reward   : how good this action was
            terminated: True if episode ended naturally (expiry)
            truncated : True if episode cut short (shouldn't happen)
            info     : dict with debug info
        """
        # Clip action to valid range
        action_value = float(np.clip(action[0], -1.0, 1.0))

        # ── UPDATE HEDGE POSITION ──────────────────────────────────────────
        # The agent's action is the CHANGE in position (not absolute)
        # We constrain total position between -1 and 1
        self.hedge_position = np.clip(
            self.hedge_position + action_value * 0.1,  # small step size
            -1.0, 1.0
        )

        # ── GET CURRENT MARKET DATA ────────────────────────────────────────
        row = self.path_data.iloc[self.current_step]
        S           = row["stock_price"]
        opt_price   = row["option_price"]
        delta_naive = row["delta"]
        t           = row["time_to_expiry"]

        # ── REWARD FUNCTION ───────────────────────────────────────────────
        # The reward punishes the agent for NOT being well-hedged.
        # Perfect hedge: hedge_position ≈ delta (hold exactly as many shares as delta says)
        # We also add a small transaction cost to discourage excessive trading.

        hedging_error      = abs(self.hedge_position - delta_naive)
        transaction_cost   = abs(action_value) * 0.001   # 0.1% transaction cost
        reward             = -(hedging_error + transaction_cost)

        # ── ADVANCE STEP ───────────────────────────────────────────────────
        self.current_step += 1
        terminated = (self.current_step >= len(self.path_data) - 1)

        # ── NEXT OBSERVATION ───────────────────────────────────────────────
        if not terminated:
            obs = self._get_observation()
        else:
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)

        info = {
            "stock_price":    S,
            "option_price":   opt_price,
            "delta_naive":    delta_naive,
            "hedge_position": self.hedge_position,
            "hedging_error":  hedging_error,
        }

        return obs, reward, terminated, False, info

    # ─────────────────────────────────────────────────────────
    # HELPER: Build observation vector
    # ─────────────────────────────────────────────────────────
    def _get_observation(self):
        """
        Build the 5-dimensional state vector the agent sees.
        We normalize values so they're all roughly in [0, 1] range —
        this helps the neural network learn faster.
        """
        row = self.path_data.iloc[self.current_step]

        stock_norm  = row["stock_price"]  / self.S_mean   # Normalize by mean
        option_norm = row["option_price"] / (self.S_mean * 0.5)  # Rough scale
        time_left   = row["time_to_expiry"]                # Already 0–1
        hedge_pos   = self.hedge_position                  # -1 to 1
        delta       = row["delta"]                         # 0 to 1

        obs = np.array(
            [stock_norm, option_norm, time_left, hedge_pos, delta],
            dtype=np.float32
        )
        return np.clip(obs, self.observation_space.low, self.observation_space.high)

    # ─────────────────────────────────────────────────────────
    # RENDER — optional visualization
    # ─────────────────────────────────────────────────────────
    def render(self):
        row = self.path_data.iloc[self.current_step]
        print(
            f"Step {self.current_step:3d} | "
            f"S={row['stock_price']:7.2f} | "
            f"Opt={row['option_price']:6.2f} | "
            f"Delta={row['delta']:.3f} | "
            f"Position={self.hedge_position:.3f} | "
            f"Error={abs(self.hedge_position - row['delta']):.3f}"
        )


# ─────────────────────────────────────────────
# QUICK TEST — verify the environment works
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import os

    print("=" * 55)
    print("  PHASE 2: Testing RL Environment")
    print("=" * 55)

    # Load data from Phase 1
    csv_path = "data/simulated_paths.csv"
    if not os.path.exists(csv_path):
        print(f"\n[!] Run phase1_simulation.py first to generate data.")
        exit(1)

    df = pd.read_csv(csv_path)
    print(f"\n[1/3] Loaded dataset: {df.shape[0]:,} rows")

    # Create environment
    env = HedgingEnv(df, mode="train")
    print(f"[2/3] Environment created!")
    print(f"      Action space:      {env.action_space}")
    print(f"      Observation space: {env.observation_space}")

    # Run one episode with random actions (no learning yet)
    print(f"\n[3/3] Running 1 episode with random actions...")
    obs, _ = env.reset()
    total_reward = 0
    step = 0

    print(f"\n{'Step':>4} | {'Stock':>7} | {'Option':>7} | {'Delta':>6} | {'Position':>8} | {'Reward':>8}")
    print("─" * 58)

    while True:
        action = env.action_space.sample()   # Random action (no learning)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step += 1

        if step % 50 == 0:   # Print every 50 steps
            print(
                f"{step:4d} | "
                f"{info['stock_price']:7.2f} | "
                f"{info['option_price']:7.2f} | "
                f"{info['delta_naive']:6.3f} | "
                f"{info['hedge_position']:8.3f} | "
                f"{reward:8.4f}"
            )

        if terminated or truncated:
            break

    print(f"\nEpisode finished after {step} steps")
    print(f"Total reward (random agent): {total_reward:.4f}")
    print(f"\n✓ Phase 2 complete! Environment is ready for training.")
