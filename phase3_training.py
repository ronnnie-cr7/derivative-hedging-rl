"""
Phase 3: RL Agent Training
==========================
We train a PPO (Proximal Policy Optimization) agent to hedge the option.

PPO is like teaching someone to hedge by letting them practice thousands of times.
After each attempt, it gets feedback (reward) and gradually improves its strategy.

After training, we compare:
    RL Agent    → learned strategy (what we built)
    Naive Delta → always hold exactly delta shares (the baseline)
    No Hedge    → hold 0 shares (worst case)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import json
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor

from phase2_environment import HedgingEnv


# ─────────────────────────────────────────────
# CUSTOM CALLBACK — prints progress during training
# ─────────────────────────────────────────────
class TrainingProgressCallback(BaseCallback):
    """
    Callback that prints training progress every N steps.
    Helps us see if the agent is actually improving.
    """
    def __init__(self, print_freq=10_000, verbose=0):
        super().__init__(verbose)
        self.print_freq   = print_freq
        self.episode_rewards = []
        self.episode_buffer  = 0

    def _on_step(self):
        # Accumulate rewards
        if self.locals.get("rewards") is not None:
            self.episode_buffer += float(np.mean(self.locals["rewards"]))

        if self.n_calls % self.print_freq == 0:
            avg = self.episode_buffer / self.print_freq if self.print_freq > 0 else 0
            print(f"  Step {self.n_calls:>8,} | Avg reward/step: {avg:.5f}")
            self.episode_rewards.append(avg)
            self.episode_buffer = 0
        return True


# ─────────────────────────────────────────────
# STEP 1: LOAD DATA AND SPLIT TRAIN/TEST
# ─────────────────────────────────────────────
def load_and_split(csv_path="data/simulated_paths.csv", test_frac=0.2):
    """
    Load the simulated dataset and split into train/test paths.

    We split by path_id — entire price paths go to either train or test.
    This prevents data leakage (test paths are completely unseen).
    """
    df = pd.read_csv(csv_path)

    all_paths  = df["path_id"].unique()
    n_test     = int(len(all_paths) * test_frac)
    test_paths = np.random.choice(all_paths, size=n_test, replace=False)
    train_paths = np.setdiff1d(all_paths, test_paths)

    train_df = df[df["path_id"].isin(train_paths)].reset_index(drop=True)
    test_df  = df[df["path_id"].isin(test_paths)].reset_index(drop=True)

    print(f"  Train paths: {len(train_paths)} | Test paths: {len(test_paths)}")
    print(f"  Train rows:  {len(train_df):,}   | Test rows:  {len(test_df):,}")
    return train_df, test_df


# ─────────────────────────────────────────────
# STEP 2: TRAIN THE PPO AGENT
# ─────────────────────────────────────────────
def train_agent(train_df, total_timesteps=200_000, n_envs=4):
    """
    Train a PPO agent on the hedging environment.

    PPO = Proximal Policy Optimization
    - It's a neural network that takes the state and outputs an action
    - It trains by collecting experience, then updating the policy
    - 'Proximal' means it doesn't make too-large updates (stable training)

    Args:
        train_df        : training data
        total_timesteps : how many steps to train for (more = better, slower)
        n_envs          : parallel environments (speeds up data collection)
    """
    # Create multiple parallel environments (faster data collection)
    def make_env():
        env = HedgingEnv(train_df, mode="train")
        return Monitor(env)

    vec_env = make_vec_env(make_env, n_envs=n_envs)

    # PPO model with MLP policy (multi-layer perceptron)
    # net_arch = [256, 256] means 2 hidden layers of 256 neurons each
    model = PPO(
        policy          = "MlpPolicy",
        env             = vec_env,
        learning_rate   = 3e-4,          # How fast to learn (Adam optimizer)
        n_steps         = 1024,          # Steps to collect before each update
        batch_size      = 64,            # Mini-batch size for gradient updates
        n_epochs        = 10,            # Times to reuse each batch of data
        gamma           = 0.99,          # Discount factor (future rewards matter)
        clip_range      = 0.2,           # PPO clipping (keeps updates small)
        ent_coef        = 0.01,          # Entropy bonus (encourages exploration)
        policy_kwargs   = dict(
            net_arch=[256, 256]          # 2-layer neural network
        ),
        verbose         = 0,
    )

    callback = TrainingProgressCallback(print_freq=20_000)
    model.learn(total_timesteps=total_timesteps, callback=callback)

    # Save the trained model
    os.makedirs("models", exist_ok=True)
    model.save("models/ppo_hedging_agent")
    print(f"\n  Model saved → models/ppo_hedging_agent.zip")

    return model


# ─────────────────────────────────────────────
# STEP 3: EVALUATE & COMPARE STRATEGIES
# ─────────────────────────────────────────────
def evaluate_strategies(model, test_df, n_episodes=50):
    """
    Compare 3 strategies on test data:
        1. RL Agent      — our trained PPO model
        2. Delta Hedge   — always hold exactly delta shares (naive)
        3. No Hedge      — hold 0 shares (worst case baseline)

    Metrics:
        - Mean hedging error    (lower is better)
        - Std of hedging error  (lower = more consistent)
        - Cumulative P&L        (how much money made/lost)
    """
    env = HedgingEnv(test_df, mode="test")

    results = {
        "rl_agent":    {"errors": [], "rewards": [], "positions": [], "prices": []},
        "delta_naive": {"errors": [], "rewards": [], "positions": [], "prices": []},
        "no_hedge":    {"errors": [], "rewards": [], "positions": [], "prices": []},
    }

    for episode in range(n_episodes):
        obs, _ = env.reset()
        done   = False
        step   = 0

        ep_data = {k: {"errors": [], "positions": [], "prices": []} for k in results}

        while not done:
            row = env.path_data.iloc[min(step, len(env.path_data) - 1)]
            delta_true = float(row["delta"])
            stock      = float(row["stock_price"])

            # ── RL Agent action ──────────────────────────────────────────
            action_rl, _ = model.predict(obs, deterministic=True)
            obs_new, reward_rl, terminated, truncated, info_rl = env.step(action_rl)
            pos_rl  = info_rl["hedge_position"]
            err_rl  = abs(pos_rl - delta_true)

            # ── Delta Naive (always hold delta shares) ───────────────────
            pos_naive = delta_true
            err_naive = abs(pos_naive - delta_true)    # Always 0 by definition
            # But naive has higher transaction costs (rebalances every step)
            # We measure from the perspective of hedging error only
            reward_naive = -err_naive - 0.001           # small tx cost

            # ── No Hedge (hold 0) ─────────────────────────────────────────
            pos_none = 0.0
            err_none = abs(0.0 - delta_true)
            reward_none = -err_none

            # Record
            for key, pos, err, rew in [
                ("rl_agent",    pos_rl,    err_rl,    reward_rl),
                ("delta_naive", pos_naive, err_naive, reward_naive),
                ("no_hedge",    pos_none,  err_none,  reward_none),
            ]:
                results[key]["errors"].append(err)
                results[key]["rewards"].append(rew)
                results[key]["positions"].append(pos)
                results[key]["prices"].append(stock)

            obs  = obs_new
            done = terminated or truncated
            step += 1

    # Compute summary stats
    summary = {}
    for key in results:
        errors  = results[key]["errors"]
        rewards = results[key]["rewards"]
        summary[key] = {
            "mean_error":  np.mean(errors),
            "std_error":   np.std(errors),
            "total_reward": np.sum(rewards) / n_episodes,
        }

    return results, summary


# ─────────────────────────────────────────────
# STEP 4: PLOT RESULTS
# ─────────────────────────────────────────────
def plot_results(results, summary, save_dir="results"):
    os.makedirs(save_dir, exist_ok=True)

    colors = {
        "rl_agent":    "#4C72B0",
        "delta_naive": "#DD8452",
        "no_hedge":    "#55A868",
    }
    labels = {
        "rl_agent":    "RL Agent (PPO)",
        "delta_naive": "Delta Hedge (Naive)",
        "no_hedge":    "No Hedge",
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Phase 3: RL Agent vs Baselines", fontsize=14, fontweight='bold')

    # ── Plot 1: Hedging Error Distribution ───────────────────────────────
    ax = axes[0, 0]
    for key in results:
        ax.hist(results[key]["errors"], bins=50, alpha=0.6,
                color=colors[key], label=labels[key])
    ax.set_title("Hedging Error Distribution")
    ax.set_xlabel("Absolute Hedging Error")
    ax.set_ylabel("Frequency")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── Plot 2: Mean Error Bar Chart ──────────────────────────────────────
    ax = axes[0, 1]
    keys  = list(summary.keys())
    means = [summary[k]["mean_error"] for k in keys]
    stds  = [summary[k]["std_error"] for k in keys]
    bars  = ax.bar([labels[k] for k in keys], means, yerr=stds,
                   color=[colors[k] for k in keys], alpha=0.8, capsize=5)
    ax.set_title("Mean Hedging Error (lower = better)")
    ax.set_ylabel("Mean |hedge - delta|")
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f"{mean:.4f}", ha='center', va='bottom', fontsize=10)

    # ── Plot 3: Cumulative Reward ─────────────────────────────────────────
    ax = axes[1, 0]
    for key in results:
        rewards = results[key]["rewards"]
        cumulative = np.cumsum(rewards)
        ax.plot(cumulative, color=colors[key], label=labels[key], alpha=0.8, linewidth=1)
    ax.set_title("Cumulative Reward Over Time")
    ax.set_xlabel("Step")
    ax.set_ylabel("Cumulative Reward")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── Plot 4: Hedge Position vs Delta (first 252 steps) ────────────────
    ax = axes[1, 1]
    n_show = 252
    steps  = range(n_show)
    ax.plot(results["delta_naive"]["positions"][:n_show],
            color=colors["delta_naive"], label="True Delta", linewidth=1.5, linestyle='--')
    ax.plot(results["rl_agent"]["positions"][:n_show],
            color=colors["rl_agent"], label="RL Agent Position", linewidth=1, alpha=0.8)
    ax.set_title("RL Position vs Delta (1 episode)")
    ax.set_xlabel("Step")
    ax.set_ylabel("Hedge Position (shares)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/comparison_results.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Results plot saved → {save_dir}/comparison_results.png")

    return fig


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    np.random.seed(42)

    print("=" * 55)
    print("  PHASE 3: Training RL Hedging Agent")
    print("=" * 55)

    # 1. Load data
    print("\n[1/4] Loading and splitting data...")
    train_df, test_df = load_and_split()

    # 2. Train
    print("\n[2/4] Training PPO agent (this takes a few minutes)...")
    print("      Progress every 20,000 steps:")
    model = train_agent(train_df, total_timesteps=200_000, n_envs=4)

    # 3. Evaluate
    print("\n[3/4] Evaluating strategies on test data (50 episodes)...")
    results, summary = evaluate_strategies(model, test_df, n_episodes=50)

    # 4. Print summary table
    print("\n" + "─" * 55)
    print(f"{'Strategy':<22} {'Mean Error':>12} {'Std Error':>10} {'Avg Reward':>12}")
    print("─" * 55)
    for key, stats in summary.items():
        label = {"rl_agent": "RL Agent (PPO)", "delta_naive": "Delta Naive", "no_hedge": "No Hedge"}[key]
        print(f"{label:<22} {stats['mean_error']:>12.4f} {stats['std_error']:>10.4f} {stats['total_reward']:>12.4f}")
    print("─" * 55)

    # 5. Save summary
    os.makedirs("results", exist_ok=True)
    with open("results/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\n  Summary saved → results/summary.json")

    # 6. Plot
    print("\n[4/4] Plotting results...")
    plot_results(results, summary)

    print("\n✓ Phase 3 complete! Agent trained and evaluated.")
    print("  Run phase4_dashboard → app.py next for the dashboard.")
