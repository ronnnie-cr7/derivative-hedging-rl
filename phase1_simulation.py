"""
Phase 1: Data Simulation
========================
We simulate stock prices using Geometric Brownian Motion (GBM),
then price a Call Option on that stock using the Black-Scholes formula.

Think of it like this:
- GBM = how a stock price randomly moves each day
- Black-Scholes = a formula that tells us how much an option is worth
- Delta = how many shares we should hold to hedge the option

Output: A CSV file with columns:
    stock_price, option_price, delta, time_to_expiry, path_id, step
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
import os

# ─────────────────────────────────────────────
# PARAMETERS — you can tweak these!
# ─────────────────────────────────────────────
S0    = 100      # Starting stock price (like ₹100 per share)
K     = 100      # Strike price (the fixed price in the option contract)
T     = 1.0      # Time to expiry in years (1 year = 252 trading days)
r     = 0.05     # Risk-free rate (like a bank interest rate, 5%)
mu    = 0.08     # Expected return of the stock (8% per year)
sigma = 0.2      # Volatility — how wildly the stock moves (20% per year)
N     = 252      # Number of time steps (252 trading days in a year)
paths = 1000     # Number of simulated paths (1000 different "scenarios")
dt    = T / N    # Size of each time step (1/252 of a year)

np.random.seed(42)  # Makes results reproducible (same random numbers every run)


# ─────────────────────────────────────────────
# STEP 1: SIMULATE STOCK PRICES USING GBM
# ─────────────────────────────────────────────
def simulate_gbm(S0, mu, sigma, T, N, paths):
    """
    Geometric Brownian Motion (GBM):
    Simulates 'paths' number of possible stock price journeys over N days.

    Formula each step:
        S[t] = S[t-1] * exp( (mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z )

    Where:
        mu    = drift (average growth direction)
        sigma = volatility (how random/wild the moves are)
        Z     = random shock from a normal distribution
        dt    = size of each time step
    """
    dt = T / N
    prices = np.zeros((N, paths))   # Matrix: rows=time steps, cols=different paths
    prices[0] = S0                   # All paths start at S0

    for t in range(1, N):
        # Z is today's random surprise (could be +ve or -ve news)
        Z = np.random.standard_normal(paths)

        # GBM formula — the core of everything
        prices[t] = prices[t - 1] * np.exp(
            (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z
        )

    return prices  # Shape: (252, 1000) — 252 days × 1000 scenarios


# ─────────────────────────────────────────────
# STEP 2: BLACK-SCHOLES OPTION PRICING
# ─────────────────────────────────────────────
def black_scholes_call(S, K, T, r, sigma):
    """
    Black-Scholes formula for a European Call Option price.

    Inputs:
        S     = current stock price
        K     = strike price (fixed price in the contract)
        T     = time remaining until expiry (in years)
        r     = risk-free interest rate
        sigma = volatility of the stock

    Output:
        call_price = fair price of the option RIGHT NOW
    """
    # Handle edge case: if T is 0 (expired), option value = max(S-K, 0)
    if np.isscalar(T) and T <= 0:
        return np.maximum(S - K, 0)

    T = np.where(T <= 0, 1e-10, T)  # Avoid division by zero

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    # norm.cdf = cumulative normal distribution (probability between 0 and 1)
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price


def black_scholes_delta(S, K, T, r, sigma):
    """
    Delta = sensitivity of option price to stock price changes.
    This is what the naive hedger always holds.

    Delta is between 0 and 1:
        ~0.1 = low chance of exercise, hold few shares
        ~0.5 = 50-50 chance, hold half shares
        ~0.9 = very likely to exercise, hold almost 1 share
    """
    T = np.where(T <= 0, 1e-10, T)

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1)  # Delta = N(d1)


# ─────────────────────────────────────────────
# STEP 3: BUILD THE TRAINING DATASET
# ─────────────────────────────────────────────
def build_dataset(prices, K, T, r, sigma, N):
    """
    For each time step and each path, record:
        - stock_price       → what the stock is worth right now
        - time_to_expiry    → how much time is left (shrinks each day)
        - option_price      → what the option is worth (Black-Scholes)
        - delta             → how many shares to hold (naive strategy)
        - path_id           → which simulation path
        - step              → which day (0 to 251)
    """
    records = []
    dt = T / N

    for step in range(N):
        time_remaining = T - step * dt     # Time left until expiry
        S = prices[step]                   # Stock prices at this step (all paths)

        option = black_scholes_call(S, K, time_remaining, r, sigma)
        delta  = black_scholes_delta(S, K, time_remaining, r, sigma)

        for path_id in range(prices.shape[1]):
            records.append({
                "path_id":        path_id,
                "step":           step,
                "stock_price":    round(float(S[path_id]), 4),
                "time_to_expiry": round(float(time_remaining), 6),
                "option_price":   round(float(option[path_id]), 4),
                "delta":          round(float(delta[path_id]), 4),
            })

    df = pd.DataFrame(records)
    return df


# ─────────────────────────────────────────────
# STEP 4: PLOT SOME PATHS (so we can see what we made)
# ─────────────────────────────────────────────
def plot_sample_paths(prices, n_show=10, save_path="results/gbm_paths.png"):
    """
    Plot a few sample stock price paths so we can visually verify they look right.
    Real stock prices look similar — random walks with slight upward drift.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Phase 1: Simulated Stock & Option Prices", fontsize=14, fontweight='bold')

    # Plot stock paths
    for i in range(n_show):
        axes[0].plot(prices[:, i], alpha=0.6, linewidth=0.8)
    axes[0].axhline(y=K, color='red', linestyle='--', linewidth=1.5, label=f'Strike K={K}')
    axes[0].set_title("Simulated Stock Price Paths (GBM)")
    axes[0].set_xlabel("Trading Day")
    axes[0].set_ylabel("Stock Price (₹)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot option price paths
    time_steps = np.linspace(T, 0, N)
    for i in range(n_show):
        opt_path = black_scholes_call(prices[:, i], K, time_steps, r, sigma)
        axes[1].plot(opt_path, alpha=0.6, linewidth=0.8)
    axes[1].set_title("Corresponding Option Prices (Black-Scholes)")
    axes[1].set_xlabel("Trading Day")
    axes[1].set_ylabel("Option Price (₹)")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot saved → {save_path}")


# ─────────────────────────────────────────────
# MAIN — Run everything
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  PHASE 1: Simulating Stock & Option Prices")
    print("=" * 55)

    # 1. Simulate stock prices
    print(f"\n[1/4] Simulating {paths} GBM paths over {N} days...")
    prices = simulate_gbm(S0, mu, sigma, T, N, paths)
    print(f"      Price matrix shape: {prices.shape}  (days × paths)")
    print(f"      Final price range: ₹{prices[-1].min():.2f} – ₹{prices[-1].max():.2f}")

    # 2. Build dataset
    print(f"\n[2/4] Computing option prices and deltas...")
    df = build_dataset(prices, K, T, r, sigma, N)
    print(f"      Dataset shape: {df.shape}  ({df.shape[0]:,} rows)")
    print(f"      Columns: {list(df.columns)}")

    # 3. Save to CSV
    os.makedirs("data", exist_ok=True)
    csv_path = "data/simulated_paths.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[3/4] Saved dataset → {csv_path}")

    # 4. Plot
    print(f"\n[4/4] Generating plots...")
    plot_sample_paths(prices)

    # 5. Print a sample
    print(f"\n{'─'*55}")
    print("Sample rows from the dataset:")
    print(df[df['path_id'] == 0].head(5).to_string(index=False))

    print(f"\n{'─'*55}")
    print("Summary statistics:")
    print(df[['stock_price', 'option_price', 'delta', 'time_to_expiry']].describe().round(4))

    print(f"\n✓ Phase 1 complete! Dataset ready for the RL environment.")
