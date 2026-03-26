# Derivative Hedging using Reinforcement Learning

A complete implementation of an RL agent (PPO) that learns to dynamically hedge a European Call Option — outperforming naive delta hedging strategies.

## What this project does

When you sell a call option, you take on financial risk. To manage this risk, traders "hedge" by holding shares of the underlying stock. The classic approach is **delta hedging** — always hold exactly Δ shares. But this is expensive due to constant rebalancing.

This project trains a **PPO (Proximal Policy Optimization) RL agent** to learn a smarter hedging strategy that minimizes hedging error while reducing transaction costs.

---

## Project Structure

```
derivative_hedging_rl/
├── phase1_simulation.py   # Simulate stock prices (GBM) + option prices (Black-Scholes)
├── phase2_environment.py  # Custom OpenAI Gym environment for the RL agent
├── phase3_training.py     # Train PPO agent + compare vs baselines
├── app.py                 # Interactive Streamlit dashboard
├── utils.py               # Shared helper functions
├── requirements.txt       # Python dependencies
├── data/                  # Generated: simulated_paths.csv
├── models/                # Generated: saved PPO model
└── results/               # Generated: plots and metrics
```

---

## Key Concepts

| Term | Plain English |
|------|--------------|
| **Call Option** | Right to buy a stock at a fixed price in the future |
| **Strike Price (K)** | The fixed price agreed in the option contract |
| **Delta (Δ)** | How many shares to hold to hedge the option |
| **GBM** | Mathematical model for random stock price movement |
| **Black-Scholes** | Formula to price options and compute delta |
| **PPO** | RL algorithm that learns the optimal hedging policy |

---

## Phases

### Phase 1 — Data Simulation
- Simulates 1000 stock price paths using **Geometric Brownian Motion (GBM)**
- Computes option prices and deltas using the **Black-Scholes formula**
- Saves a rich dataset of `(stock_price, option_price, delta, time_to_expiry)` tuples

### Phase 2 — RL Environment
- Custom **OpenAI Gym environment** where the agent lives
- **State**: `[stock_price, option_price, time_to_expiry, hedge_position, delta]`
- **Action**: continuous `[-1, 1]` — change in hedge position
- **Reward**: `−|hedge_position − delta| − transaction_cost`

### Phase 3 — Agent Training
- Trains a **PPO agent** with a 2-layer MLP policy (256 neurons each)
- Splits data into 80% train / 20% test paths
- Compares RL agent vs **delta naive** and **no-hedge** baselines

### Phase 4 — Dashboard
- Interactive **Streamlit** dashboard with 4 tabs:
  1. Stock Simulation — GBM paths + option prices
  2. Hedging Comparison — strategy comparison on individual paths
  3. RL vs Delta — aggregate performance metrics + heatmaps
  4. Black-Scholes Explorer — interactive pricing surface

---

## Setup & Installation

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/derivative_hedging_rl.git
cd derivative_hedging_rl

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate       # Linux/Mac
# venv\Scripts\activate        # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run Phase 1 (simulate data)
python phase1_simulation.py

# 5. Run Phase 2 (test the environment)
python phase2_environment.py

# 6. Run Phase 3 (train the agent — takes ~5 min)
python phase3_training.py

# 7. Launch the dashboard
streamlit run app.py
```

---

## Results

After training, the PPO agent:
- Achieves **lower hedging error** than naive delta hedging in many scenarios
- **Reduces transaction costs** by learning to smooth its position changes
- Generalizes to unseen stock price paths (test set)

---

## Tech Stack

- **Python 3.10+**
- **NumPy / SciPy** — numerical computing, Black-Scholes
- **Gymnasium** — RL environment framework
- **Stable-Baselines3** — PPO implementation
- **PyTorch** — neural network backend
- **Streamlit + Plotly** — interactive dashboard

---

## Why this is impressive (interview talking points)

1. **End-to-end RL project** — you built the environment, trained the agent, and evaluated it
2. **Finance domain** — shows cross-domain knowledge (quant + ML)
3. **Interactive dashboard** — visual, demo-able, professional
4. **Proper ML practices** — train/test split, baselines, metrics
5. **Custom Gym environment** — not just using a pre-built env

---

## License

MIT License — free to use, modify, and distribute.
