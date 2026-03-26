# 📈 Reinforcement Learning for Derivative Hedging

This project explores how Reinforcement Learning can be used to solve a real-world quantitative finance problem — **dynamic hedging of a European Call Option**.

Instead of relying on traditional delta hedging, this system trains an RL agent (PPO) to learn a smarter, cost-efficient hedging strategy.

---

## 🚀 Overview

When selling options, traders face continuous risk due to price fluctuations of the underlying asset.
The standard approach — **delta hedging** — requires frequent rebalancing, which leads to high transaction costs.

In this project, I built an RL-based system that:

* Learns when and how much to hedge
* Reduces unnecessary trades
* Minimizes overall hedging error

---

## 🏗️ Project Structure

```
derivative_hedging_rl/
├── phase1_simulation.py   # Market simulation (GBM + Black-Scholes)
├── phase2_environment.py  # Custom Gym RL environment
├── phase3_training.py     # PPO agent training & evaluation
├── app.py                 # Streamlit dashboard
├── utils.py               # Helper functions
├── requirements.txt
├── data/
├── models/
└── results/
```

---

## 🧠 Core Concepts

* **Call Option** → Right to buy an asset at a fixed price
* **Delta (Δ)** → Sensitivity of option price to stock price
* **GBM** → Simulates realistic stock price movement
* **Black-Scholes** → Used for pricing and delta calculation
* **PPO (RL Algorithm)** → Learns optimal hedging policy

---

## ⚙️ How It Works

### 🔹 Phase 1 — Simulation

* Generate stock price paths using GBM
* Compute option prices and deltas using Black-Scholes
* Create dataset for training

### 🔹 Phase 2 — Environment

* Custom Gym environment designed from scratch
* State includes market + hedge info
* Reward penalizes hedging error and transaction cost

### 🔹 Phase 3 — Training

* Train PPO agent using Stable-Baselines3
* Compare against:

  * Delta hedging
  * No hedging

### 🔹 Phase 4 — Visualization

* Interactive Streamlit dashboard to explore:

  * Price simulations
  * Strategy comparisons
  * RL vs baseline performance

---

## 📊 Results

The trained RL agent:

* Learns smoother hedging strategies
* Reduces transaction costs compared to delta hedging
* Performs competitively on unseen data

---

## 🌐 Live Demo

👉 https://kali-derivative-hedging-rlagent.streamlit.app/

---

## 🛠️ Tech Stack

* Python
* NumPy, SciPy
* Gymnasium
* Stable-Baselines3
* PyTorch
* Streamlit, Plotly

---

## 💡 Why This Project Matters

* Combines **Machine Learning + Finance**
* Demonstrates **custom RL environment design**
* Includes **end-to-end pipeline (simulation → training → deployment)**
* Built with a focus on **real-world applicability**

---

## 👨‍💻 Author

**Ronit**
