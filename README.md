# AlphaHedge — Production RL Derivative Hedging System

> **Options traders spend hours manually adjusting hedge positions. This system does it in milliseconds — with a PPO RL agent, LangGraph orchestration, volatility-aware routing, and LLM-generated risk reports.**

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?style=flat-square&logo=streamlit)](https://kali-derivative-hedging-rlagent.streamlit.app)
[![API Docs](https://img.shields.io/badge/API%20Docs-FastAPI-009688?style=flat-square&logo=fastapi)](https://ronityadav8905-alphahedge.hf.space/docs)

---

## The Problem

Selling options creates continuous risk. As the underlying stock moves, the trader must constantly adjust their hedge position — a process called delta hedging. The naive approach rebalances every single timestep, incurring massive transaction costs. Human traders guess based on intuition.

This system automates the entire decision — not with a fixed rule, but with an RL agent that **learns when to hedge, how much, and when to stay still.**

---

## What Makes This Different From a Standard RL Project

Most RL projects train a model and call it done. This one has a full production stack:

- **PPO agent with custom Gym environment** — 5D state vector, continuous action space, reward jointly penalising hedging error and transaction costs
- **96.5% error reduction** vs no-hedge baseline — evaluated across 200 held-out simulated paths
- **30% lower transaction costs** vs naive delta hedging — agent learns to avoid unnecessary rebalancing
- **LangGraph orchestration** — 5-node agent that classifies volatility regime, conditionally routes to the RL model, builds market context, and generates professional risk reports via LLM
- **Live REST API** — FastAPI serving the model with 6 endpoints, containerised with Docker, deployed on HF Spaces

This is the architecture a real quant trading system would need, not a notebook with a training loop.

---

## Results

| Strategy | Mean Hedging Error | Transaction Cost |
|---|---|---|
| **RL Agent (PPO)** | **0.0216** | **0.0035** |
| Naive Delta Hedge | 0.6245 | 0.0050 |
| No Hedge | 0.8500 | — |

- **96.5% reduction** in hedging error vs no-hedge baseline
- **30% lower** transaction costs vs naive delta hedging
- Evaluated across 200 held-out simulated paths via `/simulate` endpoint

> Note: Naive delta hedge error of 0.6245 reflects cumulative portfolio deviation over 252 timesteps including transaction costs — not single-step position tracking error.

---

## Architecture

```
Input → [MarketMonitor] → [VolatilityAnalyzer] → [Router]
                                                      ↓
                                              σ < 0.15 (low)
                                                      ↓
                                           [ContextNode] → [ReportGenerator] → Risk Report
                                                      ↑
                                              σ ≥ 0.15 (medium/high)
                                                      ↓
                                           [HedgeDecider (PPO)]
                                                      ↓
                                           [ContextNode] → [ReportGenerator (LLM)]
                                                      ↓
                                                 Risk Report

            All nodes share a TypedDict state flowing through the LangGraph
            PPO model served via FastAPI · Containerised with Docker · Deployed on HF Spaces
```

---

## The 5 LangGraph Nodes

| Node | Responsibility |
|---|---|
| **MarketMonitor** | Calculates BS delta and option price from inputs, packages market state |
| **VolatilityAnalyzer** | Classifies regime (low/medium/high) and decides whether hedging is needed |
| **HedgeDecider** | Loads PPO model directly, runs single-step inference, returns hedge recommendation |
| **ContextNode** | Builds market context (moneyness, BS delta, option price) for the report |
| **ReportGenerator** | Groq LLM (Llama 3.3 70B) writes a professional risk management report |

The conditional edge between VolatilityAnalyzer and HedgeDecider is the key design decision — low volatility environments skip the hedge entirely, avoiding unnecessary transaction costs.

---

## Tech Stack

| Layer | Technology | Why |
|---|---|---|
| RL Algorithm | PPO (Stable-Baselines3) | Sample-efficient, stable training on continuous action spaces |
| RL Environment | Custom OpenAI Gymnasium | 5D state, continuous [-1,1] action, joint reward function |
| Market Simulation | GBM + Black-Scholes | Industry-standard option pricing and delta calculation |
| Agent Orchestration | LangGraph with conditional edges | Handles branching — LangChain alone can't route dynamically |
| LLM | Groq (Llama 3.3 70B) | Fast inference, free tier, strong financial reasoning |
| Backend | FastAPI | Async, auto-generates OpenAPI docs, Pydantic validation |
| Frontend | Streamlit | 5-tab interactive dashboard with live agent integration |
| Deployment | Docker + HuggingFace Spaces + Streamlit Cloud | 24/7 uptime, free |

---

## Why Custom Gym Environment

Pre-built environments like CartPole don't model financial dynamics — there's no transaction cost penalty, no continuous action space, and no option pricing logic.

The key design decision was the reward function: `reward = -(hedging_error + λ * transaction_cost)`. The λ parameter controls the tradeoff between accuracy and cost. Without penalising transaction costs separately, the agent learns to perfectly track delta by rebalancing every step — which is exactly the naive strategy we're trying to improve on. Getting this reward shaping right is what produces the 30% transaction cost reduction.

---

## API Endpoints

```
GET  /health              → model status and health check
POST /predict             → single-step hedge recommendation
POST /simulate            → full GBM + PPO episodic evaluation (96.5% results)
GET  /model/info          → architecture and training config
POST /agent/analyze       → full LangGraph 5-node pipeline
POST /agent/scenario      → stress test: "what if price drops 10%?"
```

**Example — LangGraph agent:**
```bash
curl -X POST https://ronityadav8905-alphahedge.hf.space/agent/analyze \
  -H "Content-Type: application/json" \
  -d '{"stock_price": 105.0, "strike_price": 100.0,
       "time_to_expiry": 0.4, "volatility": 0.35, "risk_free_rate": 0.05}'
```

**Example — Scenario simulation:**
```bash
curl -X POST https://ronityadav8905-alphahedge.hf.space/agent/scenario \
  -H "Content-Type: application/json" \
  -d '{"stock_price": 105.0, "strike_price": 100.0, "time_to_expiry": 0.4,
       "volatility": 0.35, "risk_free_rate": 0.05,
       "scenario_description": "price drops 15%", "price_change_pct": -15.0}'
```

---

## Demo

1. Open the [live dashboard](https://kali-derivative-hedging-rlagent.streamlit.app)
2. Go to **🤖 RL vs Delta** tab — see aggregate performance across 200 paths
3. Go to **🧠 AlphaHedge Agent** tab
4. Set volatility above 0.30 — watch the agent route through all 5 nodes
5. Set volatility below 0.15 — watch the agent skip hedging entirely
6. Enable **Scenario Simulation** — type "price drops 15%" and see the agent stress test the position

---

## Project Structure

```
derivative-hedging-rl/
├── agent/
│   ├── graph.py                 # LangGraph StateGraph — 5 nodes, conditional edges
│   ├── nodes.py                 # Node functions (monitor/analyze/decide/context/report)
│   ├── state.py                 # TypedDict shared state definition
│   └── __init__.py
├── api/
│   ├── main.py                  # FastAPI — 6 endpoints
│   └── schemas.py               # Pydantic request/response models
├── phase1_simulation.py         # GBM simulation + Black-Scholes pricing
├── phase2_environment.py        # Custom OpenAI Gym environment
├── phase3_training.py           # PPO training + baseline evaluation
├── app.py                       # Streamlit dashboard (5 tabs)
├── utils.py                     # Helper functions
├── Dockerfile                   # Container definition
├── docker-compose.yml           # Multi-service orchestration
└── requirements.txt
```

---

## Running Locally

**Option 1 — Docker (recommended)**
```bash
git clone https://github.com/ronnnie-cr7/derivative-hedging-rl
cd derivative-hedging-rl
# Add GROQ_API_KEY to .env
docker compose up --build
# API:       http://localhost:8000/docs
# Dashboard: http://localhost:8501
```

**Option 2 — Local Python**
```bash
pip install -r requirements.txt
uvicorn api.main:app --reload
streamlit run app.py
```

---

## Future Improvements

- Compare PPO vs SAC vs TD3 — algorithm benchmarking tab
- Walk-forward validation — test on unseen market regimes
- Regime-switching market model — replace GBM with Heston or jump-diffusion
- Real market data integration — replace simulated paths with actual options data
- RAG layer — retrieve similar historical hedging scenarios to inform the LLM report

---

## Author

**Ronit Yadav** 

[![GitHub](https://img.shields.io/badge/GitHub-ronnnie--cr7-black)](https://github.com/ronnnie-cr7)