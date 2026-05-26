# 📈 AlphaHedge — Production RL Derivative Hedging System

A **production-grade** reinforcement learning system that dynamically hedges a European Call Option using a PPO agent — served via REST API, orchestrated by a LangGraph multi-agent system, containerised with Docker, and deployed live.

🔴 **Live API** → [ronityadav8905-alphahedge.hf.space/docs](https://ronityadav8905-alphahedge.hf.space/docs)
🟢 **Live Demo** → [kali-derivative-hedging-rlagent.streamlit.app](https://kali-derivative-hedging-rlagent.streamlit.app)

---

## 🏆 Results

| Strategy | Mean Hedging Error | Transaction Cost |
|---|---|---|
| **RL Agent (PPO)** | **0.0216** | **0.0035** |
| Naive Delta Hedge | 0.6245 | 0.0050 |
| No Hedge | 0.8500 | — |

- **96.5% reduction** in hedging error vs no-hedge baseline
- **30% lower** transaction costs vs naive delta hedging
- Evaluated across 200 simulated price paths

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────┐
│              Streamlit Dashboard                     │
│   Stock Sim · Hedging · RL vs Delta · BS Explorer   │
│              AlphaHedge Agent Tab                    │
└──────────────────────┬──────────────────────────────┘
                       │ HTTP
┌──────────────────────▼──────────────────────────────┐
│              FastAPI Gateway                         │
│   /health  /predict  /simulate  /agent/analyze       │
└──────┬──────────────────────────┬───────────────────┘
       │                          │
┌──────▼──────┐          ┌────────▼────────────────┐
│  PPO Model  │          │   LangGraph Agent        │
│  (SB3)      │          │                          │
│  /predict   │          │  MarketMonitor           │
│  /simulate  │          │       ↓                  │
└─────────────┘          │  VolatilityAnalyzer      │
                         │       ↓                  │
                         │  HedgeDecider (PPO)      │
                         │       ↓                  │
                         │  ReportGenerator (LLM)   │
                         └─────────────────────────┘

         All containerised with Docker · Deployed on HF Spaces
```

---

## 🤖 LangGraph Agent — 4 Node Pipeline

The agent autonomously monitors market conditions and generates risk reports.

```
Node 1: MarketMonitor      → packages market state
Node 2: VolatilityAnalyzer → classifies regime (low/medium/high)
Node 3: HedgeDecider       → calls PPO model for hedge ratio
Node 4: ReportGenerator    → LLM writes professional risk report
```

**Volatility Regimes:**
- 🟢 Low (σ < 0.15) → skip hedge, monitor only
- 🟡 Medium (0.15 ≤ σ < 0.30) → hedge recommended
- 🔴 High (σ ≥ 0.30) → hedge aggressively

**Example API call:**
```bash
curl -X POST https://ronityadav8905-alphahedge.hf.space/agent/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "stock_price": 105.0,
    "strike_price": 100.0,
    "time_to_expiry": 0.4,
    "volatility": 0.35,
    "risk_free_rate": 0.05
  }'
```

**Response:**
```json
{
  "volatility_regime": "high",
  "should_hedge": true,
  "hedge_recommendation": "Strategy: INCREASE | Hedge Position: 0.1 | Error vs BS: 0.4",
  "risk_report": "Our current risk exposure is elevated due to the high volatility regime...",
  "error": null
}
```

---

## 🚀 REST API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Model status and health check |
| POST | `/predict` | Single-step hedge recommendation |
| POST | `/simulate` | Full GBM simulation + PPO evaluation |
| GET | `/model/info` | Architecture and training config |
| POST | `/agent/analyze` | Full LangGraph agent pipeline |

---

## ⚙️ How It Works

### Phase 1 — Market Simulation
- Simulate 1,000 stock price paths using **Geometric Brownian Motion (GBM)**
- Price European Call Options using **Black-Scholes formula**
- Compute delta (hedge ratio) at each timestep across all paths

### Phase 2 — Custom RL Environment
- Built from scratch using **OpenAI Gymnasium**
- **State vector (5D):** stock price, option price, time to expiry, current hedge position, BS delta
- **Action space:** continuous [-1, 1] — change in hedge position
- **Reward:** jointly penalises hedging error + transaction costs

### Phase 3 — PPO Agent Training
- **Algorithm:** Proximal Policy Optimization (Stable-Baselines3)
- **Policy:** 2-layer MLP [256, 256 neurons]
- **Training:** 200,000 timesteps
- **Evaluation:** 200 held-out simulated paths

### Phase 4 — Production Deployment
- REST API via **FastAPI** with Pydantic validation
- **LangGraph** multi-agent orchestration (4 nodes)
- **LLM** report generation via Groq (Llama 3.3 70B)
- Containerised with **Docker** + Docker Compose
- Deployed on **Hugging Face Spaces** (live public URL)
- Interactive dashboard via **Streamlit**

---

## 🗂️ Project Structure

```
derivative-hedging-rl/
├── agent/
│   ├── graph.py             # LangGraph StateGraph — 4 nodes
│   ├── nodes.py             # Node functions (monitor/analyze/decide/report)
│   ├── state.py             # TypedDict shared state
│   └── __init__.py
├── api/
│   ├── main.py              # FastAPI app — 5 REST endpoints
│   └── schemas.py           # Pydantic request/response models
├── phase1_simulation.py     # GBM simulation + Black-Scholes pricing
├── phase2_environment.py    # Custom OpenAI Gym environment
├── phase3_training.py       # PPO training + baseline evaluation
├── app.py                   # Streamlit dashboard (5 tabs)
├── utils.py                 # Helper functions
├── Dockerfile               # Container definition
├── docker-compose.yml       # Multi-service orchestration
└── requirements.txt
```

---

## 🛠️ Tech Stack

**ML:** Python, PyTorch, Stable-Baselines3, OpenAI Gymnasium, NumPy, SciPy

**Agentic AI:** LangGraph, LangChain, Groq (Llama 3.3 70B)

**Production:** FastAPI, Uvicorn, Docker, Docker Compose, Hugging Face Spaces

**Visualization:** Streamlit, Plotly

---

## 🔧 Run Locally

**Option 1 — Docker (recommended)**
```bash
docker compose up --build
# API:       http://localhost:8000/docs
# Dashboard: http://localhost:8501
```

**Option 2 — Local Python**
```bash
pip install -r requirements.txt
uvicorn api.main:app --reload      # API at localhost:8000/docs
streamlit run app.py               # Dashboard at localhost:8501
```

**Environment variables (.env):**
```
GROQ_API_KEY=your_groq_key_here
```

---

## 👨‍💻 Author

**Ronit Yadav** — B.Tech AI/ML, NIT Kurukshetra
[![GitHub](https://img.shields.io/badge/GitHub-ronnnie--cr7-black)](https://github.com/ronnnie-cr7)
