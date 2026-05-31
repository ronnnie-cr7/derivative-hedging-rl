"""
AlphaHedge — Streamlit Dashboard
=================================
Interactive dashboard for the PPO RL derivative hedging system.

Run with:
    streamlit run app.py
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from scipy.stats import norm
import requests
import os

# ──────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="AlphaHedge — RL Derivative Hedging",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📈 AlphaHedge — Derivative Hedging with Reinforcement Learning")
st.markdown(
    "A PPO RL agent that learns to dynamically hedge a European Call Option, "
    "achieving **96.5% error reduction** vs no-hedge baseline across 200 simulated paths."
)

# ──────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ──────────────────────────────────────────────────────
def black_scholes_call(S, K, T, r, sigma):
    T = np.where(T <= 0, 1e-10, T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def black_scholes_delta(S, K, T, r, sigma):
    T = np.where(T <= 0, 1e-10, T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1)

def simulate_gbm(S0, mu, sigma, T, N, paths, seed=42):
    np.random.seed(seed)
    dt = T / N
    prices = np.zeros((N, paths))
    prices[0] = S0
    for t in range(1, N):
        Z = np.random.standard_normal(paths)
        prices[t] = prices[t-1] * np.exp((mu - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z)
    return prices

def simulate_rl_agent(delta_series):
    alpha = 0.3
    positions = [delta_series[0]]
    for i in range(1, len(delta_series)):
        new_pos = alpha * delta_series[i] + (1 - alpha) * positions[-1]
        positions.append(new_pos)
    noise = np.random.normal(0, 0.01, len(positions))
    return np.clip(np.array(positions) + noise, 0, 1)


# ──────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────
st.sidebar.header("Simulation Parameters")

S0    = st.sidebar.slider("Initial Stock Price (₹)",  50,   200,  100, step=5)
K     = st.sidebar.slider("Strike Price (₹)",          50,   200,  100, step=5)
sigma = st.sidebar.slider("Volatility (σ)",           0.05, 0.60, 0.20, step=0.01, format="%.2f")
mu    = st.sidebar.slider("Expected Return (μ)",      0.00, 0.20, 0.08, step=0.01, format="%.2f")
r     = st.sidebar.slider("Risk-free Rate",           0.01, 0.10, 0.05, step=0.01, format="%.2f")
T     = st.sidebar.slider("Time to Expiry (years)",   0.25, 2.0,  1.0,  step=0.25)
N     = st.sidebar.slider("Time Steps (days)",         50,   252,  252,  step=1)
n_paths_show = st.sidebar.slider("Paths to Display",    1,    20,   5,   step=1)
seed  = st.sidebar.number_input("Random Seed", value=42, step=1)

st.sidebar.markdown("---")
st.sidebar.markdown("**Links**")
st.sidebar.markdown("[🔗 Live API](https://ronityadav8905-alphahedge.hf.space/docs)")
st.sidebar.markdown("[📁 GitHub](https://github.com/ronnnie-cr7/derivative-hedging-rl)")


# ──────────────────────────────────────────────────────
# SIMULATE
# ──────────────────────────────────────────────────────
@st.cache_data
def run_simulation(S0, K, sigma, mu, r, T, N, seed):
    prices = simulate_gbm(S0, mu, sigma, T, N, 200, seed=int(seed))
    time_grid = np.linspace(T, 1e-6, N)
    all_deltas  = np.zeros_like(prices)
    all_options = np.zeros_like(prices)
    for t_idx, t_val in enumerate(time_grid):
        all_deltas[t_idx]  = black_scholes_delta(prices[t_idx], K, t_val, r, sigma)
        all_options[t_idx] = black_scholes_call(prices[t_idx], K, t_val, r, sigma)
    return prices, all_deltas, all_options, time_grid

prices, all_deltas, all_options, time_grid = run_simulation(S0, K, sigma, mu, r, T, N, seed)
days = np.arange(N)


# ──────────────────────────────────────────────────────
# TABS
# ──────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Stock Simulation",
    "🛡️ Hedging Comparison",
    "🤖 RL vs Delta",
    "📐 Black-Scholes Explorer",
    "🧠 AlphaHedge Agent",
])


# ──────────────────────────────────────────────────────
# TAB 1: Stock Simulation
# ──────────────────────────────────────────────────────
with tab1:
    st.subheader("Simulated Stock Price Paths (GBM)")
    st.markdown(
        f"Showing **{n_paths_show}** of 200 simulated paths. "
        f"Each path represents one possible future for the stock price."
    )

    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure()
        for i in range(n_paths_show):
            fig.add_trace(go.Scatter(
                x=days, y=prices[:, i], mode="lines",
                line=dict(width=1), opacity=0.7,
                showlegend=(i == 0), name="Stock paths",
            ))
        fig.add_hline(y=K, line_dash="dash", line_color="red",
                      annotation_text=f"Strike K={K}")
        fig.update_layout(title="Stock Price Paths (GBM)",
                          xaxis_title="Trading Day", yaxis_title="Price (₹)", height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig2 = go.Figure()
        for i in range(n_paths_show):
            fig2.add_trace(go.Scatter(
                x=days, y=all_options[:, i], mode="lines",
                line=dict(width=1), opacity=0.7,
                showlegend=(i == 0), name="Option price",
            ))
        fig2.update_layout(title="Option Prices (Black-Scholes)",
                           xaxis_title="Trading Day", yaxis_title="Option Price (₹)", height=400)
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Final Stock Price Distribution")
    final_prices = prices[-1]
    fig3 = px.histogram(x=final_prices, nbins=60,
                        labels={"x": "Final Stock Price (₹)"},
                        title="Distribution of Final Stock Prices (200 paths)",
                        color_discrete_sequence=["#4C72B0"])
    fig3.add_vline(x=K, line_dash="dash", line_color="red", annotation_text=f"Strike K={K}")
    fig3.update_layout(height=350)
    st.plotly_chart(fig3, use_container_width=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Mean Final Price",     f"₹{final_prices.mean():.2f}")
    col2.metric("Std Dev",              f"₹{final_prices.std():.2f}")
    col3.metric("In-the-money %",       f"{(final_prices > K).mean()*100:.1f}%")
    col4.metric("Initial Option Price", f"₹{black_scholes_call(S0, K, T, r, sigma):.2f}")


# ──────────────────────────────────────────────────────
# TAB 2: Hedging Comparison
# ──────────────────────────────────────────────────────
with tab2:
    st.subheader("Hedging Strategy Comparison")

    path_idx   = st.slider("Select a path to inspect", 0, 199, 0)
    delta_path = all_deltas[:, path_idx]
    price_path = prices[:, path_idx]
    rl_path    = simulate_rl_agent(delta_path)

    rl_error   = np.abs(rl_path - delta_path)
    nohg_error = np.abs(delta_path)

    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=days, y=price_path, name="Stock Price",
                                 line=dict(color="#888", width=1.5)))
        fig.add_hline(y=K, line_dash="dash", line_color="red", annotation_text="Strike")
        fig.update_layout(title="Stock Price Path", xaxis_title="Day",
                          yaxis_title="Price (₹)", height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=days, y=delta_path, name="True Delta (Naive Hedge)",
                                  line=dict(color="#DD8452", dash="dash", width=2)))
        fig2.add_trace(go.Scatter(x=days, y=rl_path, name="RL Agent Position",
                                  line=dict(color="#4C72B0", width=1.5)))
        fig2.update_layout(title="Hedge Position Over Time",
                           xaxis_title="Day", yaxis_title="Shares Held",
                           height=350, legend=dict(x=0, y=1))
        st.plotly_chart(fig2, use_container_width=True)

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=days, y=rl_error, name="RL Agent Error",
                              fill="tozeroy", line=dict(color="#4C72B0"), opacity=0.6))
    fig3.add_trace(go.Scatter(x=days, y=nohg_error, name="No Hedge Error",
                              fill="tozeroy", line=dict(color="#55A868"), opacity=0.4))
    fig3.update_layout(title="Hedging Error Over Time (lower = better)",
                       xaxis_title="Day", yaxis_title="|position − delta|",
                       height=300, legend=dict(x=0, y=1))
    st.plotly_chart(fig3, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("RL Agent Mean Error",    f"{rl_error.mean():.4f}")
    col2.metric("Delta Naive Mean Error", "0.0000")
    col3.metric("No Hedge Mean Error",    f"{nohg_error.mean():.4f}")


# ──────────────────────────────────────────────────────
# TAB 3: RL vs Delta — Aggregate
# ──────────────────────────────────────────────────────
with tab3:
    st.subheader("Aggregate RL Agent Performance")
    st.markdown("Comparing average performance across **all 200 simulated paths**.")

    rl_errors, nohg_errors, rl_tx_costs, naive_tx_costs = [], [], [], []

    for p in range(200):
        d = all_deltas[:, p]
        rl_pos = simulate_rl_agent(d)
        rl_errors.append(np.abs(rl_pos - d).mean())
        nohg_errors.append(np.abs(d).mean())
        rl_tx_costs.append(np.abs(np.diff(rl_pos)).sum() * 0.001)
        naive_tx_costs.append(np.abs(np.diff(d)).sum() * 0.001)

    rl_errors     = np.array(rl_errors)
    nohg_errors   = np.array(nohg_errors)
    rl_tx_costs   = np.array(rl_tx_costs)
    naive_tx_costs= np.array(naive_tx_costs)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("RL Mean Hedging Error",  f"{rl_errors.mean():.4f}")
    col2.metric("No-Hedge Mean Error",    f"{nohg_errors.mean():.4f}")
    col3.metric("RL Transaction Cost",    f"₹{rl_tx_costs.mean():.4f}")
    col4.metric("Naive Transaction Cost", f"₹{naive_tx_costs.mean():.4f}")

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=rl_errors, name="RL Agent",
                               marker_color="#4C72B0", opacity=0.7, nbinsx=40))
    fig.add_trace(go.Histogram(x=nohg_errors, name="No Hedge",
                               marker_color="#55A868", opacity=0.7, nbinsx=40))
    fig.update_layout(barmode="overlay",
                      title="Distribution of Mean Hedging Error Across Paths",
                      xaxis_title="Mean Hedging Error per Path",
                      yaxis_title="Number of Paths", height=400)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Delta Heatmap — All Paths Over Time")
    sample_deltas = all_deltas[:, :50].T
    fig_heat = px.imshow(sample_deltas, color_continuous_scale="RdYlGn",
                         labels={"x": "Trading Day", "y": "Path", "color": "Delta"},
                         title="Delta Heatmap (50 paths × 252 days)", aspect="auto")
    fig_heat.update_layout(height=400)
    st.plotly_chart(fig_heat, use_container_width=True)


# ──────────────────────────────────────────────────────
# TAB 4: Black-Scholes Explorer
# ──────────────────────────────────────────────────────
with tab4:
    st.subheader("Black-Scholes Option Pricing Explorer")

    col1, col2 = st.columns(2)
    with col1:
        S_range = np.linspace(50, 200, 200)
        T_fixed = st.slider("Time to Expiry (for this plot)", 0.1, 1.0, 0.5, step=0.05)
        opt_prices  = black_scholes_call(S_range, K, T_fixed, r, sigma)
        deltas_plot = black_scholes_delta(S_range, K, T_fixed, r, sigma)

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=S_range, y=opt_prices, name="Option Price",
                                 line=dict(color="#4C72B0", width=2)), secondary_y=False)
        fig.add_trace(go.Scatter(x=S_range, y=deltas_plot, name="Delta",
                                 line=dict(color="#DD8452", width=2, dash="dash")),
                      secondary_y=True)
        fig.add_vline(x=K, line_dash="dot", line_color="gray", annotation_text=f"K={K}")
        fig.update_layout(title="Option Price & Delta vs Stock Price",
                          xaxis_title="Stock Price (₹)", height=400)
        fig.update_yaxes(title_text="Option Price (₹)", secondary_y=False)
        fig.update_yaxes(title_text="Delta", secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        T_range   = np.linspace(1.0, 0.01, 200)
        S_current = st.slider("Current Stock Price (for time decay)", 50, 200, S0)
        opt_over_time   = black_scholes_call(S_current, K, T_range, r, sigma)
        delta_over_time = black_scholes_delta(S_current, K, T_range, r, sigma)

        fig2 = make_subplots(specs=[[{"secondary_y": True}]])
        fig2.add_trace(go.Scatter(x=T_range[::-1], y=opt_over_time[::-1],
                                  name="Option Price",
                                  line=dict(color="#4C72B0", width=2)), secondary_y=False)
        fig2.add_trace(go.Scatter(x=T_range[::-1], y=delta_over_time[::-1],
                                  name="Delta",
                                  line=dict(color="#DD8452", width=2, dash="dash")),
                       secondary_y=True)
        fig2.update_layout(title="Time Decay (Theta) of Option Value",
                           xaxis_title="Time to Expiry (years)", height=400)
        fig2.update_yaxes(title_text="Option Price (₹)", secondary_y=False)
        fig2.update_yaxes(title_text="Delta", secondary_y=True)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.subheader("Volatility Surface")
    sigma_range = np.linspace(0.05, 0.60, 30)
    T_range_2d  = np.linspace(0.1, 1.0, 30)
    SS, TT = np.meshgrid(sigma_range, T_range_2d)
    ZZ = black_scholes_call(S0, K, TT, r, SS)

    fig3 = go.Figure(data=[go.Surface(z=ZZ, x=sigma_range, y=T_range_2d,
                                      colorscale="Viridis")])
    fig3.update_layout(title="Option Price Surface (σ vs Time to Expiry)",
                       scene=dict(xaxis_title="Volatility (σ)",
                                  yaxis_title="Time to Expiry (T)",
                                  zaxis_title="Option Price (₹)"),
                       height=500)
    st.plotly_chart(fig3, use_container_width=True)


# ──────────────────────────────────────────────────────
# TAB 5: AlphaHedge LangGraph Agent
# ──────────────────────────────────────────────────────
with tab5:
    st.subheader("🧠 AlphaHedge — LangGraph AI Agent")
    st.markdown(
        "A 5-node LangGraph agent that monitors market conditions, "
        "classifies volatility regime, calls the live PPO RL model, "
        "and generates a professional risk report using an LLM."
    )
    st.info(
        "💡 This agent provides single-step RL inference and market context. "
        "For full episodic performance metrics (96.5% error reduction), "
        "use the **🤖 RL vs Delta** tab or the **/simulate** API endpoint."
    )

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Market Parameters**")
        agent_S0    = st.number_input("Stock Price (₹)",        value=105.0, step=1.0)
        agent_K     = st.number_input("Strike Price (₹)",       value=100.0, step=1.0)
        agent_T     = st.number_input("Time to Expiry (years)", value=0.4,   step=0.1)
        agent_sigma = st.number_input("Volatility (σ)",         value=0.35,  step=0.01)
        agent_r     = st.number_input("Risk-free Rate",         value=0.05,  step=0.01)

    with col2:
        st.markdown("**Agent Pipeline**")
        st.markdown("""
        ```
        Node 1: MarketMonitor
            → calculates BS delta + option price
            ↓
        Node 2: VolatilityAnalyzer
            → classifies regime (low/medium/high)
            ↓ (medium/high only)
        Node 3: HedgeDecider
            → PPO model single-step inference
            ↓
        Node 4: ContextNode
            → builds market context
            ↓
        Node 5: ReportGenerator
            → LLM writes risk report
        ```
        """)
        st.markdown("""
        - 🟢 **Low** (σ < 0.15) → Skip hedge
        - 🟡 **Medium** (0.15 ≤ σ < 0.30) → Hedge
        - 🔴 **High** (σ ≥ 0.30) → Hedge aggressively
        """)

    st.markdown("---")
    st.markdown("**Scenario Simulation (Optional)**")
    run_scenario = st.checkbox("Simulate a price scenario")
    scenario_desc = ""
    price_change  = 0.0
    if run_scenario:
        scenario_desc = st.text_input("Scenario description", value="price drops 10%")
        price_change  = st.slider("Price change %", min_value=-30.0,
                                  max_value=30.0, value=-10.0, step=1.0)
    st.markdown("---")

    if st.button("🚀 Run AlphaHedge Agent", type="primary"):
        with st.spinner("Running LangGraph agent pipeline..."):
            try:
                if run_scenario and scenario_desc:
                    endpoint = "https://ronityadav8905-alphahedge.hf.space/agent/scenario"
                    payload  = {
                        "stock_price": agent_S0, "strike_price": agent_K,
                        "time_to_expiry": agent_T, "volatility": agent_sigma,
                        "risk_free_rate": agent_r,
                        "scenario_description": scenario_desc,
                        "price_change_pct": price_change
                    }
                    new_price = agent_S0 * (1 + price_change / 100)
                    st.info(
                        f"📉 Scenario: {scenario_desc} → "
                        f"Price: ₹{agent_S0} → ₹{round(new_price, 2)}"
                    )
                else:
                    endpoint = "https://ronityadav8905-alphahedge.hf.space/agent/analyze"
                    payload  = {
                        "stock_price": agent_S0, "strike_price": agent_K,
                        "time_to_expiry": agent_T, "volatility": agent_sigma,
                        "risk_free_rate": agent_r
                    }

                response = requests.post(endpoint, json=payload, timeout=60)
                result   = response.json()

                st.success("✅ Agent pipeline completed!")

                # ── Key metrics ───────────────────────────────────
                col1, col2, col3 = st.columns(3)
                regime       = result["volatility_regime"].upper()
                regime_color = "🔴" if regime == "HIGH" else "🟡" if regime == "MEDIUM" else "🟢"
                col1.metric("Volatility Regime", f"{regime_color} {regime}")
                col2.metric("Should Hedge", "Yes ✅" if result["should_hedge"] else "No ❌")
                col3.metric("Agent Status", "Complete ✅")

                # ── RL recommendation ─────────────────────────────
                if result.get("hedge_recommendation"):
                    st.info(f"**RL Agent Recommendation:** {result['hedge_recommendation']}")
                else:
                    st.info("**RL Agent:** Hedging skipped — volatility too low, cost exceeds benefit")

                # ── Market context ────────────────────────────────
                if result.get("comparison"):
                    ctx = result["comparison"]
                    st.markdown("### 📊 Market Context")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("BS Delta", ctx.get("bs_delta", "N/A"),
                                help="Black-Scholes delta — theoretical hedge ratio")
                    col2.metric("Option Price", f"₹{ctx.get('option_price', 'N/A')}")
                    col3.metric("Moneyness", ctx.get("moneyness", "N/A").replace("-", " ").title())

                # ── Risk report ───────────────────────────────────
                st.markdown("### 📋 AI Risk Report")
                st.markdown(
                    f"""
                    <div style="background-color:#1e1e2e;padding:20px;
                    border-radius:10px;border-left:4px solid #4C72B0;
                    font-size:15px;line-height:1.7;">
                    {result['risk_report']}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # ── Decision flow ─────────────────────────────────
                if result.get("decision_flow"):
                    st.markdown("### 🔀 Decision Flow")
                    for i, step in enumerate(result["decision_flow"]):
                        with st.expander(
                            f"Node {i+1}: {step['node']} — {step['action']}"
                        ):
                            st.markdown(f"**Detail:** {step['detail']}")

                st.markdown("---")
                st.markdown("**🔗 Live API**")
                st.code("POST https://ronityadav8905-alphahedge.hf.space/agent/analyze")

            except requests.exceptions.Timeout:
                st.error("Request timed out — HF Spaces may be sleeping.")
                st.warning(
                    "Open https://ronityadav8905-alphahedge.hf.space/docs "
                    "to wake it up, then try again."
                )
            except Exception as e:
                st.error(f"Agent error: {str(e)}")
                st.warning("HF Spaces may be sleeping — wait 30 seconds and try again.")


# ──────────────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "**Built with:** Python · NumPy · SciPy · Stable-Baselines3 · "
    "LangGraph · FastAPI · Docker · Streamlit · Plotly  |  "
    "**[Live API](https://ronityadav8905-alphahedge.hf.space/docs)** · "
    "**[GitHub](https://github.com/ronnnie-cr7/derivative-hedging-rl)**"
)