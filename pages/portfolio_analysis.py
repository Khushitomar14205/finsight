import streamlit as st
import plotly.express as px
import pandas as pd
from pages.common import get_context, demo_banner, require_data, get_portfolio_weights
from src.portfolio import engine as PE

settings, store = get_context()
st.title("🛡️ Portfolio Performance & Risk")
demo_banner(settings)
if not require_data(store):
    st.stop()
weights = get_portfolio_weights()
if not weights:
    st.info("No active portfolio yet — build one on the **Portfolio Builder** page.")
    st.stop()
st.write("**Holdings:**", weights)

df = store.load(symbols=list(weights.keys()))
initial = st.session_state.get("initial_value", 100000.0)
rebalance = st.selectbox("Rebalancing assumption", ["daily", "none"],
                         format_func=lambda x: "Daily rebalanced to target weights" if x == "daily" else "Buy-and-hold (weights drift)")
benchmark = st.selectbox("Benchmark (optional)", ["(none)"] + [s for s in store.available_symbols() if s not in weights])
rf = st.number_input("Risk-free rate (annual, for Sharpe)", 0.0, 0.30, float(settings.risk_free_rate), step=0.005, format="%.3f")

try:
    perf = PE.portfolio_performance(df, weights, initial_value=initial, rebalance=rebalance,
                                    risk_free_rate=rf, benchmark_symbol=None if benchmark == "(none)" else benchmark)
    risk = PE.portfolio_risk(df, weights, risk_free_rate=rf)
except ValueError as e:
    st.error(str(e)); st.stop()

m = st.columns(4)
m[0].metric("Total return", f"{perf['total_return']:.2%}")
m[1].metric("Annualized return", f"{perf['annualized_return']:.2%}")
m[2].metric("Final value (₹)", f"{perf['final_value']:,.0f}")
m[3].metric("Period", f"{perf['start']} → {perf['end']}")
if "benchmark" in perf:
    st.write(f"Benchmark **{perf['benchmark']['symbol']}** total return over same dates: **{perf['benchmark']['total_return']:.2%}**")

series = perf["series"]
st.plotly_chart(px.line(series, y="value", title="Portfolio value over time"), use_container_width=True)

m2 = st.columns(3)
m2[0].metric("Annualized volatility", f"{risk['annualized_volatility']:.2%}")
m2[1].metric("Sharpe ratio", f"{risk['sharpe_ratio']:.2f}")
m2[2].metric("Max drawdown", f"{risk['max_drawdown']:.2%}")
st.caption(f"Methodology: {perf['methodology']}. Annualization: {settings.trading_days_per_year} trading days. "
           "Historical risk is a sample-period estimate, not a forecast.")

corr = pd.DataFrame(risk["correlation"])
st.plotly_chart(px.imshow(corr, text_auto=".2f", title="Correlation matrix", color_continuous_scale="RdBu", zmin=-1, zmax=1),
                use_container_width=True)
cc1, cc2 = st.columns(2)
with cc1:
    st.write("**Component risk contribution**")
    st.dataframe(pd.DataFrame({"symbol": list(risk["component_risk_contribution"].keys()),
                               "risk_contribution": list(risk["component_risk_contribution"].values())}),
                 use_container_width=True)
with cc2:
    st.write("**Asset volatility**")
    st.dataframe(pd.DataFrame({"symbol": list(risk["asset_volatility"].keys()),
                               "annualized_volatility": list(risk["asset_volatility"].values())}),
                 use_container_width=True)
st.write("**Asset contribution to return (approx: weight × mean daily return × days)**")
st.dataframe(pd.DataFrame({"symbol": list(perf["asset_contribution_approx"].keys()),
                           "contribution": list(perf["asset_contribution_approx"].values())}), use_container_width=True)
with st.expander("Covariance matrix (annualized)"):
    st.dataframe(pd.DataFrame(risk["covariance_annualized"]), use_container_width=True)
