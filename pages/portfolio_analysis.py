import streamlit as st
import plotly.express as px
import pandas as pd
from pages.common import (get_context, demo_banner, require_data, get_portfolio_weights,
                          fmt_pct, fmt_num, fmt_date, METRIC_HELP)
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
st.write("**Holdings** (weights are validated and normalized to 100%):")
st.dataframe(pd.DataFrame([{"Symbol": k, "Weight": fmt_pct(v, signed=False)}
                           for k, v in sorted(weights.items(), key=lambda kv: -kv[1])]),
             use_container_width=True, hide_index=True)

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
m[0].metric("Total return", fmt_pct(perf["total_return"]), help=METRIC_HELP["total_return"])
m[1].metric("Annualized return", fmt_pct(perf["annualized_return"]), help=METRIC_HELP["annualized_return"])
m[2].metric("Final value (₹)", f"{perf['final_value']:,.0f}")
m[3].metric("Period", f"{fmt_date(perf['start'])} → {fmt_date(perf['end'])}")
if "benchmark" in perf:
    st.write(f"Benchmark **{perf['benchmark']['symbol']}** total return over same dates: "
             f"**{fmt_pct(perf['benchmark']['total_return'])}**")

series = perf["series"]
st.plotly_chart(px.line(series, y="value", title="Portfolio value over time"), use_container_width=True)

m2 = st.columns(3)
m2[0].metric("Annualized volatility", fmt_pct(risk["annualized_volatility"]), help=METRIC_HELP["volatility"])
m2[1].metric("Sharpe ratio", fmt_num(risk["sharpe_ratio"]), help=METRIC_HELP["sharpe"])
m2[2].metric("Max drawdown", fmt_pct(risk["max_drawdown"]), help=METRIC_HELP["max_drawdown"])
st.caption(f"Methodology: {perf['methodology']}. Annualization: {settings.trading_days_per_year} trading days. "
           "Historical risk is a sample-period estimate, not a forecast.")

corr = pd.DataFrame(risk["correlation"])
st.plotly_chart(px.imshow(corr, text_auto=".2f", title="Correlation matrix", color_continuous_scale="RdBu", zmin=-1, zmax=1),
                use_container_width=True)
st.caption("Correlation of daily returns: +1 = moved together, 0 = unrelated, −1 = opposite. "
           "Lower correlations between holdings generally mean smoother portfolio swings.")
cc1, cc2 = st.columns(2)
with cc1:
    st.write("**Component risk contribution** (share of total portfolio risk)")
    st.dataframe(pd.DataFrame([{"Symbol": k, "Risk contribution": fmt_pct(v)}
                               for k, v in risk["component_risk_contribution"].items()]),
                 use_container_width=True, hide_index=True)
with cc2:
    st.write("**Asset volatility** (annualized)")
    st.dataframe(pd.DataFrame([{"Symbol": k, "Volatility": fmt_pct(v)}
                               for k, v in risk["asset_volatility"].items()]),
                 use_container_width=True, hide_index=True)
st.write("**Asset contribution to return (approx: weight × mean daily return × days)**")
st.dataframe(pd.DataFrame([{"Symbol": k, "Contribution": fmt_pct(v)}
                           for k, v in perf["asset_contribution_approx"].items()]),
             use_container_width=True, hide_index=True)
with st.expander("Covariance matrix (annualized)"):
    st.dataframe(pd.DataFrame(risk["covariance_annualized"]), use_container_width=True)
