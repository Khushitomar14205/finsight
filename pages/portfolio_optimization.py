import streamlit as st
import pandas as pd
from pages.common import get_context, demo_banner, require_data, get_portfolio_weights, fmt_pct, fmt_num
from src.optimization import optimizer as OPT
from src.portfolio import engine as PE

settings, store = get_context()
st.title("🎯 Portfolio Optimization & What-If Analysis")
st.caption("Optimized allocations are historical-model outputs from the sample period — not recommendations or guarantees.")
demo_banner(settings)
if not require_data(store):
    st.stop()

symbols = store.available_symbols()
picked = st.multiselect("Universe for optimization", symbols,
                        default=list(get_portfolio_weights().keys()) or symbols[:3])
max_w = st.slider("Maximum weight per asset", 0.2, 1.0, 1.0, 0.05)
if len(picked) >= 2 and st.button("Run optimization comparison"):
    df = store.load(symbols=picked)
    try:
        results = {}
        eq = OPT.equal_weight(picked)
        results["Equal weight"] = {"weights": eq, "performance": OPT._clean_perf(df, eq), "risk": OPT._clean_risk(df, eq)}
        results["Minimum variance"] = OPT.minimum_variance(df, picked, max_weight=max_w)
        results["Maximum Sharpe"] = OPT.maximum_sharpe(df, picked, risk_free_rate=settings.risk_free_rate, max_weight=max_w)
        current = get_portfolio_weights()
        if current and set(current) <= set(picked):
            results["Current portfolio"] = {"weights": current, "performance": OPT._clean_perf(df, current), "risk": OPT._clean_risk(df, current)}
        st.session_state["opt_results"] = results
    except Exception as e:
        st.error(str(e))

results = st.session_state.get("opt_results")
if results:
    rows = [{"Allocation": name,
             "Total return": fmt_pct(r["performance"]["total_return"]),
             "Ann. return": fmt_pct(r["performance"]["annualized_return"]),
             "Volatility": fmt_pct(r["risk"]["annualized_volatility"]),
             "Sharpe": fmt_num(r["risk"]["sharpe_ratio"]),
             "Max drawdown": fmt_pct(r["risk"]["max_drawdown"]),
             "Weights": ", ".join(f"{k} {v:.0%}" for k, v in r["weights"].items())}
            for name, r in results.items()]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

st.subheader("What-if: adjust weights")
base = get_portfolio_weights()
if not base:
    st.info("Set an active portfolio in the Portfolio Builder to run what-if analysis.")
    st.stop()
new = {}
for sym, w in base.items():
    new[sym] = st.slider(sym, 0.0, 1.0, float(w), 0.05, key=f"whatif_{sym}")
extra = st.multiselect("Add symbols", [s for s in symbols if s not in base])
for sym in extra:
    new[sym] = st.slider(sym, 0.0, 1.0, 0.0, 0.05, key=f"whatif_{sym}")
if sum(new.values()) == 0:
    st.warning("Weights can't all be zero.")
elif st.button("Compare what-if vs current"):
    df = store.load(symbols=list(new.keys()))
    try:
        out = OPT.what_if(df, base, PE.validate_weights(new), risk_free_rate=settings.risk_free_rate)
        rows = []
        for label, part in (("Current", out["base"]), ("Proposed", out["proposed"])):
            rows.append({"Scenario": label, "Total return": fmt_pct(part["performance"]["total_return"]),
                         "Volatility": fmt_pct(part["risk"]["annualized_volatility"]),
                         "Sharpe": fmt_num(part["risk"]["sharpe_ratio"]),
                         "Max drawdown": fmt_pct(part["risk"]["max_drawdown"]),
                         "Weights": ", ".join(f"{k} {v:.0%}" for k, v in part["weights"].items())})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.caption(out["note"])
    except Exception as e:
        st.error(str(e))
