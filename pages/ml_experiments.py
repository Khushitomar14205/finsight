import streamlit as st
import pandas as pd
from pages.common import get_context, demo_banner, require_data
from src.ml.experiments import run_experiment

settings, store = get_context()
st.title("🧪 Machine Learning Experiments")
st.warning("Experimental research only. Models are evaluated on held-out chronological data against a naive baseline. "
           "Historical patterns may not persist — this is not a price prediction and cannot guarantee profit.")
demo_banner(settings)
if not require_data(store):
    st.stop()

symbols = store.available_symbols()
symbol = st.selectbox("Symbol", symbols)
model = st.selectbox("Model", ["random_forest", "gradient_boosting", "ridge", "baseline"])
cost = st.slider("Transaction cost (bps per trade)", 0, 50, 10)
if st.button("Run experiment"):
    df = store.load(symbols=[symbol])
    try:
        with st.spinner("Training on chronological splits (no shuffling, no look-ahead)…"):
            out = run_experiment(df, model_name=model, transaction_cost_bps=float(cost))
        st.write(f"**Symbol:** {out['symbol']} · **Rows used:** {out['rows_used']} · **Split:** {out['split_method']}")
        st.write(f"Train {out['split']['train']} · Validation {out['split']['validation']} · Test {out['split']['test']}")
        rows = [{"Model": k, "MAE": v["mae"], "RMSE": v["rmse"], "Directional accuracy": v["directional_accuracy"]}
                for k, v in out["results"].items() if "mae" in v]
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
        strat = out["results"].get("strategy_test")
        if strat:
            st.write(f"**Test-period strategy return after costs:** {strat['cumulative_strategy_return_after_costs']:.2%} · "
                     f"**Buy-and-hold over same window:** {strat['cumulative_buy_and_hold_return_test']:.2%}")
            st.caption(strat["note"])
        if out["results"].get("feature_importance"):
            st.write("**Feature importance**")
            st.dataframe(pd.DataFrame({"feature": list(out["results"]["feature_importance"].keys()),
                                       "importance": list(out["results"]["feature_importance"].values())}),
                         use_container_width=True)
        st.info(out["limitations"])
    except ValueError as e:
        st.error(str(e))
