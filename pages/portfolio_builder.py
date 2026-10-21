import io
import streamlit as st
import pandas as pd
from pages.common import get_context, demo_banner, require_data
from src.portfolio import engine as PE

settings, store = get_context()
st.title("🧺 Portfolio Builder")
demo_banner(settings)
if not require_data(store):
    st.stop()

symbols = store.available_symbols()
tab1, tab2, tab3 = st.tabs(["Manual weights", "Import CSV", "Saved portfolios"])

with tab1:
    picked = st.multiselect("Holdings", symbols, default=symbols[:3])
    mode = st.radio("Input mode", ["Weights (%)", "Investment amounts (₹)"])
    values = {}
    for sym in picked:
        values[sym] = st.number_input(sym, min_value=0.0, value=100.0 / max(len(picked), 1) if mode == "Weights (%)" else 10000.0, key=f"w_{sym}")
    initial = st.number_input("Initial investment value (₹)", min_value=1000.0, value=100000.0)
    if picked and st.button("Set as active portfolio"):
        try:
            w = PE.validate_weights(values) if mode == "Weights (%)" else PE.weights_from_amounts(values)
            st.session_state["portfolio_weights"] = w
            st.session_state["initial_value"] = initial
            st.success(f"Active portfolio set: {w}")
        except ValueError as e:
            st.error(str(e))

with tab2:
    st.write("CSV format: columns `symbol,weight` or `symbol,amount`.")
    up = st.file_uploader("Holdings CSV", type=["csv"])
    if up is not None:
        try:
            w = PE.parse_holdings_csv(io.BytesIO(up.getvalue()))
            st.session_state["portfolio_weights"] = w
            st.success(f"Imported: {w}")
        except Exception as e:
            st.error(f"Invalid CSV: {e}")
    st.download_button("Download template CSV", "symbol,weight\nRELIANCE,40\nTCS,35\nINFY,25\n",
                       file_name="portfolio_template.csv", mime="text/csv")

with tab3:
    name = st.text_input("Portfolio name to save")
    current = st.session_state.get("portfolio_weights", {})
    if current:
        st.write("Current active portfolio:", current)
        if name and st.button("Save portfolio"):
            p = PE.save_portfolio(current, name, settings.data_dir / "portfolios")
            st.success(f"Saved to {p.name}")
    saved_dir = settings.data_dir / "portfolios"
    if saved_dir.exists():
        for f in sorted(saved_dir.glob("*.json")):
            if st.button(f"Load {f.stem}"):
                st.session_state["portfolio_weights"] = PE.load_portfolio(f)
                st.success(f"Loaded {f.stem}: {st.session_state['portfolio_weights']}")
