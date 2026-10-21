import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from pages.common import get_context, demo_banner, require_data
from src.analytics import returns as R
from src.portfolio.engine import price_wide

settings, store = get_context()
st.title("⚖️ Stock Comparison")
demo_banner(settings)
if not require_data(store):
    st.stop()

symbols = store.available_symbols()
picked = st.multiselect("Select 2–5 symbols", symbols, default=symbols[:2])
if len(picked) < 2:
    st.info("Pick at least two symbols to compare.")
    st.stop()
df = store.load(symbols=picked)
comp = R.compare_symbols(df, picked, settings.risk_free_rate)
st.caption(f"Common comparison range: {comp.get('common_start')} → {comp.get('common_end')} · "
           f"Missing observations per symbol: {comp.get('missing_observations')}")

rows = []
for sym, s in comp["symbols"].items():
    rows.append({"Symbol": sym, "Total return": s["total_return"], "Ann. return": s["annualized_return"],
                 "Volatility": s["annualized_volatility"], "Sharpe": s["sharpe_ratio"], "Max drawdown": s["max_drawdown"]})
st.dataframe(pd.DataFrame(rows), use_container_width=True)

wide = price_wide(df, picked).dropna(how="any")
norm = wide / wide.iloc[0] * 100
st.plotly_chart(px.line(norm, title="Normalized price performance (start = 100)"), use_container_width=True)
rets = wide.pct_change().dropna()
cum = (1 + rets).cumprod() - 1
st.plotly_chart(px.line(cum, title="Cumulative returns (common dates)"), use_container_width=True)
corr = rets.corr()
st.plotly_chart(px.imshow(corr, text_auto=".2f", title="Return correlation", color_continuous_scale="RdBu", zmin=-1, zmax=1),
                use_container_width=True)
st.plotly_chart(px.histogram(rets, barmode="overlay", opacity=0.6, title="Daily return distributions"), use_container_width=True)
