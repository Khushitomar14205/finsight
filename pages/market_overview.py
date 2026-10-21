import streamlit as st
import plotly.express as px
from pages.common import get_context, demo_banner, require_data
from src.analytics import returns as R
from src.data_processing.validation import validate_dataframe
from src.data_ingestion.downloader import BhavcopyDownloader

settings, store = get_context()
st.title("📈 FinSight — Market Overview")
st.caption("Historical NSE research. Closes shown are historical observations, never live prices.")
demo_banner(settings)

cov = store.coverage()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Latest available data date", cov.get("latest_available_date") or "—")
c2.metric("Symbols covered", cov.get("symbols", 0))
c3.metric("Total records", f"{cov.get('rows', 0):,}")
c4.metric("Coverage", f"{cov.get('date_min') or '—'} → {cov.get('date_max') or '—'}")

if not require_data(store):
    st.stop()

df = store.load_all()
rep = validate_dataframe(df)
st.subheader("Data quality summary")
st.write(f"Duplicates: **{rep.duplicate_symbol_date}** · OHLC violations: **{rep.ohlc_violations}** · "
         f"Non-positive prices: **{rep.nonpositive_prices}** · Extreme moves flagged: **{rep.extreme_daily_moves}** · "
         f"Missing weekday sessions (incl. holidays): **{rep.missing_weekday_sessions}**")

st.subheader("Ingestion status")
log = BhavcopyDownloader(settings, store).read_log()
if log:
    import pandas as pd
    log_df = pd.DataFrame(log)
    st.write(f"Last status: **{log_df.iloc[-1]['status']}** on {log_df.iloc[-1]['date']} · "
             f"{(log_df['status'] == 'success').sum() + (log_df['status'] == 'local_success').sum()} successful days logged, "
             f"{(log_df['status'] == 'failed').sum()} failed (retryable).")
else:
    st.write("No ingestion runs logged yet.")

latest_date = df["date"].max()
day = df[df["date"] == latest_date]
movers = R.gainers_losers(day, 10)
st.subheader(f"Top movers on latest available date ({latest_date.date()})")
col1, col2 = st.columns(2)
with col1:
    st.write("**Gainers**"); st.dataframe(movers["gainers"], use_container_width=True)
with col2:
    st.write("**Losers**"); st.dataframe(movers["losers"], use_container_width=True)

st.subheader("Quick access")
st.write("Use the sidebar: **Market Explorer** for prices · **Stock Analysis** for indicators · "
         "**Portfolio Builder** to define holdings · **AI Research Assistant** to ask questions in plain English.")
