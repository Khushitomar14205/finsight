import streamlit as st
import plotly.express as px
import pandas as pd
from pages.common import get_context, demo_banner, require_data, picker_symbols, symbol_info, trading_dates
from src.analytics import returns as R
from src.reporting.report import dataframe_to_csv_bytes

settings, store = get_context()
st.title("🔍 Historical Market Explorer")
demo_banner(settings)
if not require_data(store):
    st.stop()

equities_only = st.toggle("Equities only (EQ series)", value=True,
                          help="Off = include NCDs/bonds and other bhavcopy instruments, which trade rarely.")
symbols = picker_symbols(store, equities_only)
symbol = st.selectbox("Symbol", symbols)
_info = symbol_info(store, symbol)
if _info.get("rows"):
    st.caption(f"Series: {', '.join(_info.get('series', []))} · traded on {_info.get('coverage_pct', 0):.0%} of "
               f"all trading days" + ("" if _info.get("is_equity") else " · ⚠️ non-equity instrument (e.g. NCD/bond)"))
cov = store.coverage()
c1, c2 = st.columns(2)
start = c1.date_input("Start", value=pd.Timestamp(cov["date_min"]))
end = c2.date_input("End", value=pd.Timestamp(cov["date_max"]))
df = store.load(symbols=[symbol], start=start, end=end)
if df.empty:
    st.warning("No rows for this symbol in the selected range.")
    st.stop()
st.caption(f"Showing {len(df):,} records · {df['date'].min().date()} → {df['date'].max().date()} "
           f"(latest available observation: {df['date'].max().date()}) · Historical data, not live prices.")

df = df.sort_values("date")
df["daily_return"] = df["close"].pct_change()
df["cumulative_return"] = (1 + df["daily_return"].fillna(0)).cumprod() - 1

st.plotly_chart(px.line(df, x="date", y="close", title=f"{symbol} — Close price"), use_container_width=True)
cc1, cc2 = st.columns(2)
with cc1:
    st.plotly_chart(px.line(df, x="date", y="cumulative_return", title="Cumulative return"), use_container_width=True)
with cc2:
    st.plotly_chart(px.bar(df, x="date", y="volume", title="Volume"), use_container_width=True)

s = R.stock_summary(df, settings.risk_free_rate, settings.trading_days_per_year)
m1, m2, m3, m4 = st.columns(4)
m1.metric("Total return", f"{s['total_return']:.2%}")
m2.metric("Annualized volatility", f"{s['annualized_volatility']:.2%}")
m3.metric("Max drawdown", f"{s['max_drawdown']:.2%}")
m4.metric("Avg daily volume", f"{s['avg_daily_volume']:,.0f}" if s["avg_daily_volume"] else "—")

st.subheader("Underlying records")
st.dataframe(df, use_container_width=True, height=320)
st.download_button("Download filtered dataset (CSV)", dataframe_to_csv_bytes(df),
                   file_name=f"finsight_{symbol}_{start}_{end}.csv", mime="text/csv")

st.subheader("Daily gainers / losers (pick a date)")
dates = trading_dates(store)
pick = st.selectbox("Trading date", dates, index=len(dates) - 1)
day_df = store.load(start=pick, end=pick)   # one day only — never the full dataset
movers = R.gainers_losers(day_df, 10)
g, l = st.columns(2)
with g: st.write("**Gainers**"); st.dataframe(movers["gainers"], use_container_width=True)
with l: st.write("**Losers**"); st.dataframe(movers["losers"], use_container_width=True)
