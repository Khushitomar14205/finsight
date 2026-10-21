import streamlit as st
import plotly.express as px
import pandas as pd
from pages.common import (get_context, demo_banner, require_data, picker_symbols, symbol_info,
                          trading_dates, fmt_pct, fmt_rs, METRIC_HELP)
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
rng = st.segmented_control("Quick range", ["1M", "3M", "6M", "1Y", "3Y", "5Y", "All"], default="All")
_maxd, _mind = pd.Timestamp(cov["date_max"]), pd.Timestamp(cov["date_min"])
_offsets = {"1M": pd.DateOffset(months=1), "3M": pd.DateOffset(months=3), "6M": pd.DateOffset(months=6),
            "1Y": pd.DateOffset(years=1), "3Y": pd.DateOffset(years=3), "5Y": pd.DateOffset(years=5)}
_default_start = max(_mind, _maxd - _offsets[rng]) if rng != "All" else _mind
c1, c2 = st.columns(2)
start = c1.date_input("Start", value=_default_start, min_value=_mind, max_value=_maxd, key=f"exp_start_{rng}")
end = c2.date_input("End", value=_maxd, min_value=_mind, max_value=_maxd)
df = store.load(symbols=[symbol], start=start, end=end)
if df.empty:
    st.warning("No rows for this symbol in the selected range.")
    st.stop()
st.caption(f"Showing {len(df):,} records · {df['date'].min().date()} → {df['date'].max().date()} "
           f"(latest available observation: {df['date'].max().date()}) · Historical data, not live prices.")

df = df.sort_values("date")
df["daily_return"] = df["close"].pct_change()
df["cumulative_return"] = (1 + df["daily_return"].fillna(0)).cumprod() - 1

fig_close = px.line(df, x="date", y="close", title=f"{symbol} — Close price")
fig_close.update_layout(yaxis_title="Close (₹)", xaxis_title="Date", hovermode="x unified")
st.plotly_chart(fig_close, use_container_width=True)
cc1, cc2 = st.columns(2)
with cc1:
    fig_cum = px.line(df, x="date", y="cumulative_return", title="Cumulative return")
    fig_cum.update_layout(yaxis=dict(tickformat=".0%", title="Cumulative return"), hovermode="x unified")
    st.plotly_chart(fig_cum, use_container_width=True)
with cc2:
    fig_vol = px.bar(df, x="date", y="volume", title="Volume")
    fig_vol.update_layout(yaxis_title="Shares traded", xaxis_title="Date", hovermode="x unified")
    st.plotly_chart(fig_vol, use_container_width=True)

s = R.stock_summary(df, settings.risk_free_rate, settings.trading_days_per_year)
m1, m2, m3, m4 = st.columns(4)
m1.metric("Total return", fmt_pct(s["total_return"]), help=METRIC_HELP["total_return"])
m2.metric("Annualized volatility", fmt_pct(s["annualized_volatility"]), help=METRIC_HELP["volatility"])
m3.metric("Max drawdown", fmt_pct(s["max_drawdown"]), help=METRIC_HELP["max_drawdown"])
m4.metric("Avg daily volume", f"{s['avg_daily_volume']:,.0f}" if s["avg_daily_volume"] else "—")

st.subheader("Underlying records")
st.caption("Sortable — click any column header. Prices in ₹; returns are close-to-close on unadjusted prices.")
show_cols = [c for c in ["date", "open", "high", "low", "close", "prev_close", "volume", "turnover",
                         "daily_return", "cumulative_return"] if c in df.columns]
st.dataframe(df[show_cols], use_container_width=True, height=320, hide_index=True,
             column_config={
                 "date": st.column_config.DatetimeColumn("Date", format="YYYY-MM-DD"),
                 "open": st.column_config.NumberColumn("Open", format="%.2f"),
                 "high": st.column_config.NumberColumn("High", format="%.2f"),
                 "low": st.column_config.NumberColumn("Low", format="%.2f"),
                 "close": st.column_config.NumberColumn("Close", format="%.2f"),
                 "prev_close": st.column_config.NumberColumn("Prev close", format="%.2f"),
                 "volume": st.column_config.NumberColumn("Volume", format="%d"),
                 "turnover": st.column_config.NumberColumn("Turnover (₹)", format="%.0f"),
                 "daily_return": st.column_config.NumberColumn("Daily return", format="percent"),
                 "cumulative_return": st.column_config.NumberColumn("Cumulative return", format="percent"),
             })
st.download_button("Download filtered dataset (CSV)", dataframe_to_csv_bytes(df),
                   file_name=f"finsight_{symbol}_{start}_{end}.csv", mime="text/csv")

st.subheader("Daily gainers / losers (pick a date)")
dates = trading_dates(store)
pick = st.selectbox("Trading date", dates, index=len(dates) - 1)
day_df = store.load(start=pick, end=pick)   # one day only — never the full dataset
movers = R.gainers_losers(day_df, 10)
def _mover_table(rows):
    return pd.DataFrame([{"Symbol": r["symbol"], "Close": fmt_rs(r["close"]),
                          "Change": fmt_pct(r["change_pct"], signed=True),
                          "Volume": f"{r['volume']:,.0f}"} for r in rows])
g, l = st.columns(2)
with g: st.write("**Gainers**"); st.dataframe(_mover_table(movers["gainers"]), use_container_width=True, hide_index=True)
with l: st.write("**Losers**"); st.dataframe(_mover_table(movers["losers"]), use_container_width=True, hide_index=True)
