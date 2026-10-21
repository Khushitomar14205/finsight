import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from pages.common import (get_context, demo_banner, require_data, picker_symbols, symbol_info,
                          fmt_pct, fmt_rs, fmt_num, fmt_date, METRIC_HELP)
from src.analytics import returns as R
from src.technical_indicators import indicators as TI

settings, store = get_context()
st.title("📊 Individual Stock Analysis")
demo_banner(settings)
if not require_data(store):
    st.stop()

equities_only = st.toggle("Equities only (EQ series)", value=True,
                          help="Off = include NCDs/bonds and other bhavcopy instruments, which trade rarely.")
symbols = picker_symbols(store, equities_only)
symbol = st.selectbox("Symbol", symbols)
_info = symbol_info(store, symbol)
if _info.get("rows") and not _info.get("is_equity"):
    st.warning(f"**{symbol}** is a non-equity instrument (series: {', '.join(_info.get('series', []))}) that traded on "
               f"only {_info.get('coverage_pct', 0):.0%} of trading days — indicator analysis may be unavailable or misleading.")
df_all = store.load(symbols=[symbol])
c1, c2 = st.columns(2)
start = c1.date_input("Start", value=df_all["date"].min())
end = c2.date_input("End", value=df_all["date"].max())
df = df_all[(df_all["date"] >= pd.Timestamp(start)) & (df_all["date"] <= pd.Timestamp(end))].sort_values("date")
if len(df) < 60:
    st.warning(f"Only {len(df)} observations in range — indicators need more history (SMA-50 etc.). Widen the range.")
    st.stop()

s = R.stock_summary(df, settings.risk_free_rate, settings.trading_days_per_year)
m = st.columns(4)
m[0].metric("Total return", fmt_pct(s["total_return"]), help=METRIC_HELP["total_return"])
m[1].metric("Annualized return", fmt_pct(s["annualized_return"]), help=METRIC_HELP["annualized_return"])
m[2].metric("Volatility (ann.)", fmt_pct(s["annualized_volatility"]), help=METRIC_HELP["volatility"])
m[3].metric("Sharpe (rf={:.1%})".format(settings.risk_free_rate), fmt_num(s["sharpe_ratio"]), help=METRIC_HELP["sharpe"])
st.write(f"Max drawdown **{fmt_pct(s['max_drawdown'])}** (peak {fmt_date(s['drawdown_peak_date'])} → trough "
         f"{fmt_date(s['drawdown_trough_date'])}) · Best day **{fmt_pct(s['best_day']['return'], signed=True)}** on "
         f"{fmt_date(s['best_day']['date'])} · Worst day **{fmt_pct(s['worst_day']['return'], signed=True)}** on "
         f"{fmt_date(s['worst_day']['date'])}")
st.caption("All metrics use unadjusted close-to-close **price returns** (dividends excluded) on NSE bhavcopy data. "
           "Hover the ⓘ on any metric for its definition.")

st.subheader("Technical indicators")
cc1, cc2, cc3 = st.columns(3)
sma_w = int(cc1.number_input("SMA window", 2, 200, 20))
rsi_w = int(cc2.number_input("RSI window", 2, 50, 14))
bb_w = int(cc3.number_input("Bollinger window", 2, 100, 20))
try:
    ind = TI.add_indicators(df, sma_windows=(sma_w, 50), rsi_window=rsi_w, bb_window=bb_w)
except ValueError as e:
    st.error(str(e)); st.stop()

last = ind.iloc[-1]
lv = st.columns(6)
lv[0].metric("Latest close", fmt_rs(last["close"]), help=f"As of {fmt_date(last['date'])}")
lv[1].metric(f"RSI ({rsi_w})", fmt_num(last["rsi"], 1), help=METRIC_HELP["rsi"])
lv[2].metric("MACD", fmt_num(last["macd"]), help=METRIC_HELP["macd"])
lv[3].metric(f"SMA {sma_w}", fmt_rs(last[f"sma_{sma_w}"]))
lv[4].metric("SMA 50", fmt_rs(last["sma_50"]))
lv[5].metric("ATR (14)", fmt_rs(last["atr"]), help=METRIC_HELP["atr"])

# --- Factual technical summary (descriptive only, no signals) ---
def _vs(price, avg):
    return "above" if price >= avg else "below"
bb_range = (last["bb_upper"] - last["bb_lower"]) if pd.notna(last["bb_upper"]) else None
bb_pos = ((last["close"] - last["bb_lower"]) / bb_range) if bb_range else None
rsi_zone = ("above 70 — the conventionally 'overbought' zone" if last["rsi"] >= 70 else
            "below 30 — the conventionally 'oversold' zone" if last["rsi"] <= 30 else "in the neutral 30–70 zone")
macd_rel = "above" if last["macd"] >= last["macd_signal"] else "below"
summary = (
    f"**Technical summary for {symbol} (as of {fmt_date(last['date'])}):** the close of {fmt_rs(last['close'])} is "
    f"{_vs(last['close'], last[f'sma_{sma_w}'])} its {sma_w}-day average ({fmt_rs(last[f'sma_{sma_w}'])}) and "
    f"{_vs(last['close'], last['sma_50'])} its 50-day average ({fmt_rs(last['sma_50'])}). "
    f"RSI({rsi_w}) is {fmt_num(last['rsi'], 1)}, {rsi_zone}. MACD ({fmt_num(last['macd'])}) is {macd_rel} its signal "
    f"line ({fmt_num(last['macd_signal'])}), indicating {'strengthening' if macd_rel == 'above' else 'weakening'} "
    f"short-term momentum. The price sits at {bb_pos:.0%} of its Bollinger Band width "
    f"({fmt_rs(last['bb_lower'])} – {fmt_rs(last['bb_upper'])}). "
    f"These are descriptive statistics of past prices — not predictions and not buy/sell recommendations.")
st.info(summary)

fig = go.Figure()
fig.add_trace(go.Scatter(x=ind["date"], y=ind["close"], name="Close"))
fig.add_trace(go.Scatter(x=ind["date"], y=ind[f"sma_{sma_w}"], name=f"SMA {sma_w}"))
fig.add_trace(go.Scatter(x=ind["date"], y=ind["sma_50"], name="SMA 50"))
fig.add_trace(go.Scatter(x=ind["date"], y=ind["bb_upper"], name="BB upper", line=dict(dash="dot")))
fig.add_trace(go.Scatter(x=ind["date"], y=ind["bb_lower"], name="BB lower", line=dict(dash="dot")))
fig.update_layout(title=f"{symbol} — Price, moving averages, Bollinger Bands ({bb_w}, 2σ)",
                  yaxis_title="Price (₹)", hovermode="x unified")
st.plotly_chart(fig, use_container_width=True)

f2 = go.Figure()
f2.add_trace(go.Scatter(x=ind["date"], y=ind["rsi"], name=f"RSI {rsi_w}"))
f2.add_hline(y=70, line_dash="dot", annotation_text="70 · overbought zone")
f2.add_hline(y=30, line_dash="dot", annotation_text="30 · oversold zone")
f2.update_layout(title=f"RSI ({rsi_w}) — 0 to 100 scale", yaxis=dict(range=[0, 100], title="RSI"),
                 hovermode="x unified")
st.plotly_chart(f2, use_container_width=True)

f3 = go.Figure()
colors = ["#2ca02c" if v >= 0 else "#d62728" for v in ind["macd_hist"]]
f3.add_trace(go.Bar(x=ind["date"], y=ind["macd_hist"], name="Histogram", marker_color=colors, opacity=0.5))
f3.add_trace(go.Scatter(x=ind["date"], y=ind["macd"], name="MACD"))
f3.add_trace(go.Scatter(x=ind["date"], y=ind["macd_signal"], name="Signal"))
f3.add_hline(y=0, line_width=1)
f3.update_layout(title="MACD (12, 26, 9) with histogram", yaxis_title="MACD", hovermode="x unified")
st.plotly_chart(f3, use_container_width=True)

prices = df.set_index("date")["close"]
rets = R.daily_returns(prices)
roll = R.rolling_volatility(rets, 21, settings.trading_days_per_year)
f4 = go.Figure([go.Scatter(x=roll.index, y=roll.values, name="21-day rolling volatility (ann.)")])
f4.update_layout(title="Rolling volatility (21-day, annualized)", yaxis=dict(tickformat=".0%", title="Volatility"),
                 hovermode="x unified")
st.plotly_chart(f4, use_container_width=True)
st.caption("Indicators are descriptive statistics of historical prices — not predictions or buy/sell signals. "
           "Returns use unadjusted closes; splits/bonuses can distort them (see Data Quality).")

st.subheader("Latest indicator values (30 most recent sessions)")
cols = ["date", "close", f"sma_{sma_w}", "sma_50", "ema_20", "rsi", "macd", "macd_signal",
        "bb_upper", "bb_lower", "atr", "volume"]
tail = ind.tail(30)[[c for c in cols if c in ind.columns]].copy()
tail["date"] = pd.to_datetime(tail["date"]).dt.strftime("%Y-%m-%d")
st.dataframe(tail.round(2), use_container_width=True, hide_index=True)
