import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from pages.common import get_context, demo_banner, require_data
from src.analytics import returns as R
from src.technical_indicators import indicators as TI

settings, store = get_context()
st.title("📊 Individual Stock Analysis")
demo_banner(settings)
if not require_data(store):
    st.stop()

symbols = store.available_symbols()
symbol = st.selectbox("Symbol", symbols)
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
m[0].metric("Total return", f"{s['total_return']:.2%}")
m[1].metric("Annualized return", f"{s['annualized_return']:.2%}")
m[2].metric("Volatility (ann.)", f"{s['annualized_volatility']:.2%}")
m[3].metric("Sharpe (rf={:.1%})".format(settings.risk_free_rate), f"{s['sharpe_ratio']:.2f}")
st.write(f"Max drawdown **{s['max_drawdown']:.2%}** (peak {s['drawdown_peak_date']} → trough {s['drawdown_trough_date']}) · "
         f"Best day **{s['best_day']['return']:.2%}** on {s['best_day']['date']} · Worst day **{s['worst_day']['return']:.2%}** on {s['worst_day']['date']}")

st.subheader("Technical indicators")
cc1, cc2, cc3 = st.columns(3)
sma_w = cc1.number_input("SMA window", 2, 200, 20)
rsi_w = cc2.number_input("RSI window", 2, 50, 14)
bb_w = cc3.number_input("Bollinger window", 2, 100, 20)
try:
    ind = TI.add_indicators(df, sma_windows=(int(sma_w), 50), rsi_window=int(rsi_w), bb_window=int(bb_w))
except ValueError as e:
    st.error(str(e)); st.stop()

fig = go.Figure()
fig.add_trace(go.Scatter(x=ind["date"], y=ind["close"], name="Close"))
fig.add_trace(go.Scatter(x=ind["date"], y=ind[f"sma_{int(sma_w)}"], name=f"SMA {int(sma_w)}"))
fig.add_trace(go.Scatter(x=ind["date"], y=ind["sma_50"], name="SMA 50"))
fig.add_trace(go.Scatter(x=ind["date"], y=ind["bb_upper"], name="BB upper", line=dict(dash="dot")))
fig.add_trace(go.Scatter(x=ind["date"], y=ind["bb_lower"], name="BB lower", line=dict(dash="dot")))
fig.update_layout(title=f"{symbol} — Price, moving averages, Bollinger Bands")
st.plotly_chart(fig, use_container_width=True)

f2 = go.Figure()
f2.add_trace(go.Scatter(x=ind["date"], y=ind["rsi"], name=f"RSI {int(rsi_w)}"))
f2.add_hline(y=70, line_dash="dot"); f2.add_hline(y=30, line_dash="dot")
f2.update_layout(title="RSI")
st.plotly_chart(f2, use_container_width=True)

f3 = go.Figure()
f3.add_trace(go.Scatter(x=ind["date"], y=ind["macd"], name="MACD"))
f3.add_trace(go.Scatter(x=ind["date"], y=ind["macd_signal"], name="Signal"))
f3.update_layout(title="MACD")
st.plotly_chart(f3, use_container_width=True)

prices = df.set_index("date")["close"]
rets = R.daily_returns(prices)
roll = R.rolling_volatility(rets, 21, settings.trading_days_per_year)
st.plotly_chart(go.Figure([go.Scatter(x=roll.index, y=roll.values, name="21-day rolling volatility (ann.)")]),
                use_container_width=True)
st.caption("Indicators are descriptive statistics of historical prices — not predictions or buy/sell signals. "
           "Returns use unadjusted closes; splits/bonuses can distort them (see Data Quality).")
st.dataframe(ind.tail(30), use_container_width=True)
