import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from pages.common import get_context, demo_banner, require_data, picker_symbols, symbol_info
from src.analytics import returns as R
from src.portfolio.engine import price_wide

settings, store = get_context()
st.title("⚖️ Stock Comparison")
demo_banner(settings)
if not require_data(store):
    st.stop()

equities_only = st.toggle("Equities only (EQ series)", value=True,
                          help="The full bhavcopy universe includes NCDs/bonds and other instruments "
                               "(series NQ, GB, GS, TB…) that trade rarely and sit near face value. "
                               "Turn this off to include them.")
symbols = picker_symbols(store, equities_only)
picked = st.multiselect("Select 2–5 symbols", symbols, default=symbols[:2])
if len(picked) < 2:
    st.info("Pick at least two symbols to compare.")
    st.stop()

# --- Per-symbol availability: explains sparse / non-equity instruments ---
infos = {}
for sym in picked:
    info = symbol_info(store, sym)
    infos[sym] = info
    if info.get("rows"):
        note = ""
        if not info.get("is_equity"):
            note = (" — ⚠️ **Not an EQ equity** (series: " + ", ".join(info["series"]) + "). Instruments like "
                    "NCDs/bonds trade infrequently and stay near face value, so comparisons with shares can look flat.")
        elif info.get("coverage_pct", 1) < 0.5:
            note = " — ⚠️ Traded on less than half of all trading days (illiquid or listed/delisted mid-period)."
        st.caption(f"**{sym}**: {info['rows']:,} observations · traded on {info.get('coverage_pct', 0):.0%} of "
                   f"{info.get('total_days', 0):,} trading days · {info.get('start')} → {info.get('end')}{note}")
    else:
        st.warning(f"**{sym}**: no rows in the local dataset under this exact symbol.")

df = store.load(symbols=picked)
if df.empty:
    st.error("No data found for the selected symbols.")
    st.stop()
comp = R.compare_symbols(df, picked, settings.risk_free_rate)
st.caption(f"Common comparison range for the table below: {comp.get('common_start')} → {comp.get('common_end')} · "
           f"Missing observations per symbol within that range: {comp.get('missing_observations')}")

rows = []
for sym, s in comp["symbols"].items():
    rows.append({"Symbol": sym, "Total return": s["total_return"], "Ann. return": s["annualized_return"],
                 "Volatility": s["annualized_volatility"], "Sharpe": s["sharpe_ratio"], "Max drawdown": s["max_drawdown"]})
st.dataframe(pd.DataFrame(rows), use_container_width=True)

# --- Normalized chart: each symbol over ITS OWN trading dates ---
# (Forcing common dates here used to make rarely-traded instruments vanish.)
fig = go.Figure()
for sym in picked:
    sub = df[df["symbol"] == sym].sort_values("date")
    if len(sub) >= 2:
        norm = sub["close"] / sub["close"].iloc[0] * 100
        fig.add_trace(go.Scatter(x=sub["date"], y=norm, name=f"{sym} (from {sub['date'].min().date()})",
                                 mode="lines+markers" if len(sub) < 40 else "lines",
                                 marker=dict(size=4)))
fig.update_layout(title="Normalized price performance (each symbol = 100 at its own first observation shown)",
                  yaxis_title="Normalized price", hovermode="x unified")
st.plotly_chart(fig, use_container_width=True)
st.caption("Each line is normalized on the symbol's own available history, so rarely-traded instruments remain "
           "visible. The table, cumulative returns, and correlation below still use the common overlapping range.")

# --- Common-range analytics, only when a real overlap exists ---
wide = price_wide(df, picked)
common = wide.dropna(how="any")
if len(common) >= 5:
    rets = common.pct_change().dropna()
    cum = (1 + rets).cumprod() - 1
    st.plotly_chart(px.line(cum, title=f"Cumulative returns (common dates only, {len(common)} days)"),
                    use_container_width=True)
    corr = rets.corr()
    st.plotly_chart(px.imshow(corr, text_auto=".2f", title="Return correlation (common dates)",
                              color_continuous_scale="RdBu", zmin=-1, zmax=1), use_container_width=True)
    st.plotly_chart(px.histogram(rets, barmode="overlay", opacity=0.6,
                                 title="Daily return distributions (common dates)"), use_container_width=True)
else:
    st.info(f"Only {len(common)} overlapping trading days between these symbols — too few for meaningful "
            "cumulative-return, correlation, or distribution charts. This usually happens when one symbol is a "
            "rarely-traded instrument (e.g. an NCD/bond) or was listed/delisted mid-period.")
