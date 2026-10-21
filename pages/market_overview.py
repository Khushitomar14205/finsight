import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from pages.common import get_context, demo_banner, require_data, load_all_cached
from src.analytics import returns as R
from src.data_processing.validation import validate_dataframe
from src.data_ingestion.downloader import BhavcopyDownloader
from src.reporting.overview_stats import read_overview_stats

settings, store = get_context()

# ---------- Header ----------
st.title("📈 FinSight")
st.markdown("#### AI-Powered NSE Market & Portfolio Research Agent")
st.markdown("Historical Indian equity research on **real NSE bhavcopy data** — explore markets, analyse stocks, "
            "build portfolios, and ask questions in plain English. Every number is computed by deterministic "
            "Python tools; the AI only routes and explains.")
st.caption("Source: NSE bhavcopy (official end-of-day files) · Prices are historical closes, **not live prices** · "
           "Informational research only — no investment advice, no trade execution.")
demo_banner(settings)

cov = store.coverage()
if not require_data(store):
    st.stop()

# ---------- KPI cards ----------
kpis = st.columns(4)
cards = [
    ("Latest trading day", cov.get("latest_available_date") or "—", "Most recent bhavcopy in the local dataset"),
    ("Symbols covered", f"{cov.get('symbols', 0):,}", "Across equities, NCDs/bonds & other instruments"),
    ("Total records", f"{cov.get('rows', 0):,}", "One row per symbol per trading day"),
    ("History", f"{cov.get('date_min') or '—'} → {cov.get('date_max') or '—'}", "Full available coverage"),
]
for col, (label, value, sub) in zip(kpis, cards):
    with col:
        with st.container(border=True):
            st.metric(label, value)
            st.caption(sub)

# ---------- Numbers: precomputed stats if available, else compute live ----------
stats = read_overview_stats(settings.metadata_dir)
if stats and stats.get("generated_for_latest_date") == cov.get("latest_available_date"):
    pulse = stats["pulse"]
    yearly = pd.DataFrame(stats["yearly"])
    quality = stats["quality"]
else:
    # Fallback (fresh local data without precomputed stats): compute live.
    df = load_all_cached(str(store.processed_dir), str(store.metadata_dir))
    latest_date = df["date"].max()
    day = df[df["date"] == latest_date].dropna(subset=["prev_close", "close"]).copy()
    day = day[day["prev_close"] > 0]
    day["change_pct"] = day["close"] / day["prev_close"] - 1
    movers = R.gainers_losers(day, 10)
    pulse = {"date": str(latest_date.date()),
             "advancers": int((day["change_pct"] > 0).sum()),
             "decliners": int((day["change_pct"] < 0).sum()),
             "unchanged": int((day["change_pct"] == 0).sum()),
             "traded": int(len(day)),
             "median_change_pct": float(day["change_pct"].median()),
             "total_turnover": float(day["turnover"].sum()),
             "gainers": movers["gainers"], "losers": movers["losers"]}
    yearly = (df.assign(year=df["date"].dt.year).groupby("year")
                .agg(records=("close", "size"), symbols=("symbol", "nunique")).reset_index())
    rep = validate_dataframe(df)
    quality = {"duplicates": rep.duplicate_symbol_date, "ohlc_violations": rep.ohlc_violations,
               "nonpositive_prices": rep.nonpositive_prices, "negative_volume": rep.negative_volume,
               "extreme_moves": rep.extreme_daily_moves, "missing_sessions": rep.missing_weekday_sessions}

# ---------- Market pulse ----------
adv, dec, unch = pulse["advancers"], pulse["decliners"], pulse["unchanged"]
with st.container(border=True):
    st.subheader(f"🫀 Market pulse — {pulse['date']} (latest available day)")
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Advancers", f"{adv:,}", delta=f"{adv / max(pulse['traded'],1):.0%} of traded symbols")
    p2.metric("Decliners", f"{dec:,}", delta=f"{dec / max(pulse['traded'],1):.0%} of traded symbols", delta_color="inverse")
    p3.metric("Median change", f"{pulse['median_change_pct']:.2%}")
    p4.metric("Total turnover", f"₹{pulse['total_turnover'] / 1e7:,.0f} Cr")
    breadth = go.Figure()
    breadth.add_trace(go.Bar(y=["Market breadth"], x=[adv], orientation="h", name="Advancers",
                             marker_color="#2ca02c", text=[f"{adv:,} up"], textposition="inside"))
    breadth.add_trace(go.Bar(y=["Market breadth"], x=[unch], orientation="h", name="Unchanged",
                             marker_color="#bdbdbd", text=[f"{unch:,} flat"], textposition="inside"))
    breadth.add_trace(go.Bar(y=["Market breadth"], x=[dec], orientation="h", name="Decliners",
                             marker_color="#d62728", text=[f"{dec:,} down"], textposition="inside"))
    breadth.update_layout(barmode="stack", height=110, margin=dict(l=10, r=10, t=10, b=10),
                          legend=dict(orientation="h", y=-0.35), xaxis_title="")
    st.plotly_chart(breadth, use_container_width=True)

    if pulse["gainers"] and pulse["losers"]:
        top_g, top_l = pulse["gainers"][0], pulse["losers"][0]
        g1, g2 = st.columns(2)
        with g1:
            with st.container(border=True):
                st.markdown(f"🚀 **Top gainer: {top_g['symbol']}**")
                st.markdown(f"### {top_g['change_pct']:+.2%}")
                st.caption(f"Close ₹{top_g['close']:,.2f} · Volume {top_g['volume']:,.0f}")
        with g2:
            with st.container(border=True):
                st.markdown(f"📉 **Top loser: {top_l['symbol']}**")
                st.markdown(f"### {top_l['change_pct']:+.2%}")
                st.caption(f"Close ₹{top_l['close']:,.2f} · Volume {top_l['volume']:,.0f}")
    t1, t2 = st.columns(2)
    with t1:
        st.markdown("**Top 10 gainers**")
        st.dataframe(pulse["gainers"], use_container_width=True, hide_index=True)
    with t2:
        st.markdown("**Top 10 losers**")
        st.dataframe(pulse["losers"], use_container_width=True, hide_index=True)

# ---------- Coverage growth + data quality ----------
c_left, c_right = st.columns([3, 2])
with c_left:
    with st.container(border=True):
        st.subheader("📚 Coverage by year")
        fig = px.bar(yearly, x="year", y="records", text="symbols",
                     labels={"records": "Records", "year": "Year"},
                     title="Records per year (labels = distinct symbols that year)")
        fig.update_traces(textposition="outside")
        fig.update_layout(height=340, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)
with c_right:
    with st.container(border=True):
        st.subheader("✅ Data quality at a glance")
        st.markdown(f"**Duplicates:** {quality['duplicates']} · **OHLC violations:** {quality['ohlc_violations']}")
        st.markdown(f"**Non-positive prices:** {quality['nonpositive_prices']} · **Negative volume:** {quality['negative_volume']}")
        st.markdown(f"**Extreme moves flagged:** {quality['extreme_moves']} "
                    "(unadjusted prices — splits/bonuses; flagged, never altered)")
        st.markdown(f"**Weekday sessions with no file:** {quality['missing_sessions']} "
                    "(market holidays — see the ingestion log)")
        st.page_link("pages/data_quality.py", label="Full quality report & ingestion log →", icon="✅")
    with st.container(border=True):
        st.subheader("🔄 Ingestion status")
        log = BhavcopyDownloader(settings, store).read_log_display()
        if log:
            log_df = pd.DataFrame(log)
            st.markdown(f"**{int((log_df['status'] == 'success').sum() + (log_df['status'] == 'local_success').sum()):,}** "
                        f"days downloaded · **{int((log_df['status'] == 'holiday').sum())}** market holidays · "
                        f"**{int((log_df['status'] == 'failed').sum())}** failed (retryable)")
            st.caption(f"Latest log entry: {log_df.iloc[-1]['date']} — {log_df.iloc[-1]['status']}")
        else:
            st.markdown("No ingestion runs logged yet.")
        st.page_link("pages/data_quality.py", label="Ingest more data →", icon="⬇️")

# ---------- Quick access ----------
st.subheader("🚀 Quick access")
q1, q2, q3, q4 = st.columns(4)
q1.page_link("pages/market_explorer.py", label="Explore historical prices", icon="🔍")
q2.page_link("pages/stock_analysis.py", label="Analyse a stock (indicators)", icon="📊")
q3.page_link("pages/stock_comparison.py", label="Compare stocks", icon="⚖️")
q4.page_link("pages/portfolio_builder.py", label="Build a portfolio", icon="🧺")
q5, q6, q7, q8 = st.columns(4)
q5.page_link("pages/portfolio_analysis.py", label="Portfolio risk & performance", icon="🛡️")
q6.page_link("pages/portfolio_optimization.py", label="Optimize & what-if", icon="🎯")
q7.page_link("pages/ai_assistant.py", label="Ask the AI assistant", icon="🤖")
q8.page_link("pages/ml_experiments.py", label="ML experiments", icon="🧪")
