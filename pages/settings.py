import streamlit as st
from pages.common import get_context

settings, store = get_context()
st.title("⚙️ Settings & Data Sources")

st.subheader("Configuration (secrets are never displayed)")
summary = settings.redacted_summary()
for k, v in summary.items():
    st.write(f"- **{k}:** {v}")

issues = settings.validate()
if issues:
    st.subheader("Configuration notes")
    for i in issues:
        st.warning(i)
else:
    st.success("Configuration is valid.")

st.subheader("How configuration works")
st.write("""
- Copy `.env.example` to `.env` and fill in your own keys. `.env` is git-ignored.
- `LLM_PROVIDER` can be `gemini`, `openai`, or `none`.
- **Fallback mode:** with no LLM key, every page except LLM-written explanations works —
  the AI assistant routes questions with deterministic rules and answers from the same tools.
- API keys are read from the environment only. They never appear in this UI, logs, or reports.
""")

st.subheader("Data sources")
st.write("""
- **Primary:** NSE bhavcopy archives (`nsearchives.nseindia.com`) — legacy format before mid-2024,
  UDiFF format after. The source adapter in `src/data_ingestion/downloader.py` isolates URL formats
  so it can be updated if NSE changes them.
- NSE may block automated downloads (403/404). Failures are logged with reasons on the
  Data Quality page; use the local-file ingestion path with officially downloaded files as recovery.
- No paid market-data API is required to analyse data already stored locally.
- There is **no live-price or news feed** configured by default, so the app never presents
  historical closes as live prices.
""")

st.subheader("Application settings")
st.write(f"- Risk-free rate (default for Sharpe): **{settings.risk_free_rate:.2%}** — override per analysis on the Portfolio page.")
st.write(f"- Trading days per year: **{settings.trading_days_per_year}**")
st.write(f"- Request timeout: **{settings.request_timeout_seconds}s** · Max retries: **{settings.max_retries}**")
