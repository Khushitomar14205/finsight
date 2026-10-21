"""FinSight — AI-Powered NSE Market & Portfolio Research Agent.
Run: streamlit run app.py
All quantitative results are computed by deterministic Python modules in
src/. The LLM (when configured) only routes questions and explains tool
outputs. The app works fully for local analytics without an LLM key."""
import streamlit as st

st.set_page_config(page_title="FinSight — NSE Market & Portfolio Research", page_icon="📈", layout="wide")

pages = {
    "Market": [
        st.Page("pages/market_overview.py", title="Home / Market Overview", icon="🏠", default=True),
        st.Page("pages/market_explorer.py", title="Historical Market Explorer", icon="🔍"),
        st.Page("pages/stock_analysis.py", title="Individual Stock Analysis", icon="📊"),
        st.Page("pages/stock_comparison.py", title="Stock Comparison", icon="⚖️"),
    ],
    "Portfolio": [
        st.Page("pages/portfolio_builder.py", title="Portfolio Builder", icon="🧺"),
        st.Page("pages/portfolio_analysis.py", title="Portfolio Performance & Risk", icon="🛡️"),
        st.Page("pages/portfolio_optimization.py", title="Optimization & What-If", icon="🎯"),
    ],
    "Intelligence": [
        st.Page("pages/ai_assistant.py", title="AI Research Assistant", icon="🤖"),
        st.Page("pages/ml_experiments.py", title="Machine Learning Experiments", icon="🧪"),
    ],
    "System": [
        st.Page("pages/data_quality.py", title="Data Quality & Ingestion", icon="✅"),
        st.Page("pages/settings.py", title="Settings & Data Sources", icon="⚙️"),
    ],
}
st.navigation(pages).run()
