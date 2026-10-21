"""Shared UI context: settings, store, demo-data warning, empty states."""
import sys
from pathlib import Path
import streamlit as st
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config.settings import get_settings
from src.storage.store import MarketDataStore

@st.cache_resource
def get_context():
    settings = get_settings()
    store = MarketDataStore(settings.processed_dir, settings.metadata_dir)
    return settings, store

def demo_banner(settings):
    marker = settings.metadata_dir / "DEMO_DATA.txt"
    if marker.exists():
        st.warning("⚠️ The loaded dataset is **synthetic demo data** generated for development/testing — "
                   "it is NOT real NSE market data. Ingest real bhavcopy data (Data Quality page) for real analysis.")

def require_data(store) -> bool:
    if not store.coverage().get("has_data"):
        st.info("No local market data yet. Options:\n\n"
                "1. **Data Quality & Ingestion** page — download bhavcopy for a date range, or upload an official bhavcopy file.\n"
                "2. For trying the interface only, generate clearly-labelled synthetic demo data: "
                "`python scripts/generate_demo_data.py`\n\n"
                "All analytics below activate automatically once data exists.")
        return False
    return True

def get_portfolio_weights() -> dict:
    return st.session_state.get("portfolio_weights", {})
