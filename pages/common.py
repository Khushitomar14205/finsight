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

@st.cache_data(ttl=3600, show_spinner=False)
def load_all_cached(processed_dir: str, metadata_dir: str):
    """Full dataset, cached for an hour — several pages need it and
    re-reading 4.1M rows on every interaction is wasteful."""
    from src.storage.store import MarketDataStore
    from pathlib import Path
    return MarketDataStore(Path(processed_dir), Path(metadata_dir)).load_all()

def equity_symbols(df) -> list[str]:
    """Symbols that trade in the EQ (equity) series."""
    if df.empty or "series" not in df.columns:
        return sorted(df["symbol"].unique().tolist()) if not df.empty else []
    return sorted(df.loc[df["series"] == "EQ", "symbol"].unique().tolist())

def picker_symbols(store, equities_only: bool) -> list[str]:
    """Symbol list for pickers, optionally restricted to EQ equities.
    The full bhavcopy universe (7k+ symbols) includes NCDs/bonds and other
    instruments that trade rarely — the toggle keeps pickers usable."""
    if not equities_only:
        return store.available_symbols()
    df = load_all_cached(str(store.processed_dir), str(store.metadata_dir))
    return equity_symbols(df)

def symbol_info(store, symbol: str) -> dict:
    """Availability facts for one symbol, for honest UI captions."""
    df = load_all_cached(str(store.processed_dir), str(store.metadata_dir))
    sub = df[df["symbol"] == symbol]
    total_days = df["date"].nunique() if not df.empty else 0
    if sub.empty:
        return {"rows": 0, "total_days": total_days}
    return {"rows": int(len(sub)), "total_days": int(total_days),
            "coverage_pct": float(len(sub) / total_days) if total_days else 0.0,
            "series": sorted(sub["series"].unique().tolist()),
            "is_equity": "EQ" in set(sub["series"].unique()),
            "start": str(sub["date"].min().date()), "end": str(sub["date"].max().date())}
