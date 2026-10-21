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

# ---------- Shared formatting (one definition of how numbers/dates look) ----------
def fmt_pct(x, dp: int = 2, signed: bool = False) -> str:
    """Fraction -> percentage string. fmt_pct(0.0035) = '0.35%'."""
    if x is None: return "—"
    try:
        return f"{x:+.{dp}%}" if signed else f"{x:.{dp}%}"
    except Exception:
        return str(x)

def fmt_rs(x, dp: int = 2) -> str:
    if x is None: return "—"
    try: return f"₹{x:,.{dp}f}"
    except Exception: return str(x)

def fmt_num(x, dp: int = 2) -> str:
    if x is None: return "—"
    try: return f"{x:,.{dp}f}"
    except Exception: return str(x)

def fmt_date(d) -> str:
    """'2023-03-31' / date / Timestamp -> '31 Mar 2023' (never truncated)."""
    if d is None: return "—"
    try:
        import pandas as pd
        return pd.Timestamp(d).strftime("%d %b %Y")
    except Exception:
        return str(d)

METRIC_HELP = {
    "total_return": "Total price return over the selected period: last close ÷ first close − 1. Uses unadjusted closes, so splits/bonuses can distort it. Dividends are not included (price return, not total return).",
    "annualized_return": "Compound annual growth rate (CAGR): the constant yearly rate that connects the first and last close over the period.",
    "volatility": "Annualized volatility: standard deviation of daily returns × √252. Higher means larger day-to-day swings. It measures variability, not direction.",
    "sharpe": "Sharpe ratio: (annualized excess return over the risk-free rate) ÷ annualized volatility. Higher means more return per unit of risk taken. Uses the configured risk-free rate.",
    "max_drawdown": "Maximum drawdown: the largest peak-to-trough fall in price within the period. E.g. −25% means the price once fell 25% from an earlier peak to a later low.",
    "rsi": "Relative Strength Index (0–100): momentum of recent gains vs losses. Above 70 is conventionally 'overbought', below 30 'oversold' — descriptive zones, not buy/sell signals.",
    "macd": "MACD: difference between the 12-day and 26-day exponential moving averages; the signal line is its 9-day EMA. MACD above the signal line indicates strengthening short-term momentum.",
    "atr": "Average True Range: average daily price range (including gaps) over the window — a volatility measure in ₹.",
    "correlation": "Correlation of daily returns (−1 to +1): +1 = moved together, 0 = unrelated, −1 = moved oppositely. Computed on common trading dates only.",
}

@st.cache_resource(show_spinner=False)
def load_all_cached(processed_dir: str, metadata_dir: str):
    """Full dataset as ONE shared in-memory copy across sessions.
    (cache_resource, not cache_data: cache_data serializes a second copy of
    a 4.1M-row frame, which can exhaust small cloud containers.)"""
    from src.storage.store import MarketDataStore
    from pathlib import Path
    return MarketDataStore(Path(processed_dir), Path(metadata_dir)).load_all()

def equity_symbols(df) -> list[str]:
    """Symbols that trade in the EQ (equity) series."""
    if df.empty or "series" not in df.columns:
        return sorted(df["symbol"].unique().tolist()) if not df.empty else []
    return sorted(df.loc[df["series"] == "EQ", "symbol"].unique().tolist())

@st.cache_resource(show_spinner=False)
def _symbol_stats(metadata_dir: str):
    """Precomputed per-symbol stats (small JSON), if the store has them."""
    from src.reporting.overview_stats import read_symbol_stats
    return read_symbol_stats(metadata_dir)

def picker_symbols(store, equities_only: bool) -> list[str]:
    """Symbol list for pickers, optionally restricted to EQ equities.
    The full bhavcopy universe (7k+ symbols) includes NCDs/bonds and other
    instruments that trade rarely — the toggle keeps pickers usable.
    Uses precomputed symbol stats when available so the full dataset
    (642 MB in RAM) is never loaded just to build a dropdown."""
    stats = _symbol_stats(str(store.metadata_dir))
    if stats:
        syms = sorted(stats["symbols"].keys())
        if equities_only:
            syms = [s for s in syms if stats["symbols"][s].get("is_equity")]
        return syms
    if not equities_only:
        return store.available_symbols()
    df = load_all_cached(str(store.processed_dir), str(store.metadata_dir))
    return equity_symbols(df)

def symbol_info(store, symbol: str) -> dict:
    """Availability facts for one symbol, for honest UI captions."""
    stats = _symbol_stats(str(store.metadata_dir))
    if stats:
        entry = stats["symbols"].get(symbol)
        total_days = stats.get("total_days", 0)
        if not entry:
            return {"rows": 0, "total_days": total_days}
        return {"rows": entry["rows"], "total_days": total_days,
                "coverage_pct": entry["rows"] / total_days if total_days else 0.0,
                "series": entry["series"], "is_equity": entry["is_equity"],
                "start": entry["start"], "end": entry["end"]}
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

def trading_dates(store) -> list[str]:
    """All trading dates (ISO strings): precomputed stats if present."""
    stats = _symbol_stats(str(store.metadata_dir))
    if stats and stats.get("trading_dates"):
        return stats["trading_dates"]
    df = load_all_cached(str(store.processed_dir), str(store.metadata_dir))
    return [str(d.date()) for d in sorted(df["date"].unique())] if not df.empty else []
