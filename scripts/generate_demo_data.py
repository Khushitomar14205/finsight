"""Generate SYNTHETIC demo data for UI development and testing.

This data is randomly generated with a fixed seed. It is NOT real NSE
market data and must never be presented as such. The app shows a warning
banner whenever this dataset is the one loaded (data/metadata/DEMO_DATA.txt).
Real analysis should use ingested bhavcopy data (scripts/ingest.py)."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config.settings import get_settings
from src.storage.store import MarketDataStore

SYMBOLS = {  # symbol: (start_price, drift, vol) — synthetic parameters only
    "RELIANCE": (2400.0, 0.0004, 0.016), "TCS": (3400.0, 0.0003, 0.014),
    "HDFCBANK": (1650.0, 0.0002, 0.015), "INFY": (1450.0, 0.0003, 0.017),
    "SBIN": (620.0, 0.0005, 0.020), "TATAMOTORS": (650.0, 0.0006, 0.024),
    "ITC": (430.0, 0.0002, 0.011), "AXISBANK": (1050.0, 0.0003, 0.018),
}

def generate(start="2022-01-03", end="2024-12-31", seed=42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, end)
    rows = []
    for sym, (p0, drift, vol) in SYMBOLS.items():
        shocks = rng.normal(drift, vol, len(dates))
        close = p0 * np.exp(np.cumsum(shocks))
        open_ = close * (1 + rng.normal(0, 0.003, len(dates)))
        high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.004, len(dates))))
        low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.004, len(dates))))
        volume = rng.integers(200_000, 8_000_000, len(dates)).astype(float)
        prev = np.concatenate([[close[0]], close[:-1]])
        rows.append(pd.DataFrame({
            "date": dates, "symbol": sym, "series": "EQ", "open": open_, "high": high,
            "low": low, "close": close, "prev_close": prev, "last_price": close,
            "vwap": (high + low + close) / 3, "volume": volume,
            "turnover": volume * close, "trades": volume / 50,
            "source_file": "SYNTHETIC_DEMO_NOT_REAL_MARKET_DATA",
        }))
    return pd.concat(rows, ignore_index=True).sort_values(["date", "symbol"]).reset_index(drop=True)

def main():
    s = get_settings()
    store = MarketDataStore(s.processed_dir, s.metadata_dir)
    df = generate()
    info = store.save(df)
    (s.metadata_dir / "DEMO_DATA.txt").write_text(
        "WARNING: The currently stored dataset was generated synthetically by "
        "scripts/generate_demo_data.py for development/testing. It is NOT real "
        "NSE market data. Ingest real bhavcopy data to replace it.\n")
    print("Synthetic demo data saved:", info)

if __name__ == "__main__":
    main()
