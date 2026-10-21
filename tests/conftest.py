import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

@pytest.fixture
def sample_df():
    """Two symbols, 300 business days, deterministic synthetic prices."""
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2023-01-02", periods=300)
    frames = []
    for sym, p0 in (("AAA", 100.0), ("BBB", 200.0)):
        close = p0 * np.exp(np.cumsum(rng.normal(0.0004, 0.015, len(dates))))
        frames.append(pd.DataFrame({
            "date": dates, "symbol": sym, "series": "EQ",
            "open": close * 0.999, "high": close * 1.01, "low": close * 0.99,
            "close": close, "prev_close": np.concatenate([[close[0]], close[:-1]]),
            "last_price": close, "vwap": close, "volume": 1_000_000.0,
            "turnover": close * 1_000_000, "trades": 5000.0, "source_file": "test",
        }))
    return pd.concat(frames, ignore_index=True)
