import numpy as np
import pandas as pd
import pytest
from src.analytics import returns as R
from src.technical_indicators import indicators as TI

def _prices(vals):
    idx = pd.bdate_range("2024-01-01", periods=len(vals))
    return pd.Series(vals, index=idx, dtype=float)

def test_total_return_known():
    assert R.total_return(_prices([100, 110])) == pytest.approx(0.10)

def test_max_drawdown_known():
    mdd = R.max_drawdown(_prices([100, 120, 60, 90]))
    assert mdd["max_drawdown"] == pytest.approx(-0.5)

def test_volatility_zero_for_constant():
    rets = pd.Series([0.0, 0.0, 0.0])
    assert R.annualized_volatility(rets) == 0.0

def test_sharpe_zero_for_constant_excess():
    rets = pd.Series([0.001] * 50)
    assert R.sharpe_ratio(rets, 0.0) == pytest.approx(0.0, abs=1e-9) or R.sharpe_ratio(rets, 0.0) == 0.0

def test_sma_known():
    s = pd.Series([1.0, 2.0, 3.0, 4.0])
    assert TI.sma(s, 2).iloc[-1] == pytest.approx(3.5)

def test_rsi_bounds(sample_df):
    df = sample_df[sample_df.symbol == "AAA"]
    r = TI.rsi(df["close"].reset_index(drop=True), 14).dropna()
    assert r.between(0, 100).all()

def test_indicator_validation():
    with pytest.raises(ValueError):
        TI.sma(pd.Series([1.0, 2.0]), 1)
    df = pd.DataFrame({"date": pd.bdate_range("2024-01-01", periods=5), "close": [1,2,3,4,5],
                       "high": [1,2,3,4,5], "low": [1,2,3,4,5], "volume": [1]*5})
    with pytest.raises(ValueError):
        TI.add_indicators(df)  # insufficient history for SMA50

def test_macd_fast_slow_validation():
    with pytest.raises(ValueError):
        TI.macd(pd.Series(range(50), dtype=float), fast=26, slow=12)

def test_stock_summary_runs(sample_df):
    s = R.stock_summary(sample_df[sample_df.symbol == "AAA"], risk_free_rate=0.06)
    assert s["observations"] == 300
    assert -1 <= s["max_drawdown"] <= 0
