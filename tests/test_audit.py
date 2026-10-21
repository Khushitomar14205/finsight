"""Financial-calculation audit: every core metric checked against
hand-computed or stdlib-computed reference values."""
import statistics
import numpy as np
import pandas as pd
import pytest
from src.analytics import returns as R
from src.technical_indicators import indicators as TI

D = pd.bdate_range("2024-01-01", periods=400)

def prices(vals):
    return pd.Series(vals, index=D[:len(vals)], dtype=float)

def test_total_return_simple():
    assert R.total_return(prices([100, 110])) == pytest.approx(0.10)
    assert R.total_return(prices([100])) == 0.0          # insufficient history is safe
    assert R.total_return(prices([100, 50, 100])) == pytest.approx(0.0)

def test_annualized_return_is_cagr():
    g = 0.001
    vals = [100 * (1 + g) ** i for i in range(253)]      # exactly 252 intervals = 1 "year"
    assert R.annualized_return(prices(vals)) == pytest.approx((1 + g) ** 252 - 1, rel=1e-9)
    assert R.annualized_return(prices([100])) == 0.0

def test_volatility_matches_stdlib():
    rets = pd.Series([0.01, -0.01] * 50)
    expected = statistics.stdev(rets.tolist()) * np.sqrt(252)
    assert R.annualized_volatility(rets) == pytest.approx(expected, rel=1e-9)
    assert R.annualized_volatility(pd.Series([0.01])) == 0.0

def test_sharpe_matches_manual_formula():
    rng = np.random.default_rng(7)
    rets = pd.Series(rng.normal(0.0008, 0.011, 300))
    rf = 0.06
    excess = rets - rf / 252
    expected = excess.mean() / statistics.stdev(excess.tolist()) * np.sqrt(252)
    assert R.sharpe_ratio(rets, rf) == pytest.approx(expected, rel=1e-9)
    assert R.sharpe_ratio(pd.Series([0.001] * 50), 0.0) == 0.0   # zero variance guarded

def test_max_drawdown_hand_computed():
    p = prices([100, 120, 90, 110, 60, 130])
    out = R.max_drawdown(p)
    assert out["max_drawdown"] == pytest.approx(-0.5)     # 60 vs running peak 120
    assert out["peak_date"] == str(D[1].date())
    assert out["trough_date"] == str(D[4].date())

def test_rsi_extremes_and_flat():
    up = prices(list(range(100, 160)))
    down = prices(list(range(160, 100, -1)))
    flat = prices([100.0] * 60)
    assert TI.rsi(up).iloc[-1] == pytest.approx(100.0)
    assert TI.rsi(down).iloc[-1] == pytest.approx(0.0)
    assert TI.rsi(flat).iloc[-1] == pytest.approx(50.0)

def test_macd_constant_price_is_zero():
    m = TI.macd(prices([50.0] * 60))
    assert m["macd"].iloc[-1] == pytest.approx(0.0, abs=1e-9)
    assert m["macd_hist"].iloc[-1] == pytest.approx(0.0, abs=1e-9)

def test_bollinger_mid_is_sma_and_bands_symmetric():
    vals = [100 + 5 * np.sin(i / 3) for i in range(60)]
    b = TI.bollinger_bands(prices(vals), 20)
    assert b["bb_mid"].iloc[-1] == pytest.approx(np.mean(vals[-20:]))
    assert (b["bb_upper"] - b["bb_mid"]).iloc[-1] == pytest.approx((b["bb_mid"] - b["bb_lower"]).iloc[-1])

def test_sma_hand_computed():
    s = TI.sma(pd.Series([1., 2., 3., 4., 5.]), 3)
    assert s.iloc[-1] == pytest.approx(4.0) and np.isnan(s.iloc[1])

def test_correlation_identical_series_is_one():
    df = pd.DataFrame({"A": [0.01, -0.02, 0.03, 0.005], "B": [0.01, -0.02, 0.03, 0.005]})
    assert R.correlation_matrix(df).loc["A", "B"] == pytest.approx(1.0)

def test_gainers_losers_math():
    day = pd.DataFrame({"symbol": ["A", "B"], "close": [110., 90.], "prev_close": [100., 100.],
                        "volume": [1, 2]})
    out = R.gainers_losers(day, 1)
    assert out["gainers"][0]["symbol"] == "A" and out["gainers"][0]["change_pct"] == pytest.approx(0.10)
    assert out["losers"][0]["symbol"] == "B" and out["losers"][0]["change_pct"] == pytest.approx(-0.10)

def test_compare_symbols_correlation_uses_common_dates():
    """Both symbols fully overlapping -> correlation defined on shared dates."""
    idx = pd.bdate_range("2024-01-01", periods=60)
    a = 100 * (1 + pd.Series(np.random.default_rng(1).normal(0, 0.01, 60))).cumprod()
    df = pd.DataFrame({"date": list(idx) * 2,
                       "symbol": ["AAA"] * 60 + ["BBB"] * 60,
                       "close": list(a) + list(a), "volume": 1})
    out = R.compare_symbols(df, ["AAA", "BBB"])
    assert out["correlation"]["AAA"]["BBB"] == pytest.approx(1.0)
    assert out["common_start"] == str(idx[0].date()) and out["common_end"] == str(idx[-1].date())
