"""Deterministic return / risk analytics. All numbers come from these
functions — the LLM never computes or invents them.

Methodology notes:
- Returns are simple close-to-close returns on *unadjusted* bhavcopy prices
  unless adjusted data is supplied. Corporate actions (splits/bonuses) can
  therefore create artificial jumps; validation flags moves > 50%.
- Annualization uses sqrt/trading-days scaling (default 252)."""
from __future__ import annotations
import numpy as np
import pandas as pd

TRADING_DAYS = 252

def daily_returns(prices: pd.Series) -> pd.Series:
    return prices.dropna().pct_change().dropna()

def cumulative_returns(returns: pd.Series) -> pd.Series:
    return (1 + returns).cumprod() - 1

def total_return(prices: pd.Series) -> float:
    p = prices.dropna()
    if len(p) < 2: return 0.0
    return float(p.iloc[-1] / p.iloc[0] - 1)

def annualized_return(prices: pd.Series, trading_days: int = TRADING_DAYS) -> float:
    p = prices.dropna()
    if len(p) < 2: return 0.0
    years = (len(p) - 1) / trading_days
    if years <= 0: return 0.0
    return float((p.iloc[-1] / p.iloc[0]) ** (1 / years) - 1)

def annualized_volatility(returns: pd.Series, trading_days: int = TRADING_DAYS) -> float:
    if len(returns) < 2: return 0.0
    return float(returns.std(ddof=1) * np.sqrt(trading_days))

def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0, trading_days: int = TRADING_DAYS) -> float:
    if len(returns) < 2: return 0.0
    daily_rf = risk_free_rate / trading_days
    excess = returns - daily_rf
    sd = excess.std(ddof=1)
    if sd == 0 or np.isnan(sd): return 0.0
    return float(excess.mean() / sd * np.sqrt(trading_days))

def max_drawdown(prices: pd.Series) -> dict:
    p = prices.dropna()
    if len(p) < 2: return {"max_drawdown": 0.0, "peak_date": None, "trough_date": None}
    running_max = p.cummax()
    dd = p / running_max - 1
    trough_idx = dd.idxmin()
    peak_idx = p.loc[:trough_idx].idxmax()
    return {"max_drawdown": float(dd.min()),
            "peak_date": str(pd.Timestamp(peak_idx).date()),
            "trough_date": str(pd.Timestamp(trough_idx).date())}

def rolling_volatility(returns: pd.Series, window: int = 21, trading_days: int = TRADING_DAYS) -> pd.Series:
    return returns.rolling(window).std(ddof=1) * np.sqrt(trading_days)

def stock_summary(df_symbol: pd.DataFrame, risk_free_rate: float = 0.0,
                  trading_days: int = TRADING_DAYS) -> dict:
    """df_symbol: rows for one symbol with date/close/volume."""
    df = df_symbol.sort_values("date")
    prices = df.set_index("date")["close"]
    rets = daily_returns(prices)
    mdd = max_drawdown(prices)
    out = {
        "symbol": str(df["symbol"].iloc[0]) if len(df) else None,
        "observations": int(len(df)),
        "start": str(df["date"].min().date()) if len(df) else None,
        "end": str(df["date"].max().date()) if len(df) else None,
        "first_close": float(prices.iloc[0]) if len(prices) else None,
        "last_close": float(prices.iloc[-1]) if len(prices) else None,
        "total_return": total_return(prices),
        "annualized_return": annualized_return(prices, trading_days),
        "annualized_volatility": annualized_volatility(rets, trading_days),
        "sharpe_ratio": sharpe_ratio(rets, risk_free_rate, trading_days),
        "max_drawdown": mdd["max_drawdown"],
        "drawdown_peak_date": mdd["peak_date"],
        "drawdown_trough_date": mdd["trough_date"],
        "best_day": {"date": str(rets.idxmax().date()), "return": float(rets.max())} if len(rets) else None,
        "worst_day": {"date": str(rets.idxmin().date()), "return": float(rets.min())} if len(rets) else None,
        "avg_daily_volume": float(df["volume"].mean()) if "volume" in df and len(df) else None,
    }
    return out

def gainers_losers(df_day: pd.DataFrame, n: int = 10) -> dict:
    """df_day: all symbols for one date with prev_close available."""
    d = df_day.dropna(subset=["prev_close", "close"]).copy()
    d = d[d["prev_close"] > 0]
    d["change_pct"] = d["close"] / d["prev_close"] - 1
    cols = ["symbol", "close", "prev_close", "change_pct", "volume"]
    gainers = d.nlargest(n, "change_pct")[cols].to_dict("records")
    losers = d.nsmallest(n, "change_pct")[cols].to_dict("records")
    return {"gainers": gainers, "losers": losers}

def correlation_matrix(returns_wide: pd.DataFrame) -> pd.DataFrame:
    return returns_wide.corr()

def covariance_matrix(returns_wide: pd.DataFrame, trading_days: int = TRADING_DAYS) -> pd.DataFrame:
    return returns_wide.cov() * trading_days

def compare_symbols(store_df: pd.DataFrame, symbols: list[str], risk_free_rate: float = 0.0) -> dict:
    """Comparison on the common available date range; missing obs reported."""
    out = {"symbols": {}, "common_start": None, "common_end": None, "missing_observations": {}}
    frames = {}
    for s in symbols:
        sub = store_df[store_df["symbol"] == s.upper()].sort_values("date")
        frames[s.upper()] = sub
        out["missing_observations"][s.upper()] = 0
    if not frames or any(f.empty for f in frames.values()):
        out["note"] = "One or more symbols have no data in the local dataset."
        return out
    common_start = max(f["date"].min() for f in frames.values())
    common_end = min(f["date"].max() for f in frames.values())
    out["common_start"], out["common_end"] = str(common_start.date()), str(common_end.date())
    wide = None
    for s, f in frames.items():
        sub = f[(f["date"] >= common_start) & (f["date"] <= common_end)]
        out["symbols"][s] = stock_summary(sub, risk_free_rate)
        p = sub.set_index("date")["close"].rename(s)
        wide = p.to_frame() if wide is None else wide.join(p, how="outer")
    total_days = len(wide)
    for s in symbols:
        out["missing_observations"][s.upper()] = int(total_days - wide[s.upper()].notna().sum())
    # Correlation uses complete cases on common dates only — the SAME
    # definition as the Stock Comparison page and the correlation tool, so
    # numbers agree across modules. Needs >= 3 shared observations.
    rets = wide.pct_change().dropna(how="any")
    out["correlation"] = correlation_matrix(rets).round(4).to_dict() if len(rets) >= 3 else {}
    out["correlation_observations"] = int(len(rets))
    return out
