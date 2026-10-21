"""Portfolio allocation experiments via scipy SLSQP — deterministic
numerical optimization, never LLM-invented weights.

All outputs are *historical-model* allocations computed from the sample
period. They are not recommendations or guarantees of future performance.
Constraints: fully invested (weights sum to 1), long-only by default,
configurable maximum weight per asset."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from ..portfolio.engine import validate_weights, price_wide, portfolio_risk, portfolio_performance

def _returns_wide(store_df, symbols):
    wide = price_wide(store_df, symbols).dropna(how="any")
    if len(wide) < 10:
        raise ValueError("Need at least 10 overlapping observations to optimize.")
    return wide.pct_change().dropna(), wide

def _optimize(store_df, symbols, objective, risk_free_rate=0.0, max_weight=1.0, trading_days=252):
    symbols = [s.upper() for s in symbols]
    if len(symbols) < 2: raise ValueError("Optimization needs at least 2 symbols.")
    if not (0 < max_weight <= 1): raise ValueError("max_weight must be in (0, 1].")
    rets, _ = _returns_wide(store_df, symbols)
    mu = rets.mean().values * trading_days
    cov = (rets.cov() * trading_days).values
    n = len(symbols)
    daily_rf = risk_free_rate  # already annualized here

    def neg_sharpe(w):
        var = w @ cov @ w
        if var <= 0: return 1e6
        return -((w @ mu - daily_rf) / np.sqrt(var))
    def variance(w):
        return float(w @ cov @ w)

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, max_weight)] * n
    x0 = np.repeat(1 / n, n)
    fun = neg_sharpe if objective == "max_sharpe" else variance
    res = minimize(fun, x0, method="SLSQP", bounds=bounds, constraints=constraints)
    if not res.success:
        raise RuntimeError(f"Optimization failed: {res.message}")
    weights = {s: float(round(w, 6)) for s, w in zip(symbols, res.x)}
    return validate_weights(weights)

def equal_weight(symbols: list[str]) -> dict[str, float]:
    symbols = [s.upper() for s in symbols]
    return {s: 1 / len(symbols) for s in symbols}

def minimum_variance(store_df, symbols, max_weight=1.0, **kw) -> dict:
    w = _optimize(store_df, symbols, "min_variance", max_weight=max_weight, **kw)
    return {"weights": w, "method": "minimum_variance (historical, sample period)",
            "performance": _clean_perf(store_df, w), "risk": _clean_risk(store_df, w)}

def maximum_sharpe(store_df, symbols, risk_free_rate=0.0, max_weight=1.0, **kw) -> dict:
    w = _optimize(store_df, symbols, "max_sharpe", risk_free_rate=risk_free_rate, max_weight=max_weight, **kw)
    return {"weights": w, "method": "maximum_sharpe (historical, sample period)",
            "performance": _clean_perf(store_df, w, risk_free_rate), "risk": _clean_risk(store_df, w, risk_free_rate)}

def _clean_perf(store_df, w, rf=0.0):
    p = portfolio_performance(store_df, w, risk_free_rate=rf)
    p.pop("series", None); return p

def _clean_risk(store_df, w, rf=0.0):
    return portfolio_risk(store_df, w, risk_free_rate=rf)

def what_if(store_df, base_weights: dict, new_weights: dict, risk_free_rate: float = 0.0) -> dict:
    """Compare current vs proposed allocation on historical metrics."""
    base_w, new_w = validate_weights(base_weights), validate_weights(new_weights)
    return {
        "base": {"weights": base_w, "performance": _clean_perf(store_df, base_w, risk_free_rate),
                 "risk": _clean_risk(store_df, base_w, risk_free_rate)},
        "proposed": {"weights": new_w, "performance": _clean_perf(store_df, new_w, risk_free_rate),
                     "risk": _clean_risk(store_df, new_w, risk_free_rate)},
        "note": "Historical comparison on the same sample period — not a forecast.",
    }
