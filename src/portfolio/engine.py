"""Portfolio performance & risk engine.

Return methodology (explicit):
- Default: weights are rebalanced back to target every day, so portfolio
  daily return = sum_i w_i * r_i,t. This is the standard textbook method.
- Alternative (rebalance='none'): buy-and-hold — weights drift with prices;
  portfolio value = sum_i amount_i * (P_i,t / P_i,0).
Risk-free rate is configurable. Annualization: 252 trading days.
Historical risk is an estimate from the sample period, not a forecast."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from ..analytics.returns import annualized_volatility, sharpe_ratio, max_drawdown, TRADING_DAYS

def validate_weights(weights: dict[str, float]) -> dict[str, float]:
    if not weights:
        raise ValueError("Portfolio needs at least one holding.")
    for k, v in weights.items():
        if v < 0: raise ValueError(f"Negative weight for {k} not allowed in long-only portfolios.")
    total = sum(weights.values())
    if total <= 0: raise ValueError("Weights must sum to a positive value.")
    return {k.upper(): float(v) / total for k, v in weights.items()}

def weights_from_amounts(amounts: dict[str, float]) -> dict[str, float]:
    total = sum(amounts.values())
    if total <= 0: raise ValueError("Investment amounts must sum to a positive value.")
    return validate_weights({k: v / total for k, v in amounts.items()})

def parse_holdings_csv(path_or_buffer) -> dict[str, float]:
    """CSV with columns: symbol, weight  OR  symbol, amount. Validated."""
    df = pd.read_csv(path_or_buffer)
    cols = {c.lower().strip(): c for c in df.columns}
    if "symbol" not in cols: raise ValueError("Holdings CSV must have a 'symbol' column.")
    if "weight" in cols:
        return validate_weights(dict(zip(df[cols["symbol"]].astype(str).str.upper(), df[cols["weight"]].astype(float))))
    if "amount" in cols:
        return weights_from_amounts(dict(zip(df[cols["symbol"]].astype(str).str.upper(), df[cols["amount"]].astype(float))))
    raise ValueError("Holdings CSV must have either a 'weight' or an 'amount' column.")

def save_portfolio(weights: dict[str, float], name: str, directory: Path) -> Path:
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    safe = "".join(c for c in name if c.isalnum() or c in "-_ ").strip().replace(" ", "_")
    p = directory / f"{safe}.json"
    p.write_text(json.dumps({"name": name, "weights": validate_weights(weights)}, indent=2))
    return p

def load_portfolio(path) -> dict[str, float]:
    data = json.loads(Path(path).read_text())
    return validate_weights(data["weights"])

def price_wide(store_df: pd.DataFrame, symbols: list[str]) -> pd.DataFrame:
    sub = store_df[store_df["symbol"].isin([s.upper() for s in symbols])]
    wide = sub.pivot_table(index="date", columns="symbol", values="close").sort_index()
    return wide

def portfolio_series(store_df: pd.DataFrame, weights: dict[str, float],
                     initial_value: float = 100000.0, rebalance: str = "daily") -> pd.DataFrame:
    w = validate_weights(weights)
    wide = price_wide(store_df, list(w.keys()))
    missing = [s for s in w if s not in wide.columns]
    if missing: raise ValueError(f"No local price data for: {missing}")
    wide = wide.dropna(how="any")  # common trading calendar; count is reported by callers via len
    if len(wide) < 2: raise ValueError("Insufficient overlapping history for these holdings (need >= 2 common dates).")
    rets = wide.pct_change().dropna()
    wvec = np.array([w[s] for s in wide.columns])
    if rebalance == "daily":
        port_rets = pd.Series(rets.values @ wvec, index=rets.index, name="portfolio_return")
        value = initial_value * (1 + port_rets).cumprod()
        value = pd.concat([pd.Series([initial_value], index=[wide.index[0]]), value])
        value.index.name = "date"
    elif rebalance == "none":
        amounts = {s: initial_value * w[s] for s in wide.columns}
        value = sum(amounts[s] * wide[s] / wide[s].iloc[0] for s in wide.columns)
        value.index.name = "date"
        port_rets = value.pct_change().dropna().rename("portfolio_return")
    else:
        raise ValueError("rebalance must be 'daily' or 'none'")
    return pd.DataFrame({"value": value, "daily_return": port_rets.reindex(value.index)})

def portfolio_performance(store_df: pd.DataFrame, weights: dict[str, float],
                          initial_value: float = 100000.0, rebalance: str = "daily",
                          risk_free_rate: float = 0.0, benchmark_symbol: str | None = None,
                          trading_days: int = TRADING_DAYS) -> dict:
    w = validate_weights(weights)
    series = portfolio_series(store_df, w, initial_value, rebalance)
    rets = series["daily_return"].dropna()
    values = series["value"]
    mdd = max_drawdown(values)
    years = max((len(values) - 1) / trading_days, 1e-9)
    total = float(values.iloc[-1] / values.iloc[0] - 1)
    contrib = {}
    wide = price_wide(store_df, list(w.keys())).dropna(how="any")
    asset_rets = wide.pct_change().dropna()
    for s in w:
        contrib[s] = float(w[s] * asset_rets[s].mean() * len(asset_rets))  # approx: weight x cumulative mean contribution
    out = {
        "weights": w, "initial_value": initial_value, "final_value": float(values.iloc[-1]),
        "total_return": total,
        "annualized_return": float((1 + total) ** (1 / years) - 1) if total > -1 else -1.0,
        "observations": int(len(values)), "start": str(values.index.min().date()), "end": str(values.index.max().date()),
        "rebalance_assumption": rebalance,
        "methodology": ("Daily-rebalanced to target weights" if rebalance == "daily"
                        else "Buy-and-hold, weights drift with prices"),
    }
    if benchmark_symbol:
        b = store_df[store_df["symbol"] == benchmark_symbol.upper()].sort_values("date")
        if len(b) >= 2:
            bv = b.set_index("date")["close"].reindex(values.index).ffill().dropna()
            out["benchmark"] = {"symbol": benchmark_symbol.upper(),
                                "total_return": float(bv.iloc[-1] / bv.iloc[0] - 1)}
    out["asset_contribution_approx"] = contrib
    out["series"] = series
    return out

def portfolio_risk(store_df: pd.DataFrame, weights: dict[str, float],
                   risk_free_rate: float = 0.0, trading_days: int = TRADING_DAYS) -> dict:
    w = validate_weights(weights)
    wide = price_wide(store_df, list(w.keys())).dropna(how="any")
    if len(wide) < 3: raise ValueError("Insufficient overlapping history for risk analysis (need >= 3 common dates).")
    rets = wide.pct_change().dropna()
    wvec = np.array([w[s] for s in wide.columns])
    port_rets = pd.Series(rets.values @ wvec, index=rets.index)
    cov_daily = rets.cov()
    cov_ann = cov_daily * trading_days
    port_var = float(wvec @ cov_daily.values @ wvec)
    port_vol = float(np.sqrt(port_var) * np.sqrt(trading_days))
    # Component risk contributions: RC_i = w_i * (Sigma w)_i / sigma_p (daily), annualized by sqrt(252)
    sigma_w = cov_daily.values @ wvec
    if port_var > 0:
        comp = wvec * sigma_w / np.sqrt(port_var) * np.sqrt(trading_days)
    else:
        comp = np.zeros_like(wvec)
    series = portfolio_series(store_df, w, 100000.0, "daily")
    mdd = max_drawdown(series["value"])
    return {
        "weights": w,
        "annualized_volatility": port_vol,
        "sharpe_ratio": sharpe_ratio(port_rets, risk_free_rate, trading_days),
        "max_drawdown": mdd["max_drawdown"], "drawdown_peak_date": mdd["peak_date"], "drawdown_trough_date": mdd["trough_date"],
        "asset_volatility": {s: float(rets[s].std(ddof=1) * np.sqrt(trading_days)) for s in wide.columns},
        "correlation": rets.corr().round(4).to_dict(),
        "covariance_annualized": cov_ann.round(8).to_dict(),
        "component_risk_contribution": {s: float(c) for s, c in zip(wide.columns, comp)},
        "risk_free_rate": risk_free_rate,
        "observations": int(len(rets)),
        "diversification_note": "Component risk contributions sum to portfolio volatility. "
                                "Lower average correlation generally means stronger diversification benefit.",
    }
