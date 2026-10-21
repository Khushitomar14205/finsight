"""15 deterministic agent tools. Every quantitative answer the agent gives
is produced by one of these functions calling the analytics modules —
the LLM only routes and explains."""
from __future__ import annotations
from dataclasses import dataclass, field
import pandas as pd
from ..storage.store import MarketDataStore
from ..analytics import returns as R
from ..technical_indicators import indicators as TI
from ..portfolio import engine as PE
from ..optimization import optimizer as OPT
from ..data_processing.validation import validate_dataframe
from ..ml.experiments import run_experiment
from ..reporting.report import build_markdown_report

@dataclass
class AgentContext:
    store: MarketDataStore
    risk_free_rate: float = 0.06
    trading_days: int = 252
    portfolio_weights: dict = field(default_factory=dict)

def _df(ctx, symbols, start=None, end=None):
    return ctx.store.load(symbols=symbols, start=start, end=end)

def _one(ctx, symbol, start=None, end=None):
    df = _df(ctx, [symbol], start, end)
    if df.empty:
        raise ValueError(f"No local data for symbol {symbol} in the requested range. "
                         "Check data availability or ingest more dates.")
    return df

# 1
def validate_symbol_date(ctx, symbols=None, start=None, end=None, **kw):
    cov = ctx.store.coverage()
    available = set(ctx.store.available_symbols())
    symbols = [s.upper() for s in (symbols or [])]
    return {"symbols_requested": symbols,
            "symbols_available": [s for s in symbols if s in available],
            "symbols_missing": [s for s in symbols if s not in available],
            "dataset_date_min": cov.get("date_min"), "dataset_date_max": cov.get("date_max"),
            "requested_start": start, "requested_end": end,
            "valid": all(s in available for s in symbols) if symbols else False}

# 2
def get_historical_prices(ctx, symbols=None, start=None, end=None, **kw):
    df = _df(ctx, symbols or [], start, end)
    if df.empty: raise ValueError("No price rows found for those symbols/dates in local data.")
    return {"rows": int(len(df)), "symbols": sorted(df["symbol"].unique().tolist()),
            "start": str(df["date"].min().date()), "end": str(df["date"].max().date()),
            "latest_rows": df.sort_values("date").groupby("symbol").tail(1)
                              .assign(date=lambda d: d["date"].astype(str))
                              .to_dict("records"),
            "note": "These are historical closes, not live prices."}

# 3
def check_data_availability(ctx, symbols=None, **kw):
    cov = ctx.store.coverage()
    out = {"coverage": cov}
    for s in (symbols or []):
        df = _df(ctx, [s])
        out[s.upper()] = ({"rows": int(len(df)), "start": str(df["date"].min().date()),
                           "end": str(df["date"].max().date())} if not df.empty
                          else {"rows": 0, "note": "Symbol not present in local dataset."})
    return out

# 4
def calculate_returns(ctx, symbols=None, start=None, end=None, **kw):
    if not symbols: raise ValueError("Please name a symbol, e.g. 'return of RELIANCE in 2023'.")
    df = _one(ctx, symbols[0], start, end)
    s = R.stock_summary(df, ctx.risk_free_rate, ctx.trading_days)
    return {k: s[k] for k in ("symbol", "start", "end", "observations", "first_close", "last_close",
                              "total_return", "annualized_return", "best_day", "worst_day")}

# 5
def calculate_indicators(ctx, symbols=None, start=None, end=None, **kw):
    if not symbols: raise ValueError("Please name a symbol for indicator calculation.")
    df = _one(ctx, symbols[0], start, end)
    out = TI.add_indicators(df)
    last = out.iloc[-1]
    cols = ["rsi", "macd", "macd_signal", "sma_20", "sma_50", "bb_upper", "bb_lower", "atr"]
    return {"symbol": symbols[0].upper(), "as_of": str(last["date"].date()),
            "latest_indicators": {c: (float(last[c]) if pd.notna(last[c]) else None) for c in cols if c in out},
            "note": "Indicators describe historical behaviour; they do not predict future prices."}

# 6
def calculate_volatility_drawdown(ctx, symbols=None, start=None, end=None, **kw):
    if not symbols: raise ValueError("Please name a symbol.")
    df = _one(ctx, symbols[0], start, end)
    s = R.stock_summary(df, ctx.risk_free_rate, ctx.trading_days)
    return {k: s[k] for k in ("symbol", "start", "end", "annualized_volatility", "sharpe_ratio",
                              "max_drawdown", "drawdown_peak_date", "drawdown_trough_date")}

# 7
def compare_stocks(ctx, symbols=None, start=None, end=None, **kw):
    if not symbols or len(symbols) < 2: raise ValueError("Comparison needs at least two symbols.")
    df = _df(ctx, symbols, start, end)
    return R.compare_symbols(df, symbols, ctx.risk_free_rate)

# 8
def correlation_covariance(ctx, symbols=None, start=None, end=None, **kw):
    syms = symbols or list(ctx.portfolio_weights.keys())
    if len(syms) < 2: raise ValueError("Correlation needs at least two symbols (or a saved portfolio).")
    df = _df(ctx, syms, start, end)
    wide = PE.price_wide(df, syms).dropna(how="any")
    if len(wide) < 3: raise ValueError("Insufficient overlapping history for correlation.")
    rets = wide.pct_change().dropna()
    return {"symbols": syms, "observations": int(len(rets)),
            "correlation": R.correlation_matrix(rets).round(4).to_dict(),
            "covariance_annualized": R.covariance_matrix(rets, ctx.trading_days).round(8).to_dict()}

def _weights(ctx, weights=None):
    w = weights or ctx.portfolio_weights
    if not w: raise ValueError("No portfolio defined. Build one on the Portfolio Builder page first, or name the weights in your question's portfolio.")
    return w

# 9
def portfolio_performance(ctx, weights=None, benchmark=None, **kw):
    w = _weights(ctx, weights)
    df = _df(ctx, list(w.keys()))
    p = PE.portfolio_performance(df, w, risk_free_rate=ctx.risk_free_rate, benchmark_symbol=benchmark)
    p.pop("series", None)
    return p

# 10
def portfolio_risk(ctx, weights=None, **kw):
    w = _weights(ctx, weights)
    df = _df(ctx, list(w.keys()))
    return PE.portfolio_risk(df, w, risk_free_rate=ctx.risk_free_rate, trading_days=ctx.trading_days)

# 11
def portfolio_optimization(ctx, symbols=None, method="minimum_variance", max_weight=1.0, **kw):
    syms = symbols or list(ctx.portfolio_weights.keys())
    if len(syms) < 2: raise ValueError("Optimization needs at least two symbols.")
    df = _df(ctx, syms)
    if method == "maximum_sharpe":
        return OPT.maximum_sharpe(df, syms, risk_free_rate=ctx.risk_free_rate, max_weight=max_weight)
    if method == "equal_weight":
        w = OPT.equal_weight(syms)
        return {"weights": w, "method": "equal_weight",
                "performance": OPT._clean_perf(df, w), "risk": OPT._clean_risk(df, w)}
    return OPT.minimum_variance(df, syms, max_weight=max_weight)

# 12
def what_if_analysis(ctx, new_weights=None, weights=None, **kw):
    base = _weights(ctx, weights)
    if not new_weights: raise ValueError("What-if analysis needs the proposed new weights.")
    df = _df(ctx, list(set(base) | set(new_weights)))
    return OPT.what_if(df, base, new_weights, risk_free_rate=ctx.risk_free_rate)

# 13
def data_quality_report(ctx, **kw):
    # Prefer the precomputed quality summary (full validation of 4.1M rows is
    # heavy); fall back to live validation when no summary exists.
    from ..reporting.overview_stats import read_overview_stats
    stats = read_overview_stats(ctx.store.metadata_dir)
    if stats and stats.get("quality"):
        return {"source": "precomputed_at_ingestion", **stats["quality"]}
    df = ctx.store.load_all()
    if df.empty: return {"note": "No local data yet — nothing to validate."}
    return validate_dataframe(df).to_dict()

# 14
def run_ml_experiment(ctx, symbols=None, model="random_forest", **kw):
    if not symbols: raise ValueError("Please name a symbol for the ML experiment.")
    df = _one(ctx, symbols[0])
    return run_experiment(df, model_name=model)

# 15
def generate_report(ctx, symbols=None, start=None, end=None, **kw):
    if not symbols: raise ValueError("Please name at least one symbol for the report.")
    df = _one(ctx, symbols[0], start, end)
    s = R.stock_summary(df, ctx.risk_free_rate, ctx.trading_days)
    md = build_markdown_report(
        title=f"FinSight Research Report — {symbols[0].upper()}",
        summary={"Symbol": s["symbol"], "Period": f"{s['start']} to {s['end']}",
                 "Total return": f"{s['total_return']:.2%}", "Latest available date": s["end"]},
        sections={"Performance & risk": {k: (f"{v:.4f}" if isinstance(v, float) else v)
                                         for k, v in s.items() if not isinstance(v, (dict, list))}},
        quality=validate_dataframe(df).to_dict())
    return {"report_markdown": md, "symbol": symbols[0].upper()}

TOOL_REGISTRY = {
    "validate_symbol_date": validate_symbol_date,
    "get_historical_prices": get_historical_prices,
    "check_data_availability": check_data_availability,
    "calculate_returns": calculate_returns,
    "calculate_indicators": calculate_indicators,
    "calculate_volatility_drawdown": calculate_volatility_drawdown,
    "compare_stocks": compare_stocks,
    "correlation_covariance": correlation_covariance,
    "portfolio_performance": portfolio_performance,
    "portfolio_risk": portfolio_risk,
    "portfolio_optimization": portfolio_optimization,
    "what_if_analysis": what_if_analysis,
    "data_quality_report": data_quality_report,
    "run_ml_experiment": run_ml_experiment,
    "generate_report": generate_report,
}
