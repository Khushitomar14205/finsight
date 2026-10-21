"""Deterministic explanations for the 'How this was computed' section.

Every line is derived from the tool that ran and its actual result —
formulas are fixed text per tool (never LLM-generated), and ranges /
observation counts are read from the result itself.
"""
from __future__ import annotations

TOOL_EXPLANATIONS = {
    "calculate_returns": {
        "definition": "Total price return = last close ÷ first close − 1, using the first and last actual "
                      "observations inside the range. Annualized return = CAGR: (last ÷ first)^(1/years) − 1 "
                      "with years = (observations − 1) ÷ 252.",
        "missing": "Only the symbol's own trading days in range are used; days it did not trade do not exist "
                   "in the data (nothing is filled or interpolated).",
    },
    "calculate_volatility_drawdown": {
        "definition": "Volatility = standard deviation of daily close-to-close returns (sample, ddof=1) × √252. "
                      "Sharpe = mean(daily return − risk-free/252) ÷ std × √252. Max drawdown = the minimum of "
                      "(close ÷ running peak close − 1).",
        "missing": "Computed on the symbol's available trading days in range only.",
    },
    "compare_stocks": {
        "definition": "Each symbol's metrics are calculated over the SAME common period (latest start to "
                      "earliest end across the symbols) with identical definitions: total return = last ÷ "
                      "first − 1; CAGR; volatility = std(daily returns) × √252; Sharpe vs the configured "
                      "risk-free rate; drawdown from running peak.",
        "missing": "Missing observations per symbol are counted and reported; correlation (when shown) uses "
                   "only dates where all symbols traded.",
    },
    "correlation_covariance": {
        "definition": "Pearson correlation of daily close-to-close returns on dates where ALL symbols have "
                      "observations (complete cases). Annualized covariance = daily covariance × 252.",
        "missing": "Dates where any symbol is missing are excluded for every symbol (common dates only).",
    },
    "calculate_indicators": {
        "definition": "RSI(14) uses Wilder's smoothing of gains/losses. MACD = EMA(12) − EMA(26), signal = "
                      "EMA(9) of MACD. SMAs are simple rolling means; Bollinger Bands = SMA(20) ± 2σ; ATR(14) "
                      "is Wilder's average true range. Values shown are the latest in the requested range.",
        "missing": "Indicators need their warm-up history; early rows without a full window are excluded, "
                   "never back-filled.",
    },
    "get_historical_prices": {
        "definition": "Raw rows from the local bhavcopy store for the requested symbols/range — no calculation "
                      "beyond selection. 'Latest' rows are each symbol's most recent observation in range.",
        "missing": "Only days the symbol actually traded appear.",
    },
    "check_data_availability": {
        "definition": "Counts and date ranges read directly from the local store (one record per symbol per "
                      "trading day). Latest close is the close of the symbol's most recent record.",
        "missing": "First/last dates are the first/last observations — coverage between them is not "
                   "guaranteed continuous.",
    },
    "validate_symbol_date": {
        "definition": "Symbol existence is checked against the store's symbol list; ranges against the "
                      "dataset's stored coverage.",
        "missing": "—",
    },
    "data_quality_report": {
        "definition": "Structural validation over the stored dataset: duplicate symbol-dates, OHLC ordering "
                      "violations, non-positive prices, negative volumes, and moves beyond ±50% flagged as "
                      "extreme (flagged, never altered or deleted).",
        "missing": "Weekday sessions with no published bhavcopy are classified as market holidays.",
    },
    "run_ml_experiment": {
        "definition": "Features at day t use only data up to day t (all inputs lagged); the target is day t+1's "
                      "close-to-close return. Chronological 60/20/20 train/validation/test split, no shuffling; "
                      "the model is fit on train only and evaluated on test. Baseline = predicting a zero "
                      "return. MAE/RMSE are in daily-return units.",
        "missing": "Rows lost to feature warm-up (lags/rolling windows) are dropped before splitting.",
    },
    "portfolio_performance": {
        "definition": "Portfolio daily return = Σ wᵢ·rᵢ with weights rebalanced to target daily (unless "
                      "buy-and-hold is selected); metrics use the same return/volatility/Sharpe/drawdown "
                      "definitions as single stocks, annualized with 252 trading days.",
        "missing": "Uses dates where the held symbols have data; weights are validated and normalized to 100%.",
    },
    "portfolio_risk": {
        "definition": "Risk from the covariance matrix of daily returns (annualized ×252) and the same "
                      "volatility/Sharpe/drawdown definitions used elsewhere in FinSight.",
        "missing": "Computed on common dates across holdings.",
    },
    "portfolio_optimization": {
        "definition": "scipy SLSQP optimization over historical daily returns (sample period): minimum "
                      "variance or maximum Sharpe, long-only, weights sum to 1, optional per-asset cap. "
                      "Reported performance is the optimized portfolio's historical (in-sample) result.",
        "missing": "Uses common dates across the selected symbols.",
    },
    "what_if_analysis": {
        "definition": "Base and proposed weights are run through the identical portfolio performance/risk "
                      "calculations over the same data, so the comparison is like-for-like.",
        "missing": "Uses common dates across all involved symbols.",
    },
    "generate_report": {
        "definition": "The report assembles the same deterministic summaries used by the other tools "
                      "(returns, risk, data quality) into Markdown — no separate calculations.",
        "missing": "—",
    },
}

_DEFAULT = {"definition": "Computed by a deterministic Python tool from the local bhavcopy dataset.",
            "missing": "Missing observations are excluded, never fabricated."}

def _range_lines(result: dict) -> list[str]:
    lines = []
    if result.get("start") and result.get("end"):
        lines.append(f"**Exact range used:** {result['start']} → {result['end']}")
    if result.get("common_start") and result.get("common_end"):
        lines.append(f"**Exact range used (common period):** {result['common_start']} → {result['common_end']}")
    obs = result.get("observations") or result.get("correlation_observations") or result.get("rows_used") or result.get("rows")
    if obs:
        lines.append(f"**Observations used:** {obs}")
    cov = result.get("coverage") or {}
    if cov:
        lines.append(f"**Dataset coverage:** {cov.get('date_min')} → {cov.get('date_max')} "
                     f"({cov.get('rows', 0):,} records, {cov.get('symbols', 0):,} symbols)")
    return lines

def explain_lines(tool: str, result: dict, plan: dict | None = None) -> list[str]:
    """Markdown lines for the expander: actual range/observations + the
    fixed definition for the tool that ran."""
    result = result or {}
    exp = TOOL_EXPLANATIONS.get(tool, _DEFAULT)
    lines = _range_lines(result)
    lines.append(f"**Definition:** {exp['definition']}")
    lines.append("**Annualization:** 252 trading days per year where applicable.")
    if exp.get("missing") and exp["missing"] != "—":
        lines.append(f"**Missing data:** {exp['missing']}")
    lines.append("**Prices:** unadjusted end-of-day closes from NSE bhavcopy. Returns are price returns — "
                 "dividends are not included, and no adjusted prices are fabricated. Splits/bonuses therefore "
                 "appear as price jumps and are flagged in data quality.")
    return lines
