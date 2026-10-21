"""Turn tool results into plain-English answers.

The chat shows people answers, not JSON. Each major tool has a dedicated
formatter; anything else falls back to a readable key-value rendering.
Numbers are formatted here (percentages, ₹ prices) — never invented.
"""
from __future__ import annotations

def pct(x, signed=True):
    if x is None: return "—"
    try: return f"{x:+.2%}" if signed else f"{x:.2%}"
    except Exception: return str(x)

def rs(x):
    if x is None: return "—"
    try: return f"₹{x:,.2f}"
    except Exception: return str(x)

def num(x, dp=2):
    if x is None: return "—"
    try: return f"{x:,.{dp}f}"
    except Exception: return str(x)

def period_label(start, end):
    if start and end:
        if start[:4] == end[:4] and start[5:] == "01-01" or (start[:4] == end[:4] and start[5:7] == "01" and end[5:7] == "12"):
            return f"in {start[:4]}"
        return f"from {start} to {end}"
    if start: return f"from {start}"
    if end: return f"up to {end}"
    return "over the available history"

def _returns(r, plan):
    sym = r.get("symbol", "")
    tr = r.get("total_return")
    lines = [f"**{sym} returned {pct(tr)}** {period_label(r.get('start'), r.get('end'))} "
             f"({r.get('start')} → {r.get('end')}, {r.get('observations')} trading days), "
             f"moving from {rs(r.get('first_close'))} to {rs(r.get('last_close'))}."]
    if r.get("annualized_return") is not None:
        lines.append(f"Annualized return (CAGR): {pct(r['annualized_return'])}.")
    bd, wd = r.get("best_day") or {}, r.get("worst_day") or {}
    if bd.get("return") is not None and wd.get("return") is not None:
        lines.append(f"Best day: {pct(bd['return'])} on {bd.get('date')} · Worst day: {pct(wd['return'])} on {wd.get('date')}.")
    if isinstance(tr, (int, float)):
        lines.append(f"In plain terms: ₹100 invested at the first close would have become "
                     f"₹{100 * (1 + tr):,.2f} at the last close — price change only, before costs and taxes, "
                     "excluding dividends. Historical performance is not a prediction of future results.")
    return " ".join(lines)

def _volatility(r, plan):
    return (f"**{r.get('symbol')} risk profile** ({r.get('start')} → {r.get('end')}): "
            f"annualized volatility **{pct(r.get('annualized_volatility'), signed=False)}**, "
            f"Sharpe ratio **{num(r.get('sharpe_ratio'))}**, and a maximum drawdown of "
            f"**{pct(r.get('max_drawdown'))}** (peak {r.get('drawdown_peak_date')} → trough {r.get('drawdown_trough_date')}). "
            f"In plain terms: volatility is the size of typical day-to-day swings scaled to a year — higher means a "
            f"bumpier ride; the drawdown is the worst peak-to-trough fall an investor in this window would have sat through.")

def _compare(r, plan):
    stats = r.get("symbols") or {}
    lines = [f"**Comparison over the common period {r.get('common_start')} → {r.get('common_end')}** "
             f"— every metric below is calculated on this same range, using aligned observations and the same "
             f"close-to-close price-return definition for all symbols.",
             "",
             "| Symbol | Total return | Ann. return | Volatility | Sharpe | Max drawdown |",
             "|---|---|---|---|---|---|"]
    for sym, s in stats.items():
        lines.append(f"| {sym} | {pct(s.get('total_return'))} | {pct(s.get('annualized_return'))} | "
                     f"{pct(s.get('annualized_volatility'), signed=False)} | {num(s.get('sharpe_ratio'))} | "
                     f"{pct(s.get('max_drawdown'))} |")
    missing = {k: v for k, v in (r.get("missing_observations") or {}).items() if v}
    if missing:
        lines.append("")
        lines.append(f"Missing observations inside the common range: "
                     + ", ".join(f"{k}: {v}" for k, v in missing.items())
                     + " — the symbol simply didn't trade on those days; thin trading can distort comparisons.")
    # Interpretation strictly from the calculated values.
    if len(stats) >= 2:
        def pick(key, higher=True):
            items = [(k, v[key]) for k, v in stats.items() if v.get(key) is not None]
            return (max if higher else min)(items, key=lambda kv: kv[1]) if items else (None, None)
        hi_ret = pick("total_return"); hi_vol = pick("annualized_volatility"); big_dd = pick("max_drawdown", higher=False)
        lines.append("")
        lines.append(f"**Interpretation:** {hi_ret[0]} had the higher observed return ({pct(hi_ret[1])}); "
                     f"{hi_vol[0]} was the more volatile ({pct(hi_vol[1], signed=False)} annualized — larger day-to-day swings); "
                     f"{big_dd[0]} experienced the larger maximum drawdown ({pct(big_dd[1])} peak-to-trough). "
                     "These are historical facts about this specific period, not predictions or recommendations.")
    # Corporate-action / suspicious-price warning, verified from each symbol's worst day.
    for sym, s in stats.items():
        wd = s.get("worst_day") or {}
        if isinstance(wd.get("return"), (int, float)) and wd["return"] <= -0.25:
            lines.append("")
            lines.append(f"⚠️ **Data-quality warning — {sym}:** its worst day in this period was "
                         f"{pct(wd['return'])} on {wd.get('date')}. FinSight uses unadjusted closing prices, and a "
                         "single-day drop of this size usually reflects a corporate action such as a stock split or "
                         "bonus issue (which mechanically rescales the price) rather than an equivalent economic "
                         "loss. No adjusted prices are fabricated, so this symbol's period return should be read "
                         "with that in mind.")
    return "\n".join(lines)

def _indicators(r, plan):
    ind = r.get("latest_indicators") or {}
    parts = [f"**{r.get('symbol')} — latest indicators as of {r.get('as_of')}:**"]
    if ind.get("rsi") is not None:
        zone = "overbought territory" if ind["rsi"] >= 70 else "oversold territory" if ind["rsi"] <= 30 else "the neutral zone"
        parts.append(f"RSI(14) is {num(ind['rsi'], 1)} — {zone}.")
    if ind.get("macd") is not None and ind.get("macd_signal") is not None:
        rel = "above" if ind["macd"] >= ind["macd_signal"] else "below"
        parts.append(f"MACD ({num(ind['macd'])}) is {rel} its signal line ({num(ind['macd_signal'])}) — "
                     f"{'bullish' if rel == 'above' else 'bearish'} short-term momentum.")
    if ind.get("sma_20") is not None or ind.get("sma_50") is not None:
        parts.append(f"SMA 20: {rs(ind.get('sma_20'))} · SMA 50: {rs(ind.get('sma_50'))}.")
    if ind.get("bb_upper") is not None:
        parts.append(f"Bollinger band range: {rs(ind.get('bb_lower'))} – {rs(ind.get('bb_upper'))}.")
    if ind.get("atr") is not None:
        parts.append(f"ATR: {num(ind['atr'])}.")
    return " ".join(parts)

def _prices(r, plan):
    lines = [f"Found **{r.get('rows')} historical price records** for {', '.join(r.get('symbols') or [])} "
             f"({r.get('start')} → {r.get('end')}). Latest available observation per symbol:",
             "",
             "| Symbol | Date | Close | Volume |",
             "|---|---|---|---|"]
    for row in r.get("latest_rows") or []:
        lines.append(f"| {row.get('symbol')} | {str(row.get('date'))[:10]} | {rs(row.get('close'))} | "
                     f"{num(row.get('volume'), 0)} |")
    lines.append("")
    lines.append("These are historical end-of-day closes — not live market prices.")
    return "\n".join(lines)

def _availability(r, plan):
    cov = r.get("coverage") or {}
    total_days = r.get("dataset_trading_days")
    head = f"The local dataset holds **{cov.get('rows', 0):,} records** across {cov.get('symbols', 0):,} symbols"
    if total_days:
        head += f" over **{total_days:,} trading days**"
    head += f", spanning {cov.get('date_min')} → {cov.get('date_max')}."
    lines = [head]
    for key, v in r.items():
        if key in ("coverage", "dataset_trading_days") or not isinstance(v, dict): continue
        if v.get("rows"):
            line = (f"- **{key}**: {v['rows']:,} records = {v.get('trading_days', v['rows']):,} trading days "
                    f"(one record per day it traded), first observation {v.get('start')}, last {v.get('end')}. "
                    f"Latest close: {rs(v.get('latest_close'))} on {v.get('latest_close_date')}.")
            if total_days and v.get("trading_days"):
                line += f" It traded on {v['trading_days'] / total_days:.0%} of all dataset trading days."
            line += (" The first/last dates are simply its first and last observations — gaps in between are days "
                     "it didn't trade (not yet listed, suspended, or illiquid), so coverage inside the range is "
                     "not necessarily continuous.")
            lines.append(line)
        else:
            lines.append(f"- **{key}**: not present in the local dataset — check the symbol spelling, or it may "
                         "trade under a different ticker / only on another exchange.")
    lines.append("Records existing is not proof of completeness: FinSight validates each ingested file "
                 "(duplicates, OHLC sanity), and quality details are on the Data Quality page.")
    return "\n".join(lines)

def _ml(r, plan):
    res = r.get("results") or {}
    model_key = r.get("model") or next((k for k in res if k not in ("baseline_zero_return", "strategy_test",
                                                                    "feature_importance")), "model")
    m = res.get(model_key) or {}
    base = res.get("baseline_zero_return") or {}
    counts = r.get("split_counts") or {}
    split = r.get("split") or {}
    lines = [f"**ML experiment — {r.get('symbol')}** ({r.get('task', 'regression')} · model: **{model_key}**)",
             f"Target: {r.get('target', 'next-day return')}. Features: {', '.join(r.get('features') or [])}.",
             f"Data: {r.get('rows_used')} usable observations after feature construction, split chronologically "
             f"(no shuffling): train {counts.get('train')} rows ({(split.get('train') or ['?','?'])[0]} → "
             f"{(split.get('train') or ['?','?'])[1]}), validation {counts.get('validation')} rows, "
             f"test {counts.get('test')} rows ({(split.get('test') or ['?','?'])[0]} → {(split.get('test') or ['?','?'])[1]}). "
             "The model is fit on the training segment only; all metrics below are on the held-out test segment.",
             "",
             f"**Test results:** MAE {num(m.get('mae'), 4)}, RMSE {num(m.get('rmse'), 4)}, "
             f"directional accuracy {pct(m.get('directional_accuracy'), signed=False)}.",
             f"**Baseline (always predict zero return):** MAE {num(base.get('mae'), 4)}, RMSE {num(base.get('rmse'), 4)}, "
             f"directional accuracy {pct(base.get('directional_accuracy'), signed=False)}. "
             "A useful model must beat this baseline — directional accuracy near 50% is chance level."]
    strat = res.get("strategy_test") or {}
    if strat:
        lines.append(f"For reference, a simple long/flat rule on the model's test predictions returned "
                     f"{pct(strat.get('cumulative_strategy_return_after_costs'))} after costs vs "
                     f"{pct(strat.get('cumulative_buy_and_hold_return_test'))} for buy-and-hold over the same test "
                     "window — a backtest on one historical window, not evidence of a reliable trading signal.")
    fi = res.get("feature_importance")
    if isinstance(fi, dict) and fi:
        top = list(fi.items())[:3]
        lines.append("Most influential features: " + ", ".join(f"{k} ({v:.2f})" for k, v in top) + ".")
    if r.get("limitations"):
        lines.append(f"Limitations: {r['limitations']}")
    return "\n".join(lines)

def _validate(r, plan):
    lines = []
    if r.get("symbols_available"):
        lines.append(f"Available in the dataset: {', '.join(r['symbols_available'])}.")
    if r.get("symbols_missing"):
        lines.append(f"Not found: {', '.join(r['symbols_missing'])} — check the spelling, or the data may not be ingested.")
    lines.append(f"Dataset range: {r.get('dataset_date_min')} → {r.get('dataset_date_max')}.")
    return " ".join(lines)

def _correlation(r, plan):
    corr = r.get("correlation") or {}
    syms = r.get("symbols") or list(corr.keys())
    lines = [f"Correlations of daily returns over {r.get('observations')} overlapping trading days:"]
    for i, a in enumerate(syms):
        for b in syms[i + 1:]:
            v = (corr.get(a) or {}).get(b)
            if v is not None:
                strength = "strong" if abs(v) >= 0.7 else "moderate" if abs(v) >= 0.4 else "weak"
                lines.append(f"- **{a} – {b}**: {num(v)} ({strength} {'positive' if v >= 0 else 'negative'} co-movement)")
    return "\n".join(lines)

def _quality(r, plan):
    def get(*keys):
        for k in keys:
            if k in r: return r[k]
        return None
    lines = ["**Data quality summary** for the local dataset:"]
    checks = [("Duplicate symbol-dates", get("duplicates", "duplicate_symbol_date")),
              ("OHLC violations", get("ohlc_violations")),
              ("Non-positive prices", get("nonpositive_prices")),
              ("Negative volume rows", get("negative_volume")),
              ("Extreme daily moves flagged", get("extreme_moves", "extreme_daily_moves")),
              ("Missing weekday sessions (holidays)", get("missing_sessions", "missing_weekday_sessions"))]
    for label, v in checks:
        if v is not None: lines.append(f"- {label}: {v}")
    if r.get("rows") is not None:
        lines.append(f"Covers {r['rows']:,} rows across {r.get('symbols', '—')} symbols.")
    return "\n".join(lines)

def _weights_line(weights):
    if not weights: return None
    return "Weights: " + ", ".join(f"{k} {pct(v, signed=False)}" for k, v in weights.items())

def _portfolio(r, plan):
    lines = []
    wl = _weights_line(r.get("weights"))
    if wl: lines.append(wl + ".")
    bits = []
    if r.get("total_return") is not None: bits.append(f"total return {pct(r['total_return'])}")
    if r.get("annualized_return") is not None: bits.append(f"annualized return {pct(r['annualized_return'])}")
    if r.get("annualized_volatility") is not None: bits.append(f"volatility {pct(r['annualized_volatility'], signed=False)}")
    if r.get("sharpe_ratio") is not None: bits.append(f"Sharpe {num(r['sharpe_ratio'])}")
    if r.get("max_drawdown") is not None: bits.append(f"max drawdown {pct(r['max_drawdown'])}")
    if bits:
        period = f" ({r['start']} → {r['end']})" if r.get("start") else ""
        lines.append(f"Portfolio performance{period}: " + " · ".join(bits) + ".")
    rc = r.get("risk_contributions_pct") or r.get("risk_contributions")
    if isinstance(rc, dict):
        top = max(rc.items(), key=lambda kv: kv[1] if isinstance(kv[1], (int, float)) else 0)
        lines.append(f"Largest risk contributor: {top[0]}.")
    return "\n".join(lines) if lines else _generic(r)

_SKIP = {"methodology", "series", "note"}
def _generic(r, depth=0):
    lines = []
    for k, v in r.items():
        if k in _SKIP or v is None: continue
        label = k.replace("_", " ").capitalize()
        if isinstance(v, dict):
            if k == "weights":
                lines.append(_weights_line(v) + ".")
            elif depth < 1:
                sub = _generic(v, depth + 1)
                if sub: lines.append(f"**{label}:**\n{sub}")
        elif isinstance(v, list):
            if v and isinstance(v[0], dict):
                lines.append(f"**{label}:** {len(v)} entries (see details).")
            elif v:
                lines.append(f"- {label}: {', '.join(str(x) for x in v[:8])}")
        elif isinstance(v, float):
            if any(h in k for h in ("return", "volatility", "drawdown", "change")):
                lines.append(f"- {label}: {pct(v)}")
            else:
                lines.append(f"- {label}: {num(v, 4)}")
        else:
            lines.append(f"- {label}: {v}")
    return "\n".join(lines)

_FORMATTERS = {
    "calculate_returns": _returns,
    "calculate_volatility_drawdown": _volatility,
    "compare_stocks": _compare,
    "calculate_indicators": _indicators,
    "get_historical_prices": _prices,
    "check_data_availability": _availability,
    "validate_symbol_date": _validate,
    "correlation_covariance": _correlation,
    "data_quality_report": _quality,
    "portfolio_performance": _portfolio,
    "portfolio_risk": _portfolio,
    "portfolio_optimization": _portfolio,
    "what_if_analysis": _portfolio,
    "run_ml_experiment": _ml,
}

def humanize(tool_name: str, result: dict, plan: dict | None = None) -> str:
    if not isinstance(result, dict):
        return str(result)
    fn = _FORMATTERS.get(tool_name)
    if fn:
        try:
            text = fn(result, plan or {})
            if text and text.strip():
                return text.strip()
        except Exception:
            pass
    return _generic(result) or "The analysis completed, but there was nothing to summarize."
