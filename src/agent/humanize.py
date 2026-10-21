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
    lines = [f"**{sym} returned {pct(r.get('total_return'))}** {period_label(r.get('start'), r.get('end'))} "
             f"({r.get('start')} → {r.get('end')}, {r.get('observations')} trading days), "
             f"moving from {rs(r.get('first_close'))} to {rs(r.get('last_close'))}."]
    if r.get("annualized_return") is not None:
        lines.append(f"Annualized return: {pct(r['annualized_return'])}.")
    bd, wd = r.get("best_day") or {}, r.get("worst_day") or {}
    if bd.get("return") is not None and wd.get("return") is not None:
        lines.append(f"Best day: {pct(bd['return'])} on {bd.get('date')} · Worst day: {pct(wd['return'])} on {wd.get('date')}.")
    return " ".join(lines)

def _volatility(r, plan):
    return (f"**{r.get('symbol')} risk profile** ({r.get('start')} → {r.get('end')}): "
            f"annualized volatility **{pct(r.get('annualized_volatility'), signed=False)}**, "
            f"Sharpe ratio **{num(r.get('sharpe_ratio'))}**, and a maximum drawdown of "
            f"**{pct(r.get('max_drawdown'))}** (peak {r.get('drawdown_peak_date')} → trough {r.get('drawdown_trough_date')}).")

def _compare(r, plan):
    lines = [f"Comparing over the common period {r.get('common_start')} → {r.get('common_end')}:"]
    for sym, s in (r.get("symbols") or {}).items():
        lines.append(f"- **{sym}**: total return {pct(s.get('total_return'))} · annualized {pct(s.get('annualized_return'))} · "
                     f"volatility {pct(s.get('annualized_volatility'), signed=False)} · Sharpe {num(s.get('sharpe_ratio'))} · "
                     f"max drawdown {pct(s.get('max_drawdown'))}")
    missing = {k: v for k, v in (r.get("missing_observations") or {}).items() if v}
    if missing:
        lines.append(f"Note: missing observations inside the common range: {missing} — thin trading can distort comparisons.")
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
    lines = [f"Found **{r.get('rows')} price rows** for {', '.join(r.get('symbols') or [])} "
             f"({r.get('start')} → {r.get('end')}). Latest available closes:"]
    for row in r.get("latest_rows") or []:
        lines.append(f"- **{row.get('symbol')}**: {rs(row.get('close'))} on {str(row.get('date'))[:10]} "
                     f"(volume {num(row.get('volume'), 0)})")
    lines.append("These are historical closes, not live prices.")
    return "\n".join(lines)

def _availability(r, plan):
    cov = r.get("coverage") or {}
    lines = [f"The local dataset covers **{cov.get('date_min')} → {cov.get('date_max')}** "
             f"({cov.get('rows', 0):,} rows, {cov.get('symbols', 0):,} symbols)."]
    for key, v in r.items():
        if key == "coverage" or not isinstance(v, dict): continue
        if v.get("rows"):
            lines.append(f"- **{key}**: {v['rows']:,} rows, {v.get('start')} → {v.get('end')}.")
        else:
            lines.append(f"- **{key}**: not present in the local dataset.")
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
