"""Precomputed homepage statistics.

The homepage needs market-pulse, coverage-by-year, and data-quality numbers.
Computing them live means scanning/validating 4.1M rows on every page load —
wasteful locally and enough to exhaust small cloud containers. Instead they
are computed once (at ingestion time, or via scripts/precompute_overview.py)
and stored as a small JSON the homepage simply reads.
"""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from src.analytics import returns as R
from src.data_processing.validation import validate_dataframe

STATS_FILE = "overview_stats.json"

def build_overview_stats(store) -> dict:
    df = store.load_all()
    if df.empty:
        return {}
    latest = df["date"].max()
    day = df[df["date"] == latest].dropna(subset=["prev_close", "close"]).copy()
    day = day[day["prev_close"] > 0]
    day["change_pct"] = day["close"] / day["prev_close"] - 1
    movers = R.gainers_losers(day, 10)
    def clean(rows):
        return [{"symbol": r["symbol"], "close": float(r["close"]),
                 "change_pct": float(r["change_pct"]), "volume": float(r["volume"])} for r in rows]
    yearly = (df.assign(year=df["date"].dt.year).groupby("year")
                .agg(records=("close", "size"), symbols=("symbol", "nunique")).reset_index())
    rep = validate_dataframe(df)
    # Flagged records: the largest extreme daily moves, kept (never deleted)
    # so users can inspect exactly which observations were flagged and why.
    d = df.sort_values(["symbol", "date"])
    chg = d.groupby("symbol")["close"].pct_change()
    prev = d.groupby("symbol")["close"].shift(1)
    flagged_idx = chg[chg.abs() > 0.5].index
    flagged_records = []
    for idx in flagged_idx:
        row = d.loc[idx]
        flagged_records.append({"symbol": str(row["symbol"]), "date": str(row["date"].date()),
                                "prev_close": float(prev.loc[idx]), "close": float(row["close"]),
                                "change_pct": float(chg.loc[idx])})
    flagged_records.sort(key=lambda r: abs(r["change_pct"]), reverse=True)
    stats = {
        "generated_for_latest_date": str(latest.date()),
        "pulse": {
            "date": str(latest.date()),
            "advancers": int((day["change_pct"] > 0).sum()),
            "decliners": int((day["change_pct"] < 0).sum()),
            "unchanged": int((day["change_pct"] == 0).sum()),
            "traded": int(len(day)),
            "median_change_pct": float(day["change_pct"].median()),
            "total_turnover": float(day["turnover"].sum()),
            "gainers": clean(movers["gainers"]),
            "losers": clean(movers["losers"]),
        },
        "yearly": [{"year": int(r.year), "records": int(r.records), "symbols": int(r.symbols)}
                   for r in yearly.itertuples()],
        "quality": {
            "duplicates": int(rep.duplicate_symbol_date),
            "ohlc_violations": int(rep.ohlc_violations),
            "nonpositive_prices": int(rep.nonpositive_prices),
            "negative_volume": int(rep.negative_volume),
            "extreme_moves": int(rep.extreme_daily_moves),
            "missing_sessions": int(rep.missing_weekday_sessions),
        },
        "flagged_records": flagged_records[:200],
    }
    out = Path(store.metadata_dir) / STATS_FILE
    out.write_text(json.dumps(stats, indent=1))
    # Per-symbol stats: lets symbol pickers/info captions work WITHOUT loading
    # the full 642 MB dataset into memory (critical on small cloud hosts).
    total_days = int(df["date"].nunique())
    sym_stats = {}
    grouped = df.groupby("symbol")
    series_map = df.groupby("symbol")["series"].unique()
    for sym, g in grouped:
        sers = sorted(str(s) for s in series_map[sym])
        sym_stats[str(sym)] = {"rows": int(len(g)),
                               "start": str(g["date"].min().date()),
                               "end": str(g["date"].max().date()),
                               "series": sers, "is_equity": "EQ" in sers}
    payload = {"total_days": total_days,
               "trading_dates": [str(d.date()) for d in sorted(df["date"].unique())],
               "symbols": sym_stats}
    (Path(store.metadata_dir) / "symbol_stats.json").write_text(json.dumps(payload))
    return stats

def read_symbol_stats(metadata_dir) -> dict | None:
    path = Path(metadata_dir) / "symbol_stats.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None

def read_overview_stats(metadata_dir) -> dict | None:
    path = Path(metadata_dir) / STATS_FILE
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None
