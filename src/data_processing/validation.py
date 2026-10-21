"""Data-quality validation for normalized bhavcopy data."""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
import pandas as pd
from .normalize import COMMON_COLUMNS

@dataclass
class QualityReport:
    rows: int = 0
    symbols: int = 0
    date_min: str | None = None
    date_max: str | None = None
    duplicate_symbol_date: int = 0
    ohlc_violations: int = 0
    nonpositive_prices: int = 0
    negative_volume: int = 0
    missing_close: int = 0
    extreme_daily_moves: int = 0
    missing_weekday_sessions: int = 0
    issues: list[str] = field(default_factory=list)
    passed: bool = True

    def to_dict(self) -> dict:
        return asdict(self)

def validate_dataframe(df: pd.DataFrame) -> QualityReport:
    r = QualityReport()
    if df is None or df.empty:
        r.issues.append("Dataset is empty.")
        r.passed = False
        return r
    missing_cols = [c for c in COMMON_COLUMNS if c not in df.columns]
    if missing_cols:
        r.issues.append(f"Missing required columns: {missing_cols}")
        r.passed = False
    r.rows = int(len(df))
    r.symbols = int(df["symbol"].nunique())
    r.date_min = str(df["date"].min().date()) if pd.notna(df["date"].min()) else None
    r.date_max = str(df["date"].max().date()) if pd.notna(df["date"].max()) else None
    r.duplicate_symbol_date = int(df.duplicated(subset=["symbol", "date"]).sum())
    if r.duplicate_symbol_date: r.issues.append(f"{r.duplicate_symbol_date} duplicate symbol-date records.")
    price_cols = ["open", "high", "low", "close"]
    sub = df.dropna(subset=price_cols)
    r.ohlc_violations = int(((sub["high"] < sub[["open", "close", "low"]].max(axis=1)) |
                             (sub["low"] > sub[["open", "close", "high"]].min(axis=1))).sum())
    if r.ohlc_violations: r.issues.append(f"{r.ohlc_violations} OHLC consistency violations.")
    r.nonpositive_prices = int((df[price_cols] <= 0).any(axis=1).sum())
    if r.nonpositive_prices: r.issues.append(f"{r.nonpositive_prices} rows with non-positive prices.")
    r.negative_volume = int((df["volume"] < 0).sum()) if "volume" in df else 0
    if r.negative_volume: r.issues.append(f"{r.negative_volume} rows with negative volume.")
    r.missing_close = int(df["close"].isna().sum())
    # Extreme moves: |daily close change| > 50% within a symbol (possible split/error — flagged, not removed)
    chg = df.sort_values(["symbol", "date"]).groupby("symbol")["close"].pct_change()
    r.extreme_daily_moves = int((chg.abs() > 0.5).sum())
    if r.extreme_daily_moves: r.issues.append(f"{r.extreme_daily_moves} extreme (>50%) daily moves flagged — check corporate actions / errors. Prices are unadjusted.")
    # Missing weekday sessions between min and max (weekends excluded; holidays will appear here, so this is informational)
    if r.date_min and r.date_max:
        weekdays = pd.bdate_range(r.date_min, r.date_max)
        present = set(pd.to_datetime(df["date"]).dt.normalize().unique())
        r.missing_weekday_sessions = int(sum(1 for d in weekdays if d not in present))
    if r.passed and not r.issues:
        r.issues.append("No issues found. Note: missing weekday sessions may be market holidays; prices are unadjusted for corporate actions unless adjusted data is supplied.")
    return r
