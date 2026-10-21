"""Bhavcopy parsing + normalization.

Supports:
- Legacy NSE bhavcopy (pre-UDiFF): SYMBOL, SERIES, OPEN, HIGH, LOW, CLOSE,
  LAST, PREVCLOSE, TOTTRDQTY, TOTTRDVAL, TIMESTAMP, TOTALTRADES, ISIN
- UDiFF bhavcopy: TradDt, TckrSymb, SctySrs, OpnPric, HghPric, LwPric,
  ClsPric, LastPric, PrvsClsgPric, TtlTradgVol, TtlTrfVal, TtlNbOfTxsExctd

Both are mapped to COMMON_SCHEMA. Missing observations are never invented.
"""
from __future__ import annotations
import io, zipfile
import pandas as pd

COMMON_COLUMNS = ["date", "symbol", "series", "open", "high", "low", "close",
                  "prev_close", "last_price", "vwap", "volume", "turnover",
                  "trades", "source_file"]

LEGACY_MAP = {
    "SYMBOL": "symbol", "SERIES": "series", "OPEN": "open", "HIGH": "high",
    "LOW": "low", "CLOSE": "close", "LAST": "last_price",
    "PREVCLOSE": "prev_close", "TOTTRDQTY": "volume", "TOTTRDVAL": "turnover",
    "TIMESTAMP": "date", "TOTALTRADES": "trades",
}
UDIFF_MAP = {
    "TradDt": "date", "TckrSymb": "symbol", "SctySrs": "series",
    "OpnPric": "open", "HghPric": "high", "LwPric": "low", "ClsPric": "close",
    "LastPric": "last_price", "PrvsClsgPric": "prev_close",
    "TtlTradgVol": "volume", "TtlTrfVal": "turnover", "TtlNbOfTxsExctd": "trades",
}

def detect_format(columns: list[str]) -> str:
    cols = set(columns)
    if "TckrSymb" in cols or "TradDt" in cols:
        return "udiff"
    if "SYMBOL" in cols and "TIMESTAMP" in cols:
        return "legacy"
    return "unknown"

def read_bhavcopy_bytes(content: bytes, filename: str) -> pd.DataFrame:
    """Read a .zip / .csv.gz / .csv bhavcopy payload into a raw DataFrame."""
    name = filename.lower()
    if name.endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
            if not csv_names:
                raise ValueError(f"No CSV inside zip {filename}")
            with zf.open(csv_names[0]) as f:
                return pd.read_csv(f, dtype=str)
    if name.endswith(".csv.gz") or name.endswith(".gz"):
        return pd.read_csv(io.BytesIO(content), compression="gzip", dtype=str)
    return pd.read_csv(io.BytesIO(content), dtype=str)

def read_bhavcopy_file(path) -> pd.DataFrame:
    from pathlib import Path
    p = Path(path)
    return read_bhavcopy_bytes(p.read_bytes(), p.name)

def normalize(raw: pd.DataFrame, source_file: str) -> pd.DataFrame:
    """Map a raw bhavcopy frame to the common schema with typed columns."""
    fmt = detect_format(list(raw.columns))
    if fmt == "unknown":
        raise ValueError(f"Unrecognized bhavcopy schema; columns={list(raw.columns)[:20]}")
    mapping = LEGACY_MAP if fmt == "legacy" else UDIFF_MAP
    df = raw.rename(columns=mapping)
    df = df[[c for c in mapping.values() if c in df.columns]].copy()
    for col in COMMON_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    df["symbol"] = df["symbol"].astype(str).str.strip().str.upper()
    df["series"] = df["series"].astype(str).str.strip().str.upper()
    df = df[~df["symbol"].isin(["", "NAN", "NONE"])]
    # NSE equity bhavcopy: keep EQ series when series info exists; other series
    # (e.g. BE, BZ) are preserved but flagged via the series column.
    df["date"] = pd.to_datetime(df["date"], errors="coerce", format="mixed").dt.tz_localize(None).dt.normalize()
    numeric = ["open", "high", "low", "close", "prev_close", "last_price", "vwap", "volume", "turnover", "trades"]
    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["source_file"] = source_file
    df = df.dropna(subset=["date", "symbol", "close"])
    df = df.drop_duplicates(subset=["date", "symbol", "series"], keep="last")
    # Bhavcopy can list a symbol in several series on one date (EQ, BE, ...).
    # For equity analytics we keep one row per symbol-date, preferring EQ.
    df["_series_rank"] = (df["series"] != "EQ").astype(int)
    df = (df.sort_values(["date", "symbol", "_series_rank"])
            .drop_duplicates(subset=["date", "symbol"], keep="first")
            .drop(columns=["_series_rank"]))
    df = df.sort_values(["date", "symbol"]).reset_index(drop=True)
    return df[COMMON_COLUMNS]
