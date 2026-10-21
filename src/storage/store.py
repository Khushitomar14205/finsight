"""Partitioned Parquet storage (year=/month=) with filtered, lazy reads.
Does not load the whole dataset when symbol/date filters are given."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

class MarketDataStore:
    def __init__(self, processed_dir: Path, metadata_dir: Path):
        self.processed_dir = Path(processed_dir)
        self.metadata_dir = Path(metadata_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

    def _dataset_path(self) -> Path:
        return self.processed_dir / "prices"

    def save(self, df: pd.DataFrame) -> dict:
        """Append normalized data, de-duplicating on (date, symbol, series)."""
        if df.empty:
            return {"saved_rows": 0}
        existing = self.load_all()
        combined = df if existing.empty else pd.concat([existing, df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["date", "symbol", "series"], keep="last")
        combined = combined.sort_values(["date", "symbol"])
        combined["year"] = combined["date"].dt.year
        combined["month"] = combined["date"].dt.month
        out = self._dataset_path()
        # We rewrite the full de-duplicated dataset each save; remove the old
        # partitioned dataset first or to_parquet would append duplicate files.
        import shutil
        if out.exists():
            shutil.rmtree(out)
        combined.to_parquet(out, partition_cols=["year", "month"], index=False)
        manifest = {"rows": int(len(combined)), "symbols": int(combined["symbol"].nunique()),
                    "date_min": str(combined["date"].min().date()), "date_max": str(combined["date"].max().date())}
        (self.metadata_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        return {"saved_rows": int(len(df)), **manifest}

    def load_all(self) -> pd.DataFrame:
        path = self._dataset_path()
        if not path.exists():
            return pd.DataFrame()
        try:
            return pd.read_parquet(path)
        except Exception:
            return pd.DataFrame()

    def load(self, symbols: list[str] | None = None, start=None, end=None) -> pd.DataFrame:
        path = self._dataset_path()
        if not path.exists():
            return pd.DataFrame()
        filters = []
        if symbols:
            filters.append(("symbol", "in", [s.upper() for s in symbols]))
        if start is not None:
            filters.append(("date", ">=", pd.Timestamp(start)))
        if end is not None:
            filters.append(("date", "<=", pd.Timestamp(end)))
        try:
            df = pd.read_parquet(path, filters=filters if filters else None)
        except Exception:
            df = self.load_all()
            if symbols: df = df[df["symbol"].isin([s.upper() for s in symbols])]
            if start is not None: df = df[df["date"] >= pd.Timestamp(start)]
            if end is not None: df = df[df["date"] <= pd.Timestamp(end)]
        if "year" in df.columns: df = df.drop(columns=[c for c in ("year", "month") if c in df.columns])
        return df.sort_values(["symbol", "date"]).reset_index(drop=True)

    def coverage(self) -> dict:
        manifest_file = self.metadata_dir / "manifest.json"
        if manifest_file.exists():
            m = json.loads(manifest_file.read_text())
            return {**m, "latest_available_date": m.get("date_max"), "has_data": m.get("rows", 0) > 0}
        df = self.load_all()
        if df.empty:
            return {"rows": 0, "symbols": 0, "date_min": None, "date_max": None, "latest_available_date": None, "has_data": False}
        return {"rows": int(len(df)), "symbols": int(df["symbol"].nunique()),
                "date_min": str(df["date"].min().date()), "date_max": str(df["date"].max().date()),
                "latest_available_date": str(df["date"].max().date()), "has_data": True}

    def available_symbols(self) -> list[str]:
        df = self.load_all()
        return sorted(df["symbol"].unique().tolist()) if not df.empty else []

    def processed_dates(self) -> set[str]:
        df = self.load_all()
        return set(df["date"].dt.strftime("%Y-%m-%d")) if not df.empty else set()
