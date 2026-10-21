"""NSE bhavcopy ingestion.

Source adapter pattern: URL construction lives in NSESourceAdapter so it can
be updated if NSE changes endpoints/formats. Downloads respect timeouts and
bounded retries, log every attempt, skip weekends and already-processed
dates, and never fabricate data when a download fails.

Known source reality (verified 2026-10-21 from this environment):
nsearchives.nseindia.com may return 403 (Akamai) or 404 to non-browser
clients. Failures are recorded in the ingestion log with the HTTP status and
can be retried; alternatively, place official bhavcopy files in data/raw/
and ingest them with ingest_local_file() / scripts/ingest.py --local.
"""
from __future__ import annotations
import json, time
from dataclasses import dataclass, asdict
from datetime import date, timedelta
from pathlib import Path
import requests
from ..data_processing.normalize import read_bhavcopy_bytes, read_bhavcopy_file, normalize
from ..data_processing.validation import validate_dataframe
from ..storage.store import MarketDataStore
from ..utils.logging_utils import get_logger

logger = get_logger("finsight.ingestion")

UDIFF_CUTOFF = date(2024, 6, 1)  # NSE publishes UDiFF-format CM bhavcopy from mid-2024

class NoTradingSessionError(Exception):
    """Raised when NSE published no bhavcopy for a weekday at all (both
    formats returned HTTP 404) — i.e. a market holiday / non-trading day.
    This is NOT a download failure and is logged/displayed as 'holiday'."""

def is_holiday_detail(detail: str) -> bool:
    """True when a failure detail shows 404 from every attempted format."""
    if not detail:
        return False
    parts = [p for p in detail.split("|") if p.strip()]
    return bool(parts) and all("HTTP 404" in p for p in parts)

class NSESourceAdapter:
    """Builds official NSE archive URLs for a trading date."""
    BASE = "https://nsearchives.nseindia.com"

    def urls_for(self, d: date) -> list[tuple[str, str]]:
        """Ordered (format, url) candidates for a date."""
        udiff = f"{self.BASE}/content/cm/BhavCopy_NSE_CM_0_0_0_{d.strftime('%Y%m%d')}_F_0000.csv.zip"
        mon = d.strftime("%b").upper()
        legacy = (f"{self.BASE}/content/historical/EQUITIES/{d.year}/{mon}/"
                  f"cm{d.strftime('%d')}{mon}{d.year}bhav.csv.zip")
        if d >= UDIFF_CUTOFF:
            return [("udiff", udiff), ("legacy", legacy)]
        return [("legacy", legacy), ("udiff", udiff)]

@dataclass
class IngestionRecord:
    date: str
    status: str  # success | holiday | skipped_weekend | skipped_existing | failed | local_success
    source_file: str = ""
    rows: int = 0
    detail: str = ""

class BhavcopyDownloader:
    def __init__(self, settings, store: MarketDataStore, adapter: NSESourceAdapter | None = None):
        self.settings = settings
        self.store = store
        self.adapter = adapter or NSESourceAdapter()
        self.log_file = settings.metadata_dir / "ingestion_log.jsonl"
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
            "Accept": "*/*", "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.nseindia.com/",
        })

    def _log(self, rec: IngestionRecord) -> None:
        self.settings.metadata_dir.mkdir(parents=True, exist_ok=True)
        with open(self.log_file, "a") as f:
            f.write(json.dumps(asdict(rec)) + "\n")
        logger.info("ingestion %s %s %s", rec.date, rec.status, rec.detail)

    def read_log(self) -> list[dict]:
        if not self.log_file.exists():
            return []
        return [json.loads(l) for l in self.log_file.read_text().splitlines() if l.strip()]

    def read_log_display(self) -> list[dict]:
        """Log entries for display: 'failed' entries that are actually
        market holidays (404 from every format) are shown as 'holiday'.
        This also corrects entries written before holiday classification
        existed."""
        out = []
        for entry in self.read_log():
            e = dict(entry)
            if e.get("status") == "failed" and is_holiday_detail(e.get("detail", "")):
                e["status"] = "holiday"
                e["display_note"] = "Market holiday — no trading session, no bhavcopy published."
            out.append(e)
        return out

    def _fetch(self, url: str) -> bytes:
        last_exc: Exception | None = None
        for attempt in range(1, self.settings.max_retries + 1):
            try:
                resp = self.session.get(url, timeout=self.settings.request_timeout_seconds)
                if resp.status_code == 200 and len(resp.content) > 200:
                    return resp.content
                raise requests.HTTPError(f"HTTP {resp.status_code} for {url}")
            except Exception as e:  # bounded retries with backoff
                last_exc = e
                if attempt < self.settings.max_retries:
                    time.sleep(min(2 ** attempt, 8))
        raise RuntimeError(str(last_exc))

    def ingest_date(self, d: date, force: bool = False) -> IngestionRecord:
        ds = d.isoformat()
        if d.weekday() >= 5:
            rec = IngestionRecord(ds, "skipped_weekend")
            self._log(rec); return rec
        if not force and ds in self.store.processed_dates():
            rec = IngestionRecord(ds, "skipped_existing")
            self._log(rec); return rec
        try:
            df, fmt = self._download_day(d)
            self.store.save(df)
            rec = IngestionRecord(ds, "success", source_file=str(df["source_file"].iloc[0]),
                                  rows=int(len(df)), detail=f"format={fmt}")
            self._log(rec); return rec
        except NoTradingSessionError as e:
            rec = IngestionRecord(ds, "holiday", detail=str(e)[:500])
            self._log(rec); return rec
        except Exception as e:
            rec = IngestionRecord(ds, "failed", detail=str(e)[:500])
            self._log(rec); return rec

    def ingest_range(self, start: date, end: date, force: bool = False) -> list[IngestionRecord]:
        records, d = [], start
        while d <= end:
            records.append(self.ingest_date(d, force=force))
            d += timedelta(days=1)
        return records

    def _download_day(self, d: date) -> tuple[object, str]:
        """Download + normalize one day. Returns (DataFrame, format) or raises."""
        errors = []
        for fmt, url in self.adapter.urls_for(d):
            try:
                content = self._fetch(url)
                raw = read_bhavcopy_bytes(content, url.split("/")[-1])
                df = normalize(raw, source_file=url.split("/")[-1])
                if df.empty:
                    errors.append(f"{fmt}: parsed 0 rows"); continue
                return df, fmt
            except Exception as e:
                errors.append(f"{fmt}: {e}")
        detail = " | ".join(errors)[:500]
        if is_holiday_detail(detail):
            raise NoTradingSessionError(detail)
        raise RuntimeError(detail)

    def ingest_range_batched(self, start: date, end: date, force: bool = False,
                             progress=None) -> dict:
        """Efficient bulk ingestion: downloads day by day but writes to the
        Parquet store once per calendar month (the per-day path rewrites the
        whole store every day, which is quadratic over long ranges).
        Resumable: already-processed dates are skipped, failures are logged
        per day and can be retried by re-running the same range."""
        import pandas as pd
        existing = set() if force else self.store.processed_dates()
        summary = {"success": 0, "failed": 0, "holiday": 0, "skipped_weekend": 0, "skipped_existing": 0, "rows": 0}
        month_start = date(start.year, start.month, 1)
        while month_start <= end:
            if month_start.month == 12:
                next_month = date(month_start.year + 1, 1, 1)
            else:
                next_month = date(month_start.year, month_start.month + 1, 1)
            frames, d = [], max(month_start, start)
            month_end = min(next_month - timedelta(days=1), end)
            while d <= month_end:
                ds = d.isoformat()
                if d.weekday() >= 5:
                    summary["skipped_weekend"] += 1
                    self._log(IngestionRecord(ds, "skipped_weekend"))
                elif ds in existing:
                    summary["skipped_existing"] += 1
                    self._log(IngestionRecord(ds, "skipped_existing"))
                else:
                    try:
                        df, fmt = self._download_day(d)
                        frames.append(df)
                        existing.add(ds)
                        summary["success"] += 1; summary["rows"] += int(len(df))
                        self._log(IngestionRecord(ds, "success", source_file=str(df["source_file"].iloc[0]),
                                                  rows=int(len(df)), detail=f"format={fmt} batched"))
                    except NoTradingSessionError as e:
                        summary["holiday"] += 1
                        self._log(IngestionRecord(ds, "holiday", detail=str(e)[:500]))
                    except Exception as e:
                        summary["failed"] += 1
                        self._log(IngestionRecord(ds, "failed", detail=str(e)[:500]))
                d += timedelta(days=1)
            if frames:
                self.store.save(pd.concat(frames, ignore_index=True))
            if progress:
                progress(f"{month_start.strftime('%Y-%m')}: cumulative {summary}")
            month_start = next_month
        return summary

    def ingest_local_file(self, path) -> IngestionRecord:
        """Recovery path: ingest an official bhavcopy file the user placed locally."""
        p = Path(path)
        raw = read_bhavcopy_file(p)
        df = normalize(raw, source_file=p.name)
        if df.empty:
            rec = IngestionRecord("unknown", "failed", source_file=p.name, detail="Parsed 0 rows")
            self._log(rec); return rec
        self.store.save(df)
        report = validate_dataframe(df)
        ds = str(df["date"].min().date())
        rec = IngestionRecord(ds, "local_success", source_file=p.name, rows=int(len(df)),
                              detail=f"quality_passed={report.passed}")
        self._log(rec); return rec
