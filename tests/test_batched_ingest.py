import io, zipfile
from datetime import date
from pathlib import Path
from src.config.settings import Settings
from src.storage.store import MarketDataStore
from src.data_ingestion.downloader import BhavcopyDownloader

LEGACY = """SYMBOL,SERIES,OPEN,HIGH,LOW,CLOSE,LAST,PREVCLOSE,TOTTRDQTY,TOTTRDVAL,TIMESTAMP,TOTALTRADES,ISIN
RELIANCE,EQ,2400,2450,2380,2440,2440,2390,1000000,2440000000,{ts},50000,INE002A01018
"""

def _zip_bytes(ts: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("bhav.csv", LEGACY.format(ts=ts))
    return buf.getvalue()

def test_batched_ingest_and_resume(tmp_path):
    settings = Settings(data_dir=tmp_path, max_retries=1)
    store = MarketDataStore(tmp_path / "processed", tmp_path / "meta")
    dl = BhavcopyDownloader(settings, store)
    dates = {  # weekday dates in Jan 2023 covered by the fake download
        "cm02JAN2023bhav.csv.zip": _zip_bytes("02-JAN-2023"),
        "cm03JAN2023bhav.csv.zip": _zip_bytes("03-JAN-2023"),
    }
    def fake_fetch(url):
        for name, payload in dates.items():
            if name in url:
                return payload
        raise RuntimeError("HTTP 404 (fake)")
    dl._fetch = fake_fetch
    summary = dl.ingest_range_batched(date(2023, 1, 1), date(2023, 1, 31))
    assert summary["success"] == 2
    assert summary["rows"] == 2
    assert summary["skipped_weekend"] >= 8
    assert store.coverage()["rows"] == 2
    # Resume: same range downloads nothing new
    summary2 = dl.ingest_range_batched(date(2023, 1, 1), date(2023, 1, 31))
    assert summary2["success"] == 0
    assert summary2["skipped_existing"] == 2
    assert store.coverage()["rows"] == 2  # no duplicates after re-run
