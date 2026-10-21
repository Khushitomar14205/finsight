import json
from datetime import date
from src.config.settings import Settings
from src.storage.store import MarketDataStore
from src.data_ingestion.downloader import BhavcopyDownloader, is_holiday_detail

def test_is_holiday_detail():
    both_404 = "legacy: HTTP 404 for https://x | udiff: HTTP 404 for https://y"
    mixed = "legacy: HTTP 404 for https://x | udiff: HTTP 403 for https://y"
    assert is_holiday_detail(both_404)
    assert not is_holiday_detail(mixed)
    assert not is_holiday_detail("")

def _dl_with_fetch(tmp_path, exc):
    settings = Settings(data_dir=tmp_path, max_retries=1)
    store = MarketDataStore(tmp_path / "processed", tmp_path / "meta")
    dl = BhavcopyDownloader(settings, store)
    def fake_fetch(url):
        raise RuntimeError(exc)
    dl._fetch = fake_fetch
    return dl

def test_404_day_is_holiday_not_failed(tmp_path):
    dl = _dl_with_fetch(tmp_path, "HTTP 404 for https://example")
    # Monday 2023-01-02 only
    summary = dl.ingest_range_batched(date(2023, 1, 2), date(2023, 1, 2))
    assert summary["holiday"] == 1
    assert summary["failed"] == 0
    statuses = [e["status"] for e in dl.read_log_display()]
    assert statuses == ["holiday"]

def test_403_day_is_failed_not_holiday(tmp_path):
    dl = _dl_with_fetch(tmp_path, "HTTP 403 for https://example")
    summary = dl.ingest_range_batched(date(2023, 1, 2), date(2023, 1, 2))
    assert summary["failed"] == 1
    assert summary["holiday"] == 0

def test_display_reclassifies_old_failed_holiday_entries(tmp_path):
    settings = Settings(data_dir=tmp_path, max_retries=1)
    store = MarketDataStore(tmp_path / "processed", tmp_path / "meta")
    dl = BhavcopyDownloader(settings, store)
    dl.log_file.parent.mkdir(parents=True, exist_ok=True)
    dl.log_file.write_text(json.dumps({
        "date": "2023-01-26", "status": "failed", "source_file": "", "rows": 0,
        "detail": "legacy: HTTP 404 for https://x | udiff: HTTP 404 for https://y"}) + "\n")
    shown = dl.read_log_display()
    assert shown[0]["status"] == "holiday"
    # raw log is untouched
    assert dl.read_log()[0]["status"] == "failed"
