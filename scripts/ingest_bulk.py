"""Bulk real-data ingestion in monthly chunks (resumable).

Usage:
  python scripts/ingest_bulk.py --start 2020-01-01 --end 2026-10-06
Progress is printed per month and every day is recorded in
data/metadata/ingestion_log.jsonl. Re-running the same command safely
resumes: processed dates are skipped, failed dates are retried."""
import argparse, sys
from datetime import date
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config.settings import get_settings
from src.storage.store import MarketDataStore
from src.data_ingestion.downloader import BhavcopyDownloader

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--end", default="2026-10-06")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    s = get_settings()
    store = MarketDataStore(s.processed_dir, s.metadata_dir)
    dl = BhavcopyDownloader(s, store)
    print(f"Bulk ingestion {args.start} -> {args.end} (monthly chunks)", flush=True)
    summary = dl.ingest_range_batched(date.fromisoformat(args.start),
                                      date.fromisoformat(args.end),
                                      force=args.force,
                                      progress=lambda m: print(m, flush=True))
    print("FINAL SUMMARY:", summary, flush=True)
    print("COVERAGE:", store.coverage(), flush=True)

if __name__ == "__main__":
    main()
