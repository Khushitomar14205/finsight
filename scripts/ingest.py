"""CLI ingestion.
Usage:
  python scripts/ingest.py --start 2024-01-01 --end 2024-01-31
  python scripts/ingest.py --local data/raw/cm01JAN2024bhav.csv.zip
Failures are logged to data/metadata/ingestion_log.jsonl and can be retried
by re-running the same range (already-processed dates are skipped)."""
import argparse, sys
from datetime import date
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config.settings import get_settings
from src.storage.store import MarketDataStore
from src.data_ingestion.downloader import BhavcopyDownloader

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start"); ap.add_argument("--end")
    ap.add_argument("--local", help="Ingest a local official bhavcopy file instead of downloading")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    s = get_settings()
    store = MarketDataStore(s.processed_dir, s.metadata_dir)
    dl = BhavcopyDownloader(s, store)
    if args.local:
        rec = dl.ingest_local_file(args.local)
        print(rec); return
    if not (args.start and args.end):
        ap.error("--start and --end are required unless --local is used")
    recs = dl.ingest_range(date.fromisoformat(args.start), date.fromisoformat(args.end), force=args.force)
    ok = sum(1 for r in recs if r.status in ("success", "local_success"))
    failed = [r for r in recs if r.status == "failed"]
    print(f"Done: {ok} succeeded, {len(failed)} failed, {len(recs)} total days considered.")
    for r in failed[:10]:
        print("FAILED", r.date, r.detail)

if __name__ == "__main__":
    main()
