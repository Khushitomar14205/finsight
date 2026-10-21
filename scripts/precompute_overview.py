"""Precompute homepage statistics (data/metadata/overview_stats.json).

Run after ingestion so the homepage can render from a small JSON instead of
scanning 4.1M rows on every load (important on small cloud containers).
Usage: python scripts/precompute_overview.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config.settings import get_settings
from src.storage.store import MarketDataStore
from src.reporting.overview_stats import build_overview_stats

if __name__ == "__main__":
    s = get_settings()
    store = MarketDataStore(s.processed_dir, s.metadata_dir)
    stats = build_overview_stats(store)
    print("wrote overview_stats.json for", stats.get("generated_for_latest_date"))
