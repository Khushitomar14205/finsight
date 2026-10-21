# FinSight — AI-Powered NSE Market & Portfolio Research Agent

**Live app:** https://khushitomar14205-finsight-app-f2tuw5.streamlit.app/
**Dataset:** Real NSE bhavcopy, 2020-01-01 → 2026-10-06 — 4,106,326 rows, 7,085 symbols, 1,672 trading days (included in this repo as partitioned Parquet).

A production-minded financial research application for historical Indian equity data: NSE bhavcopy ingestion, market exploration, stock analytics with technical indicators, portfolio performance/risk/optimization, experimental ML forecasting, and a LangGraph AI research assistant that answers **only** from deterministic Python tool outputs — it never invents numbers.

> Historical closes are not live prices. Nothing here is investment advice, and the app never executes trades.

## Features

- **Data pipeline** — Programmatic NSE bhavcopy download with a swappable source adapter; legacy (pre-mid-2024) and UDiFF formats auto-detected and normalized to one schema; validation, de-duplication (one row per symbol-date, EQ series preferred), partitioned Parquet storage (year/month), incremental updates (weekends + already-processed dates skipped), bounded retries, and a full ingestion log with a local-file recovery path.
- **Market Explorer** — Symbol/date filtering, price & cumulative-return charts, volume, daily gainers/losers, underlying-record inspection, CSV download.
- **Stock Analysis** — Total/annualized return, volatility, Sharpe, max drawdown, best/worst days; SMA, EMA, RSI, MACD, Bollinger Bands, ATR, volume MA — all parameter-validated.
- **Stock Comparison** — Normalized performance, cumulative returns, correlation, drawdown, return distributions on a common date range with missing observations reported.
- **Portfolio Engine** — Manual weights/amounts, CSV import, saved portfolios; daily-rebalanced or buy-and-hold methodology (explicit); benchmark comparison; volatility, Sharpe (configurable risk-free rate), drawdown, correlation/covariance, component risk contributions.
- **Optimization & What-If** — Equal-weight, minimum-variance, and maximum-Sharpe allocations via scipy SLSQP with long-only, fully-invested, and max-weight constraints; interactive what-if weight sliders. Labelled as historical-model outputs.
- **ML Experiments** — Leakage-safe features (all shifted), chronological 60/20/20 splits, naive baseline first, Ridge / Random Forest / Gradient Boosting, MAE/RMSE/directional accuracy, and a transaction-cost-adjusted test-window backtest.
- **AI Assistant** — LangGraph workflow (route → tool → respond) with 15 registered tools. With an LLM key, the LLM only proposes a validated routing plan; without one, a deterministic rule-based router provides the same answers. Tool failures produce an honest explanation, never fabricated numbers.
- **Data Quality** — Required-column, OHLC-consistency, duplicate, non-positive-price, negative-volume, extreme-move and missing-session checks, surfaced in the UI.

## Architecture

```
app.py                  Streamlit entry (st.navigation, 11 pages)
pages/                  One module per page + common.py (shared context)
src/
  config/               Settings from env/.env; secrets never logged or rendered
  data_ingestion/       NSESourceAdapter (URLs) + BhavcopyDownloader (retry/log)
  data_processing/      normalize.py (legacy/UDiFF -> common schema), validation.py
  storage/              MarketDataStore — partitioned Parquet, filtered reads
  analytics/            Returns, volatility, Sharpe, drawdown, comparison
  technical_indicators/ SMA/EMA/RSI/MACD/Bollinger/ATR
  portfolio/            Performance & risk engine, CSV import, saved portfolios
  optimization/         scipy SLSQP allocations + what-if
  ml/                   Leakage-safe experiments
  agent/                LangGraph graph, 15 tools, prompts, state
  reporting/            Markdown reports + CSV export from computed results
  utils/                Structured logging + secret redaction
data/  raw/ processed/ metadata/   (git-ignored contents)
scripts/ ingest.py, generate_demo_data.py
tests/  pytest suite (synthetic data with known answers; no live API calls)
```

The LLM provider, data source, analytics, and UI are independent layers: swap any one without touching the others.

## Installation

```bash
cd finsight
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # optional: add GEMINI_API_KEY or OPENAI_API_KEY
```

## Configuration

| Variable | Meaning |
|---|---|
| `LLM_PROVIDER` | `gemini`, `openai`, or `none` |
| `GEMINI_API_KEY` / `OPENAI_API_KEY` | LLM key (env only — never hardcoded, logged, or shown in the UI) |
| `LLM_MODEL` | Model name for the provider |
| `RISK_FREE_RATE` | Default annual risk-free rate for Sharpe (e.g. `0.06`) |
| `TRADING_DAYS_PER_YEAR` | Annualization basis (default 252) |
| `FINSIGHT_DATA_DIR` | Data directory (default `data`) |

**Fallback mode:** with no key, every analytical page works and the AI assistant uses deterministic rule-based routing. Only LLM-written explanation prose is unavailable.

## Data ingestion

```bash
# Download a range (weekends/existing dates auto-skipped; failures logged, just re-run to retry)
python scripts/ingest.py --start 2025-01-01 --end 2025-01-31

# Recovery path: ingest an officially downloaded bhavcopy file placed in data/raw/
python scripts/ingest.py --local data/raw/cm02JAN2023bhav.csv.zip
```

Full-history ingestion was completed from this environment on 2026-10-21 using `scripts/ingest_bulk.py --start 2020-01-01 --end 2026-10-06` (monthly chunks, resumable). Final verified dataset: **4,106,326 rows, 7,085 symbols, 1,672 trading days, 2020-01-01 → 2026-10-06, 189 MB of partitioned Parquet**. Per-date final status: 1,672 days with data, 93 market holidays (weekdays where NSE published no bhavcopy — classified `holiday`, not `failed`), 0 unresolved failures. Data-quality validation on the full dataset: 0 duplicate symbol-dates, 0 OHLC violations, 0 non-positive prices, 0 negative volumes; 797 extreme (>50%) daily moves flagged (expected with unadjusted prices — splits/bonuses). One mid-run interruption (NSE read-timeouts followed by a process restart) was recovered simply by re-running the same command. Occasional HTTP 403 responses from NSE's Akamai layer are recorded in `data/metadata/ingestion_log.jsonl` with the HTTP detail and are retried on re-runs.

For trying the interface without real data:

```bash
python scripts/generate_demo_data.py   # SYNTHETIC, seeded, clearly flagged in the UI — not real market data
```

(Note: this replaces the stored dataset with synthetic data. The real 2020–2026 dataset described above is what the app ships with locally; don't run the demo script unless you want to reset to demo data — real data can be restored with `ingest_bulk.py`.)

## Run

```bash
streamlit run app.py
```

Works offline on locally stored data once ingested. Internet is needed only for ingestion and (optionally) the LLM.

## Tests

```bash
python -m pytest tests/ -q
```

26 tests: legacy/UDiFF parsing, zip handling, URL adapter, validation catches (OHLC, duplicates), known-answer return/drawdown/SMA checks, portfolio weighting and component-risk identity, optimization constraints, holdings CSV, ML chronological split, Parquet round-trip, agent routing and tool execution including graceful failure on unknown symbols. No live API calls.

## Known limitations

- Prices are **unadjusted** bhavcopy values; splits/bonuses can create artificial jumps (flagged as extreme moves in Data Quality). Return calculations inherit this limitation.
- Some symbols appear in multiple series intraday-file-wide; FinSight keeps one row per symbol-date, preferring EQ.
- No live prices or news feed is configured; the assistant will say so instead of guessing.
- Historical risk, optimized weights, and ML results are sample-period estimates — not forecasts or recommendations.
- NSE may rate-limit or block automated downloads; use the ingestion log + local-file path.

## Deployment

Local: `streamlit run app.py`. Streamlit Community Cloud: point it at this repo's `app.py`, add secrets in the Cloud secrets manager (not in the repo), and note that ingested Parquet data does not persist on ephemeral cloud disks — ingest on demand or mount external storage.

## Troubleshooting

- **"No local market data yet"** — run an ingestion range or the demo-data script.
- **403/404 in ingestion log** — NSE blocked/lacks that file (holiday or Akamai). Retry later or use `--local`.
- **Indicator error: insufficient observations** — widen the date range.
- **AI assistant in fallback mode** — set `LLM_PROVIDER` + matching key in `.env` and restart.
