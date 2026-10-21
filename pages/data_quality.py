import io
from datetime import date
import streamlit as st
import pandas as pd
from pages.common import get_context, demo_banner
from src.data_processing.validation import validate_dataframe
from src.data_ingestion.downloader import BhavcopyDownloader

settings, store = get_context()
st.title("✅ Data Quality & Ingestion Monitor")
demo_banner(settings)
dl = BhavcopyDownloader(settings, store)

cov = store.coverage()
st.write(f"**Stored data:** {cov.get('rows', 0):,} rows · {cov.get('symbols', 0)} symbols · "
         f"{cov.get('date_min') or '—'} → {cov.get('date_max') or '—'}")

if cov.get("has_data"):
    rep = validate_dataframe(store.load_all())
    st.subheader("Data-quality report")
    st.dataframe(pd.DataFrame([{"check": k, "value": str(v)} for k, v in rep.to_dict().items()]),
                 use_container_width=True)
    for issue in rep.issues:
        st.write(f"- {issue}")
    st.caption("Prices are unadjusted bhavcopy values. Corporate actions can create flagged extreme moves; "
               "missing weekday sessions include market holidays and are not automatically errors.")

st.subheader("Ingest new data")
c1, c2 = st.columns(2)
start = c1.date_input("Start date", value=date(2024, 1, 1))
end = c2.date_input("End date", value=date(2024, 1, 31))
st.caption("Downloads official NSE bhavcopy (legacy or UDiFF, auto-detected). Weekends and already-processed "
           "dates are skipped. If NSE blocks the request (403/404), the failure is logged below with its reason "
           "and the range can simply be re-run later — or use the local-file option.")
if st.button("Download & ingest range"):
    with st.spinner("Downloading… (bounded retries, failures are logged)"):
        recs = dl.ingest_range(start, end)
    ok = sum(1 for r in recs if r.status in ("success", "local_success"))
    failed = [r for r in recs if r.status == "failed"]
    st.write(f"**{ok}** days ingested · **{len(failed)}** failed · {len(recs) - ok - len(failed)} skipped.")
    for r in failed[:5]:
        st.error(f"{r.date}: {r.detail}")

st.subheader("Ingest a local bhavcopy file (recovery path)")
up = st.file_uploader("Official bhavcopy .zip / .csv / .csv.gz", type=["zip", "csv", "gz"])
if up is not None and st.button("Ingest uploaded file"):
    target = settings.raw_dir / up.name
    target.write_bytes(up.getvalue())
    try:
        rec = dl.ingest_local_file(target)
        st.success(f"{rec.status}: {rec.rows} rows from {rec.source_file}")
    except Exception as e:
        st.error(f"Could not ingest file: {e}")

st.subheader("Ingestion log")
log = dl.read_log()
if log:
    st.dataframe(pd.DataFrame(log).iloc[::-1], use_container_width=True, height=300)
else:
    st.info("No ingestion attempts logged yet.")
