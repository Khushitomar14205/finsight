import io, zipfile
import pandas as pd
import pytest
from src.data_processing.normalize import normalize, detect_format, read_bhavcopy_bytes
from src.data_processing.validation import validate_dataframe
from src.data_ingestion.downloader import NSESourceAdapter
from datetime import date

LEGACY = """SYMBOL,SERIES,OPEN,HIGH,LOW,CLOSE,LAST,PREVCLOSE,TOTTRDQTY,TOTTRDVAL,TIMESTAMP,TOTALTRADES,ISIN
RELIANCE,EQ,2400,2450,2380,2440,2440,2390,1000000,2440000000,02-JAN-2024,50000,INE002A01018
TCS,EQ,3400,3450,3380,3440,3440,3390,500000,1720000000,02-JAN-2024,30000,INE467B01029
"""
UDIFF = """TradDt,ISIN,TckrSymb,SctySrs,OpnPric,HghPric,LwPric,ClsPric,LastPric,PrvsClsgPric,TtlTradgVol,TtlTrfVal,TtlNbOfTxsExctd
2024-10-07,INE002A01018,RELIANCE,EQ,2900,2950,2880,2940,2940,2890,2000000,5880000000,90000
"""

def test_detect_and_normalize_legacy():
    raw = pd.read_csv(io.StringIO(LEGACY), dtype=str)
    assert detect_format(list(raw.columns)) == "legacy"
    df = normalize(raw, "legacy.csv")
    assert list(df["symbol"]) == ["RELIANCE", "TCS"]
    assert df.iloc[0]["close"] == 2440.0
    assert str(df.iloc[0]["date"].date()) == "2024-01-02"
    assert df.iloc[0]["volume"] == 1000000.0

def test_detect_and_normalize_udiff():
    raw = pd.read_csv(io.StringIO(UDIFF), dtype=str)
    assert detect_format(list(raw.columns)) == "udiff"
    df = normalize(raw, "udiff.csv")
    assert df.iloc[0]["symbol"] == "RELIANCE"
    assert df.iloc[0]["close"] == 2940.0
    assert str(df.iloc[0]["date"].date()) == "2024-10-07"

def test_unknown_schema_rejected():
    with pytest.raises(ValueError):
        normalize(pd.DataFrame({"foo": ["1"]}), "x.csv")

def test_zip_parsing():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("cm02JAN2024bhav.csv", LEGACY)
    raw = read_bhavcopy_bytes(buf.getvalue(), "test.zip")
    assert len(raw) == 2

def test_adapter_urls():
    a = NSESourceAdapter()
    old = a.urls_for(date(2023, 1, 2))
    assert old[0][0] == "legacy" and "cm02JAN2023bhav.csv.zip" in old[0][1]
    new = a.urls_for(date(2025, 1, 2))
    assert new[0][0] == "udiff" and "20250102" in new[0][1]

def test_validation_catches_problems(sample_df):
    bad = sample_df.copy()
    bad.loc[0, "high"] = bad.loc[0, "low"] - 10  # OHLC violation
    dup = pd.concat([bad, bad.iloc[[1]]], ignore_index=True)
    rep = validate_dataframe(dup)
    assert rep.ohlc_violations >= 1
    assert rep.duplicate_symbol_date >= 1

def test_validation_clean(sample_df):
    rep = validate_dataframe(sample_df)
    assert rep.rows == len(sample_df)
    assert rep.duplicate_symbol_date == 0
    assert rep.ohlc_violations == 0
