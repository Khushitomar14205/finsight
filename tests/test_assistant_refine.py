"""Assistant refinement tests: availability detail, comparison table +
interpretation + corporate-action warning, ML reporting, and
agent-answer == calculated-result consistency. Deterministic data only."""
import numpy as np
import pandas as pd
import pytest
from src.storage.store import MarketDataStore
from src.agent.tools import AgentContext, check_data_availability, calculate_returns, compare_stocks
from src.agent.humanize import humanize
from src.agent.graph import run_agent
from src.analytics import returns as R
from src.ml.experiments import run_experiment

def _store(tmp_path, df):
    store = MarketDataStore(tmp_path / "processed", tmp_path / "meta")
    store.save(df)
    return store

def test_availability_records_vs_days_and_latest_close(tmp_path, sample_df):
    store = _store(tmp_path, sample_df)
    ctx = AgentContext(store=store)
    out = check_data_availability(ctx, symbols=["AAA"])
    a = out["AAA"]
    assert a["rows"] == 300 and a["trading_days"] == 300
    last = sample_df[sample_df["symbol"] == "AAA"].sort_values("date").iloc[-1]
    assert a["latest_close"] == pytest.approx(float(last["close"]))
    assert a["latest_close_date"] == str(last["date"].date())
    text = humanize("check_data_availability", out, {})
    assert "trading days" in text and "Latest close" in text and "not necessarily continuous" in text

def test_compare_answer_table_interpretation_and_warning():
    result = {"common_start": "2024-01-01", "common_end": "2024-12-31",
              "missing_observations": {"AAA": 0, "BBB": 3},
              "symbols": {
                  "AAA": {"total_return": -0.5308, "annualized_return": -0.5408, "annualized_volatility": 0.5541,
                          "sharpe_ratio": -1.11, "max_drawdown": -0.6236,
                          "worst_day": {"date": "2024-10-28", "return": -0.4976},
                          "best_day": {"date": "2024-01-29", "return": 0.0702}},
                  "BBB": {"total_return": 0.0744, "annualized_return": 0.0766, "annualized_volatility": 0.2111,
                          "sharpe_ratio": 0.17, "max_drawdown": -0.13,
                          "worst_day": {"date": "2024-06-04", "return": -0.045},
                          "best_day": {"date": "2024-02-01", "return": 0.05}}}}
    text = humanize("compare_stocks", result, {})
    assert "| Symbol | Total return |" in text and "-53.08%" in text and "+7.44%" in text
    assert "BBB had the higher observed return" in text
    assert "AAA was the more volatile" in text
    assert "Data-quality warning — AAA" in text and "bonus" in text
    assert "BBB: 3" in text  # missing observations explained

def test_ml_reporting_fields_and_chronology(sample_df):
    df = sample_df[sample_df["symbol"] == "AAA"]
    out = run_experiment(df, model_name="random_forest")
    assert out["model"] == "random_forest" and out["target"]
    c = out["split_counts"]
    assert c["train"] + c["validation"] + c["test"] == out["rows_used"]
    assert out["split"]["train"][1] < out["split"]["test"][0]      # chronological, no overlap
    res = out["results"]
    assert "baseline_zero_return" in res and "random_forest" in res
    for m in (res["baseline_zero_return"], res["random_forest"]):
        assert set(("mae", "rmse", "directional_accuracy")) <= set(m)
    text = humanize("run_ml_experiment", out, {})
    assert "MAE" in text and "Baseline" in text and "Limitations" in text and "{" not in text

def test_agent_answer_matches_calculation(tmp_path, sample_df):
    store = _store(tmp_path, sample_df)
    ctx = AgentContext(store=store, risk_free_rate=0.06)
    tool_out = calculate_returns(ctx, symbols=["AAA"], start="2023-01-01", end="2023-12-31")
    res = run_agent("What was the return of AAA in 2023?", ctx)
    expected = f"{tool_out['total_return']:+.2%}"
    assert expected in res["answer"], f"expected {expected} in answer"
    assert res["tool_used"] == "calculate_returns"
    # Start/end prices in the answer are the tool's actual observations
    assert f"{tool_out['first_close']:,.2f}" in res["answer"]
    assert f"{tool_out['last_close']:,.2f}" in res["answer"]

def test_agent_compare_uses_common_period(tmp_path, sample_df):
    store = _store(tmp_path, sample_df)
    ctx = AgentContext(store=store, risk_free_rate=0.06)
    tool_out = compare_stocks(ctx, symbols=["AAA", "BBB"], start="2023-01-01", end="2023-12-31")
    res = run_agent("Compare AAA and BBB in 2023", ctx)
    assert tool_out["common_start"] in res["answer"] and tool_out["common_end"] in res["answer"]
    for sym, s in tool_out["symbols"].items():
        assert f"{s['total_return']:.2%}" in res["answer"] or f"{s['total_return']:+.2%}" in res["answer"]

def test_explain_lines_reflect_tool_and_result():
    from src.agent.explain import explain_lines
    lines = explain_lines("calculate_returns",
                          {"start": "2023-01-02", "end": "2023-12-29", "observations": 245})
    joined = " ".join(lines)
    assert "2023-01-02" in joined and "245" in joined and "first close" in joined
    assert "252" in joined and "dividends are not included" in joined
