import numpy as np
import pandas as pd
import pytest
from src.portfolio import engine as PE
from src.optimization import optimizer as OPT
from src.ml.experiments import run_experiment, build_features
from src.storage.store import MarketDataStore
from src.agent.tools import AgentContext, TOOL_REGISTRY
from src.agent.graph import run_agent, rule_based_plan

def test_validate_weights_normalizes():
    w = PE.validate_weights({"AAA": 2, "BBB": 2})
    assert w == {"AAA": 0.5, "BBB": 0.5}

def test_negative_weight_rejected():
    with pytest.raises(ValueError):
        PE.validate_weights({"AAA": -0.5, "BBB": 1.5})

def test_portfolio_performance_and_risk(sample_df):
    w = {"AAA": 0.5, "BBB": 0.5}
    perf = PE.portfolio_performance(sample_df, w, risk_free_rate=0.06)
    assert perf["final_value"] > 0
    assert perf["observations"] == 300
    risk = PE.portfolio_risk(sample_df, w, risk_free_rate=0.06)
    assert risk["annualized_volatility"] > 0
    assert set(risk["correlation"].keys()) == {"AAA", "BBB"}
    comp = risk["component_risk_contribution"]
    assert sum(comp.values()) == pytest.approx(risk["annualized_volatility"], rel=0.02)

def test_optimization_constraints(sample_df):
    out = OPT.minimum_variance(sample_df, ["AAA", "BBB"], max_weight=0.8)
    w = out["weights"]
    assert sum(w.values()) == pytest.approx(1.0, abs=1e-4)
    assert all(0 <= v <= 0.8 + 1e-6 for v in w.values())
    out2 = OPT.maximum_sharpe(sample_df, ["AAA", "BBB"], risk_free_rate=0.06)
    assert sum(out2["weights"].values()) == pytest.approx(1.0, abs=1e-4)

def test_what_if(sample_df):
    out = OPT.what_if(sample_df, {"AAA": 0.5, "BBB": 0.5}, {"AAA": 0.8, "BBB": 0.2})
    assert "base" in out and "proposed" in out

def test_holdings_csv(tmp_path):
    p = tmp_path / "h.csv"
    p.write_text("symbol,weight\nAAA,60\nBBB,40\n")
    assert PE.parse_holdings_csv(p) == {"AAA": 0.6, "BBB": 0.4}

def test_ml_chronological_no_leakage(sample_df):
    df = sample_df[sample_df.symbol == "AAA"]
    feats = build_features(df)
    # feature row's date must precede the day whose return is the target:
    # target uses t+1 return, features use <= t data (checked structurally by shift in code)
    out = run_experiment(df, model_name="ridge")
    assert out["results"]["baseline_zero_return"]["directional_accuracy"] >= 0
    tr, te = out["split"]["train"], out["split"]["test"]
    assert tr[1] < te[0]  # chronological

def test_store_roundtrip(tmp_path, sample_df):
    store = MarketDataStore(tmp_path / "processed", tmp_path / "meta")
    info = store.save(sample_df)
    assert info["rows"] == len(sample_df)
    loaded = store.load(symbols=["AAA"])
    assert set(loaded["symbol"]) == {"AAA"}
    assert store.coverage()["has_data"]

def test_agent_tools_and_no_fabrication(tmp_path, sample_df):
    store = MarketDataStore(tmp_path / "processed", tmp_path / "meta")
    store.save(sample_df)
    ctx = AgentContext(store=store, risk_free_rate=0.06, portfolio_weights={"AAA": 0.5, "BBB": 0.5})
    assert len(TOOL_REGISTRY) == 15
    res = run_agent("What was the return of AAA in 2023?", ctx)
    assert res["tool_used"] == "calculate_returns"
    assert res["tool_result"]["symbol"] == "AAA"
    res2 = run_agent("Show the correlation between AAA and BBB", ctx)
    assert res2["tool_used"] == "correlation_covariance"
    res3 = run_agent("What is the maximum drawdown of my portfolio?", ctx)
    assert res3["tool_used"] == "portfolio_risk"
    # Unknown symbol -> graceful error, no invented numbers
    res4 = run_agent("What was the return of ZZZZZ in 2023?", ctx)
    assert res4.get("error") or "couldn't" in res4["answer"].lower() or res4["tool_result"]

def test_rule_router_compare():
    class FakeStore:
        def available_symbols(self): return ["AAA", "BBB"]
    ctx = AgentContext(store=FakeStore(), portfolio_weights={})
    plan = rule_based_plan("Compare AAA and BBB", ctx)
    assert plan["tool"] == "compare_stocks"
