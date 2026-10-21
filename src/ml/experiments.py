"""Experimental ML forecasting — historical research only.

Anti-leakage rules enforced here:
- Features at day t use only data up to day t; the target is day t+1's return.
- Chronological train/validation/test split. Never shuffled.
- A naive baseline is always reported first.
Historical patterns may not persist. No model here can reliably predict
prices or guarantee profit."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from ..technical_indicators.indicators import rsi as rsi_fn

def build_features(df_symbol: pd.DataFrame) -> pd.DataFrame:
    df = df_symbol.sort_values("date").copy()
    close = df["close"]
    ret = close.pct_change()
    feats = pd.DataFrame(index=df.index)
    feats["ret_lag1"] = ret.shift(1)
    feats["ret_lag2"] = ret.shift(2)
    feats["ret_lag5"] = ret.shift(5)
    feats["volatility_10"] = ret.shift(1).rolling(10).std()
    feats["volatility_21"] = ret.shift(1).rolling(21).std()
    feats["sma_ratio_20"] = close.shift(1) / close.shift(1).rolling(20).mean() - 1
    feats["sma_ratio_50"] = close.shift(1) / close.shift(1).rolling(50).mean() - 1
    feats["rsi_14"] = rsi_fn(close, 14).shift(1)
    if "volume" in df:
        feats["volume_change"] = df["volume"].pct_change().shift(1)
    feats["target_next_return"] = ret.shift(-1)
    feats["date"] = df["date"]
    return feats.dropna().reset_index(drop=True)

def run_experiment(df_symbol: pd.DataFrame, model_name: str = "random_forest",
                   transaction_cost_bps: float = 10.0) -> dict:
    feats = build_features(df_symbol)
    if len(feats) < 120:
        raise ValueError(f"Insufficient data for an ML experiment: need >= 120 usable rows after features, have {len(feats)}. "
                         "Try a longer history.")
    feature_cols = [c for c in feats.columns if c not in ("target_next_return", "date")]
    n = len(feats)
    n_train, n_val = int(n * 0.6), int(n * 0.2)
    train, val, test = feats.iloc[:n_train], feats.iloc[n_train:n_train + n_val], feats.iloc[n_train + n_val:]
    Xtr, ytr = train[feature_cols], train["target_next_return"]
    Xte, yte = test[feature_cols], test["target_next_return"]

    models = {
        "ridge": Ridge(alpha=1.0),
        "random_forest": RandomForestRegressor(n_estimators=200, max_depth=6, random_state=42, n_jobs=-1),
        "gradient_boosting": GradientBoostingRegressor(random_state=42),
    }
    if model_name not in models:
        raise ValueError(f"Unknown model '{model_name}'. Choose from {list(models)} or 'baseline'.")
    # Baseline: predict next return = 0 (no-change) — the honest benchmark.
    base_pred = np.zeros(len(yte))
    results = {}
    def metrics(y, p):
        return {"mae": float(mean_absolute_error(y, p)),
                "rmse": float(np.sqrt(mean_squared_error(y, p))),
                "directional_accuracy": float((np.sign(p) == np.sign(y.values if hasattr(y, 'values') else y)).mean())}
    results["baseline_zero_return"] = metrics(yte, base_pred)
    if model_name != "baseline":
        model = models[model_name]
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        results[model_name] = metrics(yte, pred)
        # Simple long/flat strategy on test period, net of transaction costs
        position = (pred > 0).astype(float)
        trades = np.abs(np.diff(np.concatenate([[0.0], position])))
        cost = trades * (transaction_cost_bps / 10000.0)
        strat_ret = position * yte.values - cost
        results["strategy_test"] = {
            "cumulative_strategy_return_after_costs": float(np.prod(1 + strat_ret) - 1),
            "cumulative_buy_and_hold_return_test": float(np.prod(1 + yte.values) - 1),
            "transaction_cost_bps_per_trade": transaction_cost_bps,
            "note": "Backtest on the held-out test window only; past performance does not predict future results.",
        }
        results["feature_importance"] = (dict(sorted(zip(feature_cols, model.feature_importances_.tolist()),
                                                     key=lambda kv: -kv[1])) if hasattr(model, "feature_importances_") else None)
    return {
        "symbol": str(df_symbol["symbol"].iloc[0]),
        "model": model_name, "task": "regression",
        "target": "next-day close-to-close return",
        "rows_used": int(n), "features": feature_cols,
        "split": {"train": [str(train['date'].min().date()), str(train['date'].max().date())],
                  "validation": [str(val['date'].min().date()), str(val['date'].max().date())] if len(val) else None,
                  "test": [str(test['date'].min().date()), str(test['date'].max().date())]},
        "split_counts": {"train": int(n_train), "validation": int(n_val), "test": int(n - n_train - n_val)},
        "split_method": "chronological 60/20/20, no shuffling, features use only past data; "
                        "model is fit on the training segment only and evaluated on the test segment",
        "results": results,
        "limitations": "Experimental research only. Financial time series are noisy and non-stationary; "
                       "directional accuracy near 50% is chance. This is not a price prediction or investment advice.",
    }
