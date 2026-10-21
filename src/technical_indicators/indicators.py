"""Descriptive technical indicators. These describe historical price
behaviour — they are NOT predictions of future performance."""
from __future__ import annotations
import pandas as pd

def _check(df: pd.DataFrame, window: int, name: str) -> None:
    if window < 2:
        raise ValueError(f"{name} window must be >= 2, got {window}")
    if len(df) < window:
        raise ValueError(f"Insufficient observations for {name}: need >= {window}, have {len(df)}")

def sma(prices: pd.Series, window: int = 20) -> pd.Series:
    if window < 2: raise ValueError("SMA window must be >= 2")
    return prices.rolling(window).mean()

def ema(prices: pd.Series, window: int = 20) -> pd.Series:
    if window < 2: raise ValueError("EMA window must be >= 2")
    return prices.ewm(span=window, adjust=False).mean()

def rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    if window < 2: raise ValueError("RSI window must be >= 2")
    delta = prices.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / window, adjust=False).mean()
    import numpy as np
    rs = gain / loss.where(loss != 0)
    out = 100 - 100 / (1 + rs)
    out = out.where(loss != 0, 100.0)          # only gains -> RSI 100
    out = out.where(~((loss == 0) & (gain == 0)), 50.0)  # flat -> neutral
    return out.astype(float)

def macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    if fast >= slow: raise ValueError("MACD fast window must be < slow window")
    line = ema(prices, fast) - ema(prices, slow)
    sig = line.ewm(span=signal, adjust=False).mean()
    return pd.DataFrame({"macd": line, "macd_signal": sig, "macd_hist": line - sig})

def bollinger_bands(prices: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    if num_std <= 0: raise ValueError("Bollinger num_std must be > 0")
    mid = sma(prices, window)
    sd = prices.rolling(window).std(ddof=1)
    return pd.DataFrame({"bb_mid": mid, "bb_upper": mid + num_std * sd, "bb_lower": mid - num_std * sd})

def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """df needs high/low/close."""
    if window < 2: raise ValueError("ATR window must be >= 2")
    prev_close = df["close"].shift(1)
    tr = pd.concat([(df["high"] - df["low"]),
                    (df["high"] - prev_close).abs(),
                    (df["low"] - prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / window, adjust=False).mean()

def volume_sma(volume: pd.Series, window: int = 20) -> pd.Series:
    if window < 2: raise ValueError("Volume MA window must be >= 2")
    return volume.rolling(window).mean()

def add_indicators(df: pd.DataFrame, sma_windows=(20, 50), ema_window: int = 20,
                   rsi_window: int = 14, bb_window: int = 20, atr_window: int = 14,
                   vol_ma_window: int = 20) -> pd.DataFrame:
    """Return a copy with indicator columns. Validates history length."""
    out = df.sort_values("date").copy()
    close = out["close"]
    for w in sma_windows:
        _check(out, w, f"SMA{w}"); out[f"sma_{w}"] = sma(close, w)
    _check(out, ema_window, "EMA"); out[f"ema_{ema_window}"] = ema(close, ema_window)
    _check(out, rsi_window, "RSI"); out["rsi"] = rsi(close, rsi_window)
    m = macd(close); out["macd"], out["macd_signal"], out["macd_hist"] = m["macd"], m["macd_signal"], m["macd_hist"]
    b = bollinger_bands(close, bb_window); out["bb_upper"], out["bb_mid"], out["bb_lower"] = b["bb_upper"], b["bb_mid"], b["bb_lower"]
    _check(out, atr_window, "ATR"); out["atr"] = atr(out, atr_window)
    if "volume" in out: out[f"volume_ma_{vol_ma_window}"] = volume_sma(out["volume"], vol_ma_window)
    return out
