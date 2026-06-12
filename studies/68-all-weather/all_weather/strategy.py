"""Risk parity (the "All-Weather" idea) vs the usual mixes. Weight each asset inversely to its recent
volatility, rebalance monthly, and compare risk-adjusted return and drawdown to equal-weight, 60/40 and
plain equities. The question isn't "does it return more" (unlevered, it won't) but "is the
diversification real" — better Sharpe, smaller drawdown — and whether the famous outperformance survives.
"""
from __future__ import annotations
import numpy as np, pandas as pd

TRADING_DAYS = 252


def inverse_vol_weights(vol: pd.Series) -> pd.Series:
    """Risk-parity (inverse-volatility) weights from a row of asset vols — normalised to sum to 1."""
    iv = 1.0 / vol.replace(0.0, np.nan)
    return (iv / iv.sum()).fillna(0.0)


def risk_parity(vol: pd.Series) -> pd.Series:
    return inverse_vol_weights(vol)


def equal_weight(vol: pd.Series) -> pd.Series:
    return pd.Series(1.0 / len(vol), index=vol.index)


def sixty_forty(vol: pd.Series, equity="SPY", bond="IEF") -> pd.Series:
    w = pd.Series(0.0, index=vol.index)
    w[equity] = 0.6; w[bond] = 0.4
    return w


def single(vol: pd.Series, asset="SPY") -> pd.Series:
    w = pd.Series(0.0, index=vol.index); w[asset] = 1.0
    return w


def backtest(ret: pd.DataFrame, weights_fn, lookback: int = 60) -> pd.Series:
    """Daily portfolio return for a monthly-rebalanced allocator. Weights at the start of each month are
    ``weights_fn(trailing_vol)`` using a ``lookback``-day vol; held through the month."""
    vol = ret.rolling(lookback).std()
    out, cur_w, last_mo = [], None, None
    for dt, row in ret.iterrows():
        mo = (dt.year, dt.month)
        if mo != last_mo:
            v = vol.loc[:dt].iloc[-1]
            cur_w = weights_fn(v.fillna(v.median())) if v.notna().any() else equal_weight(row)
            last_mo = mo
        out.append(float((row * cur_w).sum()))
    return pd.Series(out, index=ret.index, name="port")


def stats(r: pd.Series) -> dict:
    r = pd.Series(r).astype(float).dropna()
    if len(r) < 2:
        return {k: np.nan for k in ("ann", "vol", "sharpe", "max_drawdown", "n")}
    ann, vol = r.mean() * TRADING_DAYS, r.std(ddof=1) * np.sqrt(TRADING_DAYS)
    eq = (1.0 + r).cumprod(); dd = (eq / eq.cummax() - 1.0).min()
    return {"ann": float(ann), "vol": float(vol), "sharpe": float(ann / vol) if vol > 0 else np.nan,
            "max_drawdown": float(dd), "n": int(len(r))}


def compare(ret: pd.DataFrame) -> pd.DataFrame:
    """Stats table for risk-parity, equal-weight, 60/40 and SPY-only over the same window."""
    rows = {"risk-parity": backtest(ret, risk_parity), "equal-weight": backtest(ret, equal_weight),
            "60/40": backtest(ret, sixty_forty), "SPY only": backtest(ret, single)}
    return pd.DataFrame({k: stats(v) for k, v in rows.items()}).T
