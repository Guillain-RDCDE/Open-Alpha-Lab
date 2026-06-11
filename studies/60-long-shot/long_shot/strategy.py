"""The commodity skewness signal and the cross-sectional long-short.

Fuertes-Miffre-Fernandez (2015): rank commodities by the skewness of their recent daily returns; the
high-skew (lottery-like) names underperform. The trade is long low-skew, short high-skew. We compute
trailing-window skewness, sample it monthly, and measure the hedge.
"""
from __future__ import annotations
import numpy as np, pandas as pd
TRADING_DAYS = 252


def skewness_signal(daily: pd.DataFrame, window: int = TRADING_DAYS) -> pd.DataFrame:
    """Monthly skewness per commodity: the skew of its trailing ``window`` daily returns, month-end."""
    return daily.rolling(window).skew().resample("ME").last()


def monthly_returns(daily: pd.DataFrame) -> pd.DataFrame:
    return (1.0 + daily).resample("ME").prod() - 1.0


def cross_section_hedge(daily: pd.DataFrame, signal: pd.DataFrame, q: float = 0.33,
                        long_high: bool = False, min_names: int = 6) -> pd.Series:
    """Monthly long/short hedge on a cross-sectional ``signal`` (lagged). ``long_high=False`` is the
    textbook skewness trade: long the bottom-``q`` skew, short the top-``q``."""
    mret = monthly_returns(daily)
    sig = signal.reindex(mret.index)
    out = {}
    idx = mret.index
    for i in range(1, len(idx)):
        s = sig.loc[idx[i - 1]].dropna()
        n = mret.loc[idx[i]].dropna()
        common = s.index.intersection(n.index)
        if len(common) < min_names:
            continue
        s, n = s.loc[common], n.loc[common]
        lo, hi = s <= s.quantile(q), s >= s.quantile(1.0 - q)
        a, b = (hi, lo) if long_high else (lo, hi)
        out[idx[i]] = float(n[a].mean() - n[b].mean())
    return pd.Series(out, name="hedge").dropna()


def net_of_cost(hedge: pd.Series, cost_bps: float, turnover: float = 1.2) -> pd.Series:
    return (hedge - turnover * cost_bps / 1e4).rename(hedge.name)


def stats(spread: pd.Series, periods_per_year: int = 12) -> dict:
    r = pd.Series(spread).astype(float).dropna()
    if len(r) < 2:
        return {k: np.nan for k in ("mean_ann", "sharpe", "tstat", "hit_rate", "n")}
    sr_m = r.mean() / r.std(ddof=1) if r.std(ddof=1) > 0 else np.nan
    se = np.sqrt((1.0 + 0.5 * sr_m**2) / len(r))
    return {"mean_ann": float(r.mean() * periods_per_year), "sharpe": float(sr_m * np.sqrt(periods_per_year)),
            "tstat": float(sr_m / se) if se > 0 else np.nan, "hit_rate": float((r > 0).mean()), "n": int(len(r))}
