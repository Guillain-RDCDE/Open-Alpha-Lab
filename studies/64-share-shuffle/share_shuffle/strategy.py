"""The net-issuance long-short, scored annually.

Net issuance = the year-on-year change in shares outstanding. The anomaly: high issuance precedes low
returns. The trade is long the low-issuance / buyback names, short the high-issuance ones. We measure
the hedge (long − short) vs next-year returns and its significance.
"""
from __future__ import annotations
import numpy as np, pandas as pd


def net_issuance(shares: pd.DataFrame) -> pd.DataFrame:
    """Year-on-year fractional change in shares outstanding (positive = net issuance)."""
    return shares.pct_change()


def quantile_hedge(signal: pd.DataFrame, fwd_ret: pd.DataFrame, q: float = 0.2, long_high: bool = True
                   ) -> pd.DataFrame:
    """Annual long-``q`` / short-``q`` hedge on a (year x ticker) ``signal`` vs next-year returns.
    For net issuance use ``long_high=False`` (long the LOW issuance / buybacks)."""
    rows = {}
    for y in signal.index:
        s = signal.loc[y].dropna()
        if len(s) < 20 or y not in fwd_ret.index:
            continue
        nxt = fwd_ret.loc[y].dropna()
        hi = s[s >= s.quantile(1.0 - q)].index
        lo = s[s <= s.quantile(q)].index
        rh = nxt.reindex(hi).dropna().mean()
        rl = nxt.reindex(lo).dropna().mean()
        if np.isnan(rh) or np.isnan(rl):
            continue
        long_leg, short_leg = (rh, rl) if long_high else (rl, rh)
        rows[y] = {"high": rh, "low": rl, "hedge": long_leg - short_leg}
    return pd.DataFrame(rows).T.sort_index()


def market_annual(fwd_ret: pd.DataFrame) -> pd.Series:
    return fwd_ret.mean(axis=1).rename("market").dropna()


def summary(annual_returns: pd.Series) -> dict:
    r = pd.Series(annual_returns).astype(float).dropna()
    if len(r) < 2:
        return {k: np.nan for k in ("mean", "vol", "sharpe", "tstat", "hit_rate", "max_drawdown", "n")}
    mean, std = r.mean(), r.std(ddof=1)
    eq = (1.0 + r).cumprod(); dd = (eq / eq.cummax() - 1.0).min()
    sr = mean / std if std > 0 else np.nan
    return {"mean": float(mean), "vol": float(std), "sharpe": float(sr),
            "tstat": float(sr * np.sqrt(len(r))) if std > 0 else np.nan,
            "hit_rate": float((r > 0).mean()), "max_drawdown": float(dd), "n": int(len(r))}
