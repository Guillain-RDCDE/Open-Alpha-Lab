"""Piotroski's F-score, computed annually, and the high-F / low-F long-short.

The F-score (Piotroski 2000) sums nine binary fundamental-health signals — profitability, leverage /
liquidity, and operating-efficiency trends — into a 0–9 score. The claim: high-F firms (8–9) beat low-F
firms (0–2). We build the score from EDGAR concepts and measure the annual hedge vs next-year returns.
"""
from __future__ import annotations
import numpy as np, pandas as pd

# the nine us-gaap concepts the F-score needs (year x ticker frames keyed by these names)
CONCEPTS = ("NetIncomeLoss", "NetCashProvidedByUsedInOperatingActivities", "Assets", "GrossProfit",
            "Revenues", "WeightedAverageNumberOfDilutedSharesOutstanding", "LongTermDebtNoncurrent",
            "AssetsCurrent", "LiabilitiesCurrent")


def piotroski_fscore(p: dict) -> pd.DataFrame:
    """The 0–9 F-score for each firm-year from a dict of (year x ticker) concept frames ``p``.

    Components — profitability: ROA>0, CFO>0, ΔROA>0, accrual (CFO>ROA); leverage/liquidity:
    Δleverage<0, Δcurrent-ratio>0, no net issuance; efficiency: Δgross-margin>0, Δasset-turnover>0.
    Returned only where the core inputs (ROA, CFO, assets) exist."""
    A = p["Assets"]
    roa = p["NetIncomeLoss"] / A
    cfo_a = p["NetCashProvidedByUsedInOperatingActivities"] / A
    lev = p["LongTermDebtNoncurrent"] / A
    curr = p["AssetsCurrent"] / p["LiabilitiesCurrent"]
    gm = p["GrossProfit"] / p["Revenues"]
    turn = p["Revenues"] / A
    iss = p["WeightedAverageNumberOfDilutedSharesOutstanding"].pct_change()
    d = lambda x: x.diff()
    f = ((roa > 0).astype(float) + (cfo_a > 0).astype(float) + (d(roa) > 0).astype(float)
         + (cfo_a > roa).astype(float) + (d(lev) < 0).astype(float) + (d(curr) > 0).astype(float)
         + (iss <= 0).astype(float) + (d(gm) > 0).astype(float) + (d(turn) > 0).astype(float))
    valid = roa.notna() & cfo_a.notna() & A.notna()
    return f.where(valid)


def quantile_hedge(signal: pd.DataFrame, fwd_ret: pd.DataFrame, q: float = 0.3, long_high: bool = True
                   ) -> pd.DataFrame:
    """Annual long-high / short-low hedge on a (year x ticker) ``signal`` vs next-year returns.
    For the F-score use ``long_high=True`` (long the high scores). ``q`` = tail fraction (terciles by
    default — F-scores are coarse, so wide buckets give breadth)."""
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
