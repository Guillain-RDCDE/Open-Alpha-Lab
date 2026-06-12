"""The pre-FOMC announcement drift (Lucca & Moench 2015): equities drift up in the ~24 hours before a
scheduled FOMC statement. We tag the trading day *before* each announcement and compare its return to
all other days, then split the sample to test whether the drift survived publication.
"""
from __future__ import annotations
import numpy as np, pandas as pd

from .data import TRADING_DAYS


def pre_fomc_mask(index: pd.DatetimeIndex, fomc_dates: pd.DatetimeIndex, lead: int = 1) -> pd.Series:
    """Boolean over ``index``: True on the trading day ``lead`` sessions *before* each FOMC announcement
    (lead=1 ⇒ the day before; the Lucca-Moench pre-FOMC window). Robust to announcement dates that fall
    on a non-trading day (snaps to the prior session)."""
    idx = pd.DatetimeIndex(index)
    mask = pd.Series(False, index=idx)
    for d in pd.DatetimeIndex(fomc_dates):
        pos = idx.searchsorted(d)            # first session >= d
        j = pos - lead                       # the pre-FOMC session
        if 0 <= j < len(idx):
            mask.iloc[j] = True
    return mask


def drift_table(ret: pd.Series, fomc_dates: pd.DatetimeIndex, lead: int = 1) -> dict:
    """Compare pre-FOMC days to all other days: mean daily return, t-stat, and the share of the total
    cumulative return earned on the (rare) pre-FOMC days."""
    ret = ret.dropna()
    mask = pre_fomc_mask(ret.index, fomc_dates, lead).reindex(ret.index).fillna(False)
    pre, rest = ret[mask], ret[~mask]
    n_pre = int(mask.sum())
    # Welch t on the difference in means
    sp, sr = pre.std(ddof=1), rest.std(ddof=1)
    se = np.sqrt(sp**2 / len(pre) + sr**2 / len(rest)) if len(pre) > 1 else np.nan
    t = (pre.mean() - rest.mean()) / se if se and se > 0 else np.nan
    total = ret.sum()
    return {"pre_mean": float(pre.mean()), "rest_mean": float(rest.mean()),
            "pre_ann": float(pre.mean() * n_pre / (len(ret) / TRADING_DAYS)),  # avg pre-FOMC P&L per year
            "tstat": float(t), "n_pre": n_pre, "n_total": int(len(ret)),
            "pre_share": float(pre.sum() / total) if total != 0 else np.nan,
            "pre_sum": float(pre.sum()), "total_sum": float(total)}


def split_by_date(ret: pd.Series, fomc_dates: pd.DatetimeIndex, cut: str = "2011-01-01", lead: int = 1):
    """Drift table before vs after ``cut`` (the Lucca-Moench publication era) — does the drift survive?"""
    cut = pd.Timestamp(cut)
    pre_era = drift_table(ret[ret.index < cut], fomc_dates, lead)
    post_era = drift_table(ret[ret.index >= cut], fomc_dates, lead)
    return {"pre_publication": pre_era, "post_publication": post_era}


def summary(daily: pd.Series) -> dict:
    r = pd.Series(daily).astype(float).dropna()
    if len(r) < 2:
        return {k: np.nan for k in ("mean", "vol", "sharpe", "n")}
    mean, std = r.mean(), r.std(ddof=1)
    return {"mean": float(mean), "vol": float(std),
            "sharpe": float(mean / std * np.sqrt(TRADING_DAYS)) if std > 0 else np.nan, "n": int(len(r))}
