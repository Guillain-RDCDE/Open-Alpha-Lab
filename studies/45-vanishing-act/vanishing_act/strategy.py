"""The size premium (SMB), and the honest way to score a factor that predates its own data.

SMB = small-cap return minus large-cap return. The questions: is the long-run premium distinguishable
from zero, do small-caps even win on a *risk-adjusted* basis, and is the apparent "worked early, died
late" shape a real regime change or a window choice? We measure the spread, its Lo (2002) t-stat, the
two legs' Sharpes **on the same months** (never compare Sharpes across different samples), and any
sub-period split comes with a circular-block-bootstrap test of whether the two windows actually differ.

One discipline this module enforces by construction: Banz published in 1981 and the longest free proxy
starts in 1987, so *every* observation here is post-publication — there is no "before" on this bench,
and no split of this sample can demonstrate publication decay. A split can only show window-dependence,
and it is labelled and tested as such.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MONTHS = 12


def smb(returns: pd.DataFrame, small: str, large: str) -> pd.Series:
    """The size spread: monthly ``small − large`` return, dropping months either leg is missing."""
    df = returns[[small, large]].dropna()
    return (df[small] - df[large]).rename("smb")


def leg_summary(returns: pd.DataFrame, small: str, large: str) -> dict:
    """Stand-alone stats for both legs **on their common months** — aligned by construction.

    Comparing a 1993-start SPY Sharpe to a 2000-start IWM Sharpe smuggles the dot-com run-up into one
    leg only; this intersects the indexes first and reports the period it actually compared, so that
    mistake can't be made. Returns ``{"small": …, "large": …, "start", "end", "n"}``.
    """
    df = returns[[small, large]].dropna()
    return {
        "small": summary(df[small]),
        "large": summary(df[large]),
        "start": df.index.min(),
        "end": df.index.max(),
        "n": int(len(df)),
    }


def smb_stats(spread: pd.Series, periods_per_year: int = MONTHS) -> dict:
    """Annualised SMB mean, Sharpe, the Lo (2002) t-stat, and hit-rate."""
    r = pd.Series(spread).astype(float).dropna()
    if len(r) < 2:
        return {k: np.nan for k in ("mean_ann", "sharpe", "tstat", "hit_rate", "n")}
    sr_m = r.mean() / r.std(ddof=1) if r.std(ddof=1) > 0 else np.nan
    se = np.sqrt((1.0 + 0.5 * sr_m**2) / len(r))   # Lo (2002) SE on the per-period Sharpe
    return {
        "mean_ann": float(r.mean() * periods_per_year),
        "sharpe": float(sr_m * np.sqrt(periods_per_year)),
        "tstat": float(sr_m / se) if se > 0 else np.nan,
        "hit_rate": float((r > 0).mean()),
        "n": int(len(r)),
    }


def window_split(spread: pd.Series, cut_year: int = 2010) -> pd.DataFrame:
    """SMB before vs after ``cut_year`` — a **window split**, not a decay test.

    On a sample that is entirely post-publication, a split can only show that the spread's sign
    depends on the window. Pair it with :func:`split_difference` so the pre/post gap carries an
    uncertainty, not just a sign.
    """
    r = pd.Series(spread).astype(float).dropna()
    out = {}
    for lab, sub in [(f"pre-{cut_year}", r[r.index.year < cut_year]),
                     (f"{cut_year}-on", r[r.index.year >= cut_year])]:
        s = smb_stats(sub)
        out[lab] = {"mean_ann": s["mean_ann"], "sharpe": s["sharpe"], "n": s["n"]}
    return pd.DataFrame(out).T


def split_difference(spread: pd.Series, cut_year: int = 2010, n_boot: int = 5000,
                     seed: int = 45, block_size: int | None = None) -> dict:
    """Is the pre/post-``cut_year`` difference in mean SMB real, or a window artefact?

    Circular block bootstrap (Politis & Romano 1994) on each segment independently — block length
    ``round(n^(1/3))`` by default, preserving the spread's serial dependence — then the two-sided
    bootstrap p-value of ``mean(pre) − mean(post)``. Returns annualised means, the difference, and
    ``p_value`` (≥ ~0.05 ⇒ the celebrated "turn" is indistinguishable from picking a lucky window).
    """
    r = pd.Series(spread).astype(float).dropna()
    pre = r[r.index.year < cut_year].to_numpy()
    post = r[r.index.year >= cut_year].to_numpy()
    if len(pre) < 24 or len(post) < 24:
        return {k: np.nan for k in ("pre_ann", "post_ann", "diff_ann", "p_value", "n_pre", "n_post")}
    rng = np.random.default_rng(seed)

    def cbb_means(x: np.ndarray) -> np.ndarray:
        n = x.size
        blk = int(block_size) if block_size is not None else max(1, round(n ** (1.0 / 3.0)))
        blk = max(1, min(blk, n))
        n_blocks = int(np.ceil(n / blk))
        starts = rng.integers(0, n, (n_boot, n_blocks))
        idx = (starts[:, :, None] + np.arange(blk)[None, None, :]) % n
        return x[idx.reshape(n_boot, -1)[:, :n]].mean(axis=1)

    diffs = cbb_means(pre) - cbb_means(post)
    obs = pre.mean() - post.mean()
    p = 2.0 * min(float((diffs <= 0).mean()), float((diffs >= 0).mean()))
    return {
        "pre_ann": float(pre.mean() * MONTHS),
        "post_ann": float(post.mean() * MONTHS),
        "diff_ann": float(obs * MONTHS),
        "p_value": min(p, 1.0),
        "n_pre": int(pre.size),
        "n_post": int(post.size),
    }


def rolling_mean_ann(spread: pd.Series, window_months: int = 120) -> pd.Series:
    """Trailing ``window_months`` annualised mean of the spread — the sign-flip exhibit.

    If a '10-year size premium' swings between clearly positive and clearly negative depending on
    where the decade ends, the 'premium' is a window choice, not a property of the asset class.
    """
    r = pd.Series(spread).astype(float).dropna()
    return (r.rolling(window_months).mean() * MONTHS).rename("smb_rolling_ann")


def summary(returns: pd.Series, periods_per_year: int = MONTHS) -> dict:
    """Annualised Sharpe, CAGR, vol, max-drawdown for a monthly return series."""
    r = pd.Series(returns).astype(float).dropna()
    if len(r) < 2:
        return {k: np.nan for k in ("sharpe", "cagr", "vol_ann", "max_drawdown", "n")}
    mean, std = r.mean(), r.std(ddof=1)
    eq = (1.0 + r).cumprod()
    dd = (eq / eq.cummax() - 1.0).min()
    years = len(r) / periods_per_year
    cagr = eq.iloc[-1] ** (1.0 / years) - 1.0 if eq.iloc[-1] > 0 else np.nan
    return {
        "sharpe": float(mean / std * np.sqrt(periods_per_year)) if std > 0 else np.nan,
        "cagr": float(cagr),
        "vol_ann": float(std * np.sqrt(periods_per_year)),
        "max_drawdown": float(dd),
        "n": int(len(r)),
    }
