"""Strategy + inference for Study 503 — Expected Idiosyncratic Skewness.

The claim (Boyer, Mitton & Vorkink, 2010, *Expected Idiosyncratic Skewness*): stocks with high
*expected idiosyncratic skewness* — a fat right tail in their own-return distribution once the
market is stripped out — are lottery-like names investors over-pay for, and they *underperform*
going forward. We proxy expected idio-skew with **trailing residual skew** (the skewness of the
residual from a daily market regression over the prior 12 months) and trade **long the low-skew
quintile, short the high-skew quintile**, rebalanced monthly.

This module:

  * sorts the monthly idio-skew panel into **quintiles** each month and earns each quintile's
    *next-month* return (the panel is already lagged: skew at month-end pairs with month t+1's
    return — one execution lag, documented);
  * forms the **long-short** spread Q1 (low skew) − Q5 (high skew);
  * tests its mean with a **Newey-West (HAC) t-stat** and a **placebo / sign-flip null** (the
    small-sample workhorse), reports the **monthly win-rate** vs a 50% coin, and charges
    **one-way costs × turnover** plus a **short borrow** on the high-skew leg.

The decisive object is the HAC t of the long-short on the *real* tape: literature support for
the CRSP cross-sectional anomaly does not certify a tradable large-cap spread (the desk's t ≥ 2
bar, on the tape the study actually ran).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MONTHS_PER_YEAR = 12


# --------------------------------------------------------------------------- #
# Quintile sort
# --------------------------------------------------------------------------- #
def quintile_returns(panel: dict, n_q: int = 5) -> pd.DataFrame:
    """Equal-weight next-month return of each idio-skew quintile, month by month.

    For each month with at least ``2*n_q`` valid (idio-skew, fwd_ret) pairs, rank names by
    idiosyncratic skew, split into ``n_q`` quantiles (Q1 = lowest skew … Qn = highest skew),
    and average each quantile's forward return. Returns a DataFrame indexed by month with
    columns ``Q1..Qn``.
    """
    sk = panel["iskew"]
    fwd = panel["fwd_ret"]
    rows = {}
    for dt in sk.index:
        s = sk.loc[dt]
        f = fwd.loc[dt]
        both = pd.concat([s.rename("skew"), f.rename("fwd")], axis=1).dropna()
        if len(both) < 2 * n_q:
            continue
        try:
            q = pd.qcut(both["skew"].rank(method="first"), n_q, labels=False)
        except ValueError:
            continue
        means = both.groupby(q)["fwd"].mean()
        rows[dt] = {f"Q{int(i)+1}": means.get(i, np.nan) for i in range(n_q)}
    out = pd.DataFrame.from_dict(rows, orient="index")
    out.index.name = "date"
    return out.dropna(how="any")


def long_short(qret: pd.DataFrame, low: str = "Q1", high: str = "Q5",
               cost_bps: float = 0.0, borrow_ann_bps: float = 0.0,
               turnover: float = 1.0) -> pd.Series:
    """Long low-skew (``low``) / short high-skew (``high``) monthly spread, net of frictions.

    Gross spread is ``qret[low] - qret[high]`` (buy the symmetric tail, short the lottery tail).
    Costs: a one-way ``cost_bps`` × ``turnover`` charge is applied to *each* of the two legs
    every rebalance (so a full two-sided round-trip is ``2 * cost_bps * turnover`` per month,
    the conservative monthly-rebalance case). ``borrow_ann_bps`` charges short financing on the
    high-skew leg (annual, pro-rated monthly). ``turnover`` defaults to 1.0 (full monthly
    turnover — quintile membership churns for a monthly signal).
    """
    s = (qret[low] - qret[high]).dropna()
    monthly_cost = 2.0 * cost_bps * 1e-4 * turnover
    monthly_borrow = borrow_ann_bps * 1e-4 / MONTHS_PER_YEAR
    return s - monthly_cost - monthly_borrow


# --------------------------------------------------------------------------- #
# Leg / spread statistics
# --------------------------------------------------------------------------- #
def ann_return(r: pd.Series) -> float:
    """Geometric annualised (CAGR) of a monthly simple-return series."""
    r = r.dropna()
    if r.empty:
        return float("nan")
    growth = float((1.0 + r).prod())
    yrs = len(r) / MONTHS_PER_YEAR
    return growth ** (1.0 / yrs) - 1.0 if yrs > 0 and growth > 0 else float("nan")


def ann_vol(r: pd.Series) -> float:
    """Annualised volatility of a monthly simple-return series."""
    r = r.dropna()
    return float(r.std(ddof=1) * np.sqrt(MONTHS_PER_YEAR)) if len(r) > 1 else float("nan")


def sharpe(r: pd.Series) -> float:
    """Annualised Sharpe of a monthly series (the spread is self-financing, so rf = 0)."""
    r = r.dropna()
    if len(r) < 2 or r.std(ddof=1) == 0:
        return float("nan")
    return float(r.mean() / r.std(ddof=1) * np.sqrt(MONTHS_PER_YEAR))


def hac_tstat(r: pd.Series) -> float:
    """Newey-West (HAC) t-stat on the mean of a monthly return series.

    Same lag rule ``floor(4*(n/100)^(2/9))`` the desk uses elsewhere; no quantlab dependency.
    """
    x = r.dropna().to_numpy(dtype=float)
    n = x.size
    if n < 6:
        return float("nan")
    mu = x.mean()
    e = x - mu
    lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    return float(mu / se) if se > 0 else float("nan")


def placebo_pvalue(spread: pd.Series, n_draws: int = 20_000, seed: int = 503) -> dict:
    """Sign-flip placebo null for the long-short mean.

    Randomly flip the sign of each monthly spread observation ``n_draws`` times and ask how
    often the resampled mean is **>=** the observed mean (one-sided to the claim's direction:
    low-skew beats high-skew, so the edge should be *positive*). Returns the observed mean, the
    placebo mean (~0), and the empirical p-value.
    """
    x = spread.dropna().to_numpy(dtype=float)
    n = x.size
    if n < 6:
        return {"n": n, "obs_mean": float("nan"), "placebo_mean": float("nan"),
                "p_value": float("nan")}
    rng = np.random.default_rng(seed)
    obs = float(x.mean())
    means = np.empty(n_draws)
    for i in range(n_draws):
        signs = rng.choice([-1.0, 1.0], size=n)
        means[i] = (x * signs).mean()
    p = float((means >= obs).mean())
    return {"n": n, "obs_mean": obs, "placebo_mean": float(means.mean()), "p_value": p}


def win_rate(r: pd.Series) -> float:
    """Fraction of months the spread is positive (vs a 50% coin base rate)."""
    r = r.dropna()
    return float((r > 0).mean()) if len(r) else float("nan")


def spread_stats(s: pd.Series) -> dict:
    """Headline statistics for a spread series: n, annualised mean, vol, Sharpe, HAC t,
    win-rate, placebo p."""
    s = s.dropna()
    pl = placebo_pvalue(s)
    return {
        "n": int(len(s)),
        "mean_ann": float(s.mean() * MONTHS_PER_YEAR) if len(s) else float("nan"),
        "vol_ann": ann_vol(s),
        "sharpe": sharpe(s),
        "tstat": hac_tstat(s),
        "win": win_rate(s),
        "p_placebo": pl["p_value"],
    }


def quintile_summary(qret: pd.DataFrame) -> pd.DataFrame:
    """Per-quintile annualised mean, vol, Sharpe — the cross-sectional monotonicity card."""
    rows = {}
    for c in qret.columns:
        r = qret[c].dropna()
        rows[c] = {
            "mean_ann": float(r.mean() * MONTHS_PER_YEAR),
            "vol_ann": ann_vol(r),
            "sharpe": sharpe(r),
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def seed_robust_synth(edge: float, seeds=range(20), **kw) -> dict:
    """Average the long-short HAC t over many synthetic seeds (seed-robustness for any
    synthetic-dependent claim, per the house bar). Returns mean/std of mean%/yr and HAC t.

    Computes only the HAC t and annualised mean per seed (no 20k-draw placebo) so the 20-seed
    sweep stays fast enough for the notebooks and CI; the placebo null is reported separately on
    the single headline seed in :func:`spread_stats`.
    """
    from . import data as _data
    means, ts = [], []
    for sd in seeds:
        syn = _data.synthetic_panel(edge=edge, seed=int(sd), **kw)
        q = quintile_returns(syn)
        ls = long_short(q).dropna()
        means.append(float(ls.mean() * MONTHS_PER_YEAR) * 100)
        ts.append(hac_tstat(ls))
    means = np.array(means); ts = np.array(ts)
    return {"edge": edge, "n_seeds": len(means),
            "mean_pct": float(means.mean()), "mean_pct_sd": float(means.std(ddof=1)),
            "t_mean": float(ts.mean()), "t_sd": float(ts.std(ddof=1))}
