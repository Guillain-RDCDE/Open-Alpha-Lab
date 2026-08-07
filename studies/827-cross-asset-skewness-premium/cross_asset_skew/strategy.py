"""Strategy + inference for Study 827 — Cross-Asset Skewness Premium.

The claim: the single-name realized-skewness reversal (Amaya, Christoffersen, Jacobs &
Vasquez 2015; Study 803) has an **asset-class analogue**. Measure each asset-class ETF's
**trailing realized skewness** of daily returns; each month sort the nine classes and go
**long the low-skew / short the high-skew** book. If lottery-overpricing operates at the
asset-class level, low-skew classes out-earn high-skew ones and the long-low/short-high
spread is positive.

This is distinct from:

* [803-realized-skewness-reversal](../../803-realized-skewness-reversal/) — the **single-name**
  version, sorting a cross-section of individual **stocks** on their own realized skewness.
  This study sorts a cross-section of **asset classes** (nine ETFs), one skew per class.
* [660-carry-everywhere](../../660-carry-everywhere/) — cross-asset **carry** (the yield /
  roll signal), not the third moment of the return distribution.
* [638-value-momentum-everywhere](../../638-value-momentum-everywhere/) — cross-asset **value**
  and **momentum** (level & trend signals), not skewness.

Method:

* **Daily total-return closes -> daily simple returns**, per asset-class ETF.
* **Trailing realized skewness.** On each asset the rolling ``window``-day sample skewness of
  daily returns (value on day ``t`` uses returns through ``t``), vectorised via the moment
  identity ``skew = m3 / m2**1.5`` (population moments, ddof=0).
* **Monthly point-in-time sort.** At each month-end the nine assets are ranked by the trailing
  skewness **known at that month-end**; the book is **held over the following month**. Long the
  bottom ``frac`` (low skew), short the top ``frac`` (high skew); equal weight. One documented
  execution lag (signal at month-end ``m-1``, hold month ``m``); zero look-ahead.
* **Inference.** Newey-West (HAC) *t* on the monthly long-short spread; a one-sample *t* and a
  pooled Welch *t* (low book vs high book) cross-check; an asset-label permutation placebo
  breaks the signal->outcome link; a costed timer charges the monthly round-trip friction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252
MONTHS = 12


# --------------------------------------------------------------------------- #
# Returns + signal
# --------------------------------------------------------------------------- #
def daily_returns(closes: pd.DataFrame) -> pd.DataFrame:
    """Daily simple total-return returns (index=date, columns=ticker)."""
    return closes.sort_index().pct_change()


def _skew(x: np.ndarray) -> float:
    """Sample skewness (bias-uncorrected, population moment; matches ``trailing_skew``)."""
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 3:
        return float("nan")
    m = x.mean()
    s = x.std(ddof=0)
    if s <= 0:
        return float("nan")
    return float(np.mean(((x - m) / s) ** 3))


def trailing_skew(ret: pd.DataFrame, window: int = 126) -> pd.DataFrame:
    """Rolling ``window``-day realized skewness of daily returns, per asset.

    Vectorised via the moment identity ``skew = m3 / m2**1.5`` where ``m2``/``m3`` are the
    population 2nd/3rd central moments (ddof=0, matching :func:`_skew`). Value on row ``t``
    uses returns through ``t`` (inclusive); :func:`skew_spreads` samples the signal at each
    month-end and holds the *following* month, so no forward information leaks.
    """
    r = ret
    mean = r.rolling(window, min_periods=window).mean()
    e2 = (r ** 2).rolling(window, min_periods=window).mean()
    e3 = (r ** 3).rolling(window, min_periods=window).mean()
    m2 = e2 - mean ** 2
    m3 = e3 - 3 * mean * e2 + 2 * mean ** 3
    out = m3 / m2.pow(1.5)
    return out.where(m2 > 0)


def monthly_returns(closes: pd.DataFrame) -> pd.DataFrame:
    """Per-asset **simple monthly** total returns (compounded from daily), month-end indexed."""
    r = daily_returns(closes)
    return (1.0 + r).resample("ME").prod(min_count=1) - 1.0


def monthly_signal(closes: pd.DataFrame, window: int = 126) -> pd.DataFrame:
    """Trailing realized skew sampled at each **month-end** (the value known at that close)."""
    sk = trailing_skew(daily_returns(closes), window)
    return sk.resample("ME").last()


# --------------------------------------------------------------------------- #
# The cross-sectional sort -> long-low-skew / short-high-skew monthly spread
# --------------------------------------------------------------------------- #
def skew_spreads(
    closes: pd.DataFrame,
    window: int = 126,
    frac: float = 0.34,
    min_names: int = 6,
) -> pd.DataFrame:
    """Monthly equal-weight bottom-minus-top realized-skew fractile spread across assets.

    At month-end ``m-1`` assets are ranked by the trailing skewness **known then**; the book is
    held over month ``m`` (the signal frame is ``shift(1)`` on the month grid). ``lo`` = mean
    month-``m`` return of the bottom ``frac`` (low skew, the long); ``hi`` = mean of the top
    ``frac`` (high skew, the short). ``spread = lo - hi`` (long low-skew, short high-skew).
    Months with fewer than ``min_names`` ranked assets are dropped.
    """
    sig_m = monthly_signal(closes, window).shift(1)   # known at end of prior month
    ret_m = monthly_returns(closes)
    # Align the two month grids.
    idx = sig_m.index.intersection(ret_m.index)
    sig_m = sig_m.loc[idx]
    ret_m = ret_m.loc[idx]
    S = sig_m.to_numpy(dtype=float)
    R = ret_m.to_numpy(dtype=float)

    out_spread, out_lo, out_hi, out_n, out_t = [], [], [], [], []
    for i in range(len(idx)):
        row = S[i]
        valid = np.where(~np.isnan(row))[0]
        n = len(valid)
        if n < min_names:
            continue
        k = max(1, int(np.floor(n * frac)))
        order = valid[np.argsort(row[valid], kind="stable")]
        low = order[:k]        # low skew  -> long
        high = order[-k:]      # high skew -> short
        rr = R[i]
        lo = float(np.nanmean(rr[low]))
        hi = float(np.nanmean(rr[high]))
        if np.isnan(lo) or np.isnan(hi):
            continue
        out_spread.append(lo - hi); out_lo.append(lo); out_hi.append(hi)
        out_n.append(n); out_t.append(idx[i])
    return pd.DataFrame(
        {"spread": out_spread, "lo": out_lo, "hi": out_hi, "n": out_n}, index=out_t
    ).sort_index()


# --------------------------------------------------------------------------- #
# Inference primitives
# --------------------------------------------------------------------------- #
def one_sample_t(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return float("nan")
    se = x.std(ddof=1) / np.sqrt(len(x))
    return float(x.mean() / se) if se > 0 else float("nan")


def welch_t(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return float((a.mean() - b.mean()) / se) if se > 0 else float("nan")


def newey_west_t(x: np.ndarray, lags: int = 6) -> float:
    """HAC (Newey-West, Bartlett kernel) t of mean(x) vs 0."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 3:
        return float("nan")
    mu = x.mean()
    u = x - mu
    gamma0 = float(u @ u) / n
    var = gamma0
    for l in range(1, min(lags, n - 1) + 1):
        w = 1.0 - l / (lags + 1.0)
        cov = float(u[l:] @ u[:-l]) / n
        var += 2.0 * w * cov
    if var <= 0:
        return float("nan")
    se = np.sqrt(var / n)
    return float(mu / se) if se > 0 else float("nan")


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    z2 = z * z
    mid = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return (mid - half, mid + half)


# --------------------------------------------------------------------------- #
# Headline stats
# --------------------------------------------------------------------------- #
def skew_stats(spreads: pd.DataFrame, nw_lags: int = 6) -> dict:
    sp = spreads["spread"].to_numpy(dtype=float)
    mean = float(np.nanmean(sp)) if len(sp) else float("nan")
    sd = float(np.nanstd(sp, ddof=1)) if len(sp) > 1 else float("nan")
    return {
        "n_months": int(len(spreads)),
        "spread_bps": mean * 1e4,
        "t_nw": newey_west_t(sp, nw_lags),
        "t_1s": one_sample_t(sp),
        "lo_bps": float(np.nanmean(spreads["lo"].to_numpy()) * 1e4) if len(spreads) else float("nan"),
        "hi_bps": float(np.nanmean(spreads["hi"].to_numpy()) * 1e4) if len(spreads) else float("nan"),
        "welch_t": welch_t(spreads["lo"].to_numpy(), spreads["hi"].to_numpy()),
        "sharpe": (mean / sd * np.sqrt(MONTHS)) if sd and sd > 0 else float("nan"),
    }


# --------------------------------------------------------------------------- #
# Placebo — is the spread real, or a lucky alignment of the sort?
# --------------------------------------------------------------------------- #
def placebo_pvalue(
    closes: pd.DataFrame,
    window: int = 126,
    frac: float = 0.34,
    min_names: int = 6,
    n_seeds: int = 20,
    n_draws_per_seed: int = 50,
    base_seed: int = 827,
) -> dict:
    """Keep the trailing-skew sort but read each month's forward return from an
    **asset-label-permuted** panel (signal->outcome link broken, each month's
    cross-sectional return distribution preserved). ``p`` = share of permuted worlds whose
    spread mean is >= observed (right-tail test on the long-low/short-high spread)."""
    sig_m = monthly_signal(closes, window).shift(1)
    ret_m = monthly_returns(closes)
    idx = sig_m.index.intersection(ret_m.index)
    sig_m = sig_m.loc[idx]; ret_m = ret_m.loc[idx]
    S = sig_m.to_numpy(dtype=float)
    R = ret_m.to_numpy(dtype=float)
    ncol = R.shape[1]

    obs = float(skew_spreads(closes, window, frac, min_names)["spread"].mean())

    rows_idx, lows, highs = [], [], []
    for i in range(len(idx)):
        row = S[i]
        valid = np.where(~np.isnan(row))[0]
        if len(valid) < min_names:
            continue
        k = max(1, int(np.floor(len(valid) * frac)))
        order = valid[np.argsort(row[valid], kind="stable")]
        rows_idx.append(i)
        lows.append(order[:k])
        highs.append(order[-k:])
    rows_idx = np.asarray(rows_idx)

    means: list[float] = []
    if len(rows_idx):
        M = R[rows_idx]
        kl = max(len(a) for a in lows)
        kh = max(len(a) for a in highs)

        def _pad(books, kmax):
            P = np.zeros((len(books), kmax), dtype=int)
            V = np.zeros((len(books), kmax), dtype=bool)
            for j, a in enumerate(books):
                P[j, :len(a)] = a
                V[j, :len(a)] = True
            return P, V

        LOW, LOWv = _pad(lows, kl)
        HIGH, HIGHv = _pad(highs, kh)
        rows_ar = np.arange(len(rows_idx))[:, None]

        def _masked_mean(pos, valid, perm):
            vals = M[rows_ar, perm[pos]]
            vals = np.where(valid, vals, np.nan)
            return np.nanmean(vals, axis=1)

        for seed in range(n_seeds):
            rng = np.random.default_rng(base_seed + seed)
            for _ in range(n_draws_per_seed):
                perm = rng.permutation(ncol)
                lo_v = _masked_mean(LOW, LOWv, perm)
                hi_v = _masked_mean(HIGH, HIGHv, perm)
                means.append(float(np.nanmean(lo_v - hi_v)))
    means = np.asarray(means)
    return {
        "obs_bps": obs * 1e4,
        "placebo_mean_bps": float(means.mean() * 1e4) if len(means) else float("nan"),
        "placebo_sd_bps": float(means.std(ddof=1) * 1e4) if len(means) > 1 else float("nan"),
        "p_value": float((means >= obs).mean()) if len(means) else float("nan"),
        "n_draws": int(len(means)),
        "draws_bps": means * 1e4,
    }


# --------------------------------------------------------------------------- #
# The costed timer
# --------------------------------------------------------------------------- #
def timer_stats(
    spreads: pd.DataFrame,
    cost_bps: float = 5.0,
    borrow_bps_yr: float = 50.0,
    turnover: float = 1.0,
) -> dict:
    """Cost the long-low-skew / short-high-skew monthly book.

    The book rebalances monthly. We charge ``2 sides x one-way cost x NAV x turnover`` per
    rebalance on the long-short book (``turnover=1`` is a full monthly replacement — the
    conservative bound; the sort actually rotates less), plus one month of borrow on the
    short leg (``borrow_bps_yr / 12``).
    """
    sp = spreads["spread"].to_numpy(dtype=float)
    sp = sp[~np.isnan(sp)]
    n = len(sp)
    round_trip_cost = 2.0 * turnover * cost_bps / 1e4
    borrow_month = (borrow_bps_yr / 1e4) / MONTHS
    net = sp - round_trip_cost - borrow_month
    gross_mean = float(sp.mean()) if n else float("nan")
    net_mean = float(net.mean()) if n else float("nan")
    sd = float(net.std(ddof=1)) if n > 1 else float("nan")
    sharpe = net_mean / sd * np.sqrt(MONTHS) if sd and sd > 0 else float("nan")
    return {
        "n_months": n,
        "gross_bps": gross_mean * 1e4,
        "net_bps": net_mean * 1e4,
        "cost_bps_per_month": (round_trip_cost + borrow_month) * 1e4,
        "ann_net_pct": net_mean * MONTHS * 100,
        "sharpe_net": sharpe,
        "t_net": one_sample_t(net),
    }


# --------------------------------------------------------------------------- #
# Synthetic-control detector (the machinery proof)
# --------------------------------------------------------------------------- #
def synthetic_detect(closes: pd.DataFrame, window: int = 126, frac: float = 0.34,
                     min_names: int = 3) -> dict:
    """Run the headline skew stats on a synthetic closes panel (positive control)."""
    sp = skew_spreads(closes, window, frac, min_names=min_names)
    ts = skew_stats(sp)
    return {"spread_bps": ts["spread_bps"], "t_nw": ts["t_nw"],
            "welch_t": ts["welch_t"], "n_months": ts["n_months"]}
