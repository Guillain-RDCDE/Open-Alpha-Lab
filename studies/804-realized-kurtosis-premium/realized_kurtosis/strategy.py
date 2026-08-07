"""Strategy + inference for Study 804 — Realized-Kurtosis Premium.

The claim (Amaya, Christoffersen, Jacobs & Vasquez 2015): the same paper that finds a
strong negative realized-**skewness** relation also tests realized **kurtosis** — a
name's recent fat-tailedness — and there the signal is **weak / ambiguous**, mostly
subsumed by skewness and volatility. We sort a cross-section of stocks on their
**trailing realized kurtosis** and test the specified direction: **long the high-kurt**
(fat-tailed) names, **short the low-kurt** names, and measure the equal-weight spread.

This is distinct from:

* [803-realized-skewness-reversal](../../803-realized-skewness-reversal/) — the
  **third** moment (asymmetry / lottery right-tail) of a name's own returns; kurtosis is
  the **fourth** moment (fat-tailedness, symmetric), the sibling signal in the same
  paper;
* [501-idiosyncratic-volatility](../../501-idiosyncratic-volatility/) — the **second**
  moment (dispersion) of a name's residual returns; kurtosis is scale-free (it divides
  out the variance), so it is not a volatility tilt in disguise;
* [365-lottery-max-effect](../../365-lottery-max-effect/) — the single **maximum daily
  return** (MAX), a one-number extreme order statistic, not the full fourth moment of the
  distribution.

Method:

* **Close-to-close returns.** Build a per-name daily simple-return panel from adjusted
  Close.
* **Trailing realized kurtosis.** On each name compute the rolling ``window``-day
  population kurtosis of daily returns (``m4 / m2**2``, the 4th standardised central
  moment), vectorised via rolling raw moments (NOT ``rolling.apply``). Value on row ``t``
  uses returns through ``t``.
* **Point-in-time sort.** On each day ``t`` rank the cross-section by the trailing
  kurtosis known at the close of ``t-1`` (one ``shift``) and hold day ``t``. Long the top
  ``frac`` (high kurt), short the bottom ``frac`` (low kurt); equal weight.
* **Inference.** Newey-West (HAC) *t* on the daily long-short spread; a one-sample *t*
  and a pooled Welch *t* (high-kurt book vs low-kurt book) cross-check; a permutation
  placebo breaks the signal->outcome link; a costed timer charges the round-trip friction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# --------------------------------------------------------------------------- #
# Return panel + signal
# --------------------------------------------------------------------------- #
def close_returns(panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Daily simple close-to-close returns (index=date, columns=ticker)."""
    closes = pd.DataFrame({s: panel[s]["Close"] for s in panel}).sort_index()
    return closes.pct_change()


def _kurt(x: np.ndarray) -> float:
    """Population kurtosis (4th standardised central moment, m4/m2**2; ddof=0).

    Normal-distribution reference value is 3.0 (this is the *non-excess* kurtosis)."""
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 4:
        return float("nan")
    m = x.mean()
    d = x - m
    m2 = np.mean(d ** 2)
    if m2 <= 0:
        return float("nan")
    m4 = np.mean(d ** 4)
    return float(m4 / m2 ** 2)


def trailing_kurt(ret: pd.DataFrame, window: int = 21) -> pd.DataFrame:
    """Rolling ``window``-day realized kurtosis of daily returns, per name.

    Vectorised via the raw-moment identity ``kurt = m4 / m2**2`` where the central
    moments are expanded in rolling raw moments (matches :func:`_kurt`, ddof=0):

        m2 = e2 - mean**2
        m4 = e4 - 4*mean*e3 + 6*mean**2*e2 - 3*mean**4

    with ``ek`` the rolling mean of ``r**k``. NOT ``rolling.apply`` (which would be
    orders of magnitude slower). Value on row ``t`` uses returns through ``t``
    (inclusive); the sort in :func:`kurt_spreads` shifts by one day so a day-``t``
    position is formed on information known at ``t-1``.
    """
    r = ret
    mean = r.rolling(window, min_periods=window).mean()
    e2 = (r ** 2).rolling(window, min_periods=window).mean()
    e3 = (r ** 3).rolling(window, min_periods=window).mean()
    e4 = (r ** 4).rolling(window, min_periods=window).mean()
    m2 = e2 - mean ** 2
    m4 = e4 - 4 * mean * e3 + 6 * mean ** 2 * e2 - 3 * mean ** 4
    out = m4 / m2.pow(2)
    return out.where(m2 > 0)


# --------------------------------------------------------------------------- #
# The cross-sectional sort -> long-high-kurt / short-low-kurt spread
# --------------------------------------------------------------------------- #
def kurt_spreads(
    ret: pd.DataFrame,
    window: int = 21,
    frac: float = 0.3,
    min_names: int = 10,
) -> pd.DataFrame:
    """Daily equal-weight top-minus-bottom realized-kurtosis fractile spread.

    On each day ``t`` names are ranked by the trailing kurtosis known at the close of
    ``t-1`` (one ``shift``). ``hi`` = mean forward day-``t`` return of the top ``frac``
    (high kurt, the long); ``lo`` = mean of the bottom ``frac`` (low kurt, the short).
    ``spread = hi - lo`` (long high-kurt, short low-kurt). Days with fewer than
    ``min_names`` ranked names are dropped.
    """
    sig = trailing_kurt(ret, window).shift(1)  # known at close t-1
    S = sig.to_numpy(dtype=float)
    R = ret.to_numpy(dtype=float)
    idx = ret.index
    out_spread, out_lo, out_hi, out_n, out_t = [], [], [], [], []
    for i in range(len(idx)):
        row = S[i]
        valid = np.where(~np.isnan(row))[0]
        n = len(valid)
        if n < min_names:
            continue
        k = max(1, int(np.floor(n * frac)))
        order = valid[np.argsort(row[valid], kind="stable")]
        low = order[:k]        # low kurt  -> short
        high = order[-k:]      # high kurt -> long
        rr = R[i]
        lo = float(np.nanmean(rr[low]))
        hi = float(np.nanmean(rr[high]))
        out_spread.append(hi - lo); out_lo.append(lo); out_hi.append(hi)
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


def newey_west_t(x: np.ndarray, lags: int = 10) -> float:
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
def kurt_stats(spreads: pd.DataFrame, nw_lags: int = 10) -> dict:
    sp = spreads["spread"].to_numpy(dtype=float)
    return {
        "n_days": int(len(spreads)),
        "spread_bps": float(np.nanmean(sp) * 1e4),
        "t_nw": newey_west_t(sp, nw_lags),
        "t_1s": one_sample_t(sp),
        "lo_bps": float(np.nanmean(spreads["lo"].to_numpy()) * 1e4),
        "hi_bps": float(np.nanmean(spreads["hi"].to_numpy()) * 1e4),
        "welch_t": welch_t(spreads["hi"].to_numpy(), spreads["lo"].to_numpy()),
    }


# --------------------------------------------------------------------------- #
# Placebo — is the spread real, or a lucky alignment of the sort?
# --------------------------------------------------------------------------- #
def placebo_pvalue(
    ret: pd.DataFrame,
    window: int = 21,
    frac: float = 0.3,
    min_names: int = 10,
    n_seeds: int = 20,
    n_draws_per_seed: int = 50,
    base_seed: int = 804,
) -> dict:
    """Keep the trailing-kurt sort but read each day's forward return from a
    **column-permuted** panel (signal->outcome link broken, each day's
    cross-sectional distribution preserved). p = share of permuted worlds whose
    spread mean is >= observed (right-tail test on the long-high/short-low spread)."""
    cols = list(ret.columns)
    ncol = len(cols)
    sig = trailing_kurt(ret, window).shift(1)
    obs = float(kurt_spreads(ret, window, frac, min_names)["spread"].mean())

    ret_mat = ret.to_numpy(dtype=float)
    pos_of = {c: i for i, c in enumerate(cols)}
    rows_idx, lows, highs = [], [], []
    row_lookup = {t: r for r, t in enumerate(ret.index)}
    for t in ret.index:
        s = sig.loc[t].dropna()
        if len(s) < min_names:
            continue
        k = max(1, int(np.floor(len(s) * frac)))
        order = s.sort_values()
        rows_idx.append(row_lookup[t])
        lows.append(np.array([pos_of[c] for c in order.index[:k]]))
        highs.append(np.array([pos_of[c] for c in order.index[-k:]]))
    rows_idx = np.asarray(rows_idx)

    means = []
    if len(rows_idx):
        M = ret_mat[rows_idx]
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
                means.append(np.nanmean(hi_v - lo_v))
    means = np.asarray(means)
    return {
        "obs_bps": obs * 1e4,
        "placebo_mean_bps": float(means.mean() * 1e4) if len(means) else float("nan"),
        "placebo_sd_bps": float(means.std(ddof=1) * 1e4) if len(means) > 1 else float("nan"),
        "p_value": float((means >= obs).mean()) if len(means) else float("nan"),
        "n_draws": len(means),
        "draws_bps": means * 1e4,
    }


# --------------------------------------------------------------------------- #
# The costed timer
# --------------------------------------------------------------------------- #
def timer_stats(
    spreads: pd.DataFrame,
    cost_bps: float = 5.0,
    borrow_bps_yr: float = 50.0,
) -> dict:
    """Cost the long-high-kurt / short-low-kurt book.

    The signal is a trailing-window kurtosis that turns over roughly monthly, but names
    drift across the fractile boundary daily; we charge a conservative daily round-trip
    on the 2x-NAV book. To stay comparable to the desk's other cross-sectional timers we
    charge 2 sides x one-way cost x NAV per day on the long-short book, plus borrow on the
    short leg.
    """
    sp = spreads["spread"].to_numpy(dtype=float)
    sp = sp[~np.isnan(sp)]
    n = len(sp)
    round_trip_cost = 2.0 * cost_bps / 1e4
    borrow_daily = (borrow_bps_yr / 1e4) / 365.0
    net = sp - round_trip_cost - borrow_daily
    gross_mean = float(sp.mean())
    net_mean = float(net.mean())
    sd = float(net.std(ddof=1)) if n > 1 else float("nan")
    sharpe = net_mean / sd * np.sqrt(TRADING_DAYS) if sd and sd > 0 else float("nan")
    return {
        "n_days": n,
        "gross_bps": gross_mean * 1e4,
        "net_bps": net_mean * 1e4,
        "cost_bps_per_day": (round_trip_cost + borrow_daily) * 1e4,
        "ann_net_pct": net_mean * TRADING_DAYS * 100,
        "sharpe_net": sharpe,
        "t_net": one_sample_t(net),
    }


# --------------------------------------------------------------------------- #
# Synthetic-control detector (the machinery proof)
# --------------------------------------------------------------------------- #
def synthetic_detect(panel: dict[str, pd.DataFrame], window: int = 21,
                     frac: float = 0.3) -> dict:
    """Run the headline kurt stats on a synthetic panel."""
    ret = close_returns(panel)
    sp = kurt_spreads(ret, window, frac)
    ts = kurt_stats(sp)
    return {"spread_bps": ts["spread_bps"], "t_nw": ts["t_nw"],
            "welch_t": ts["welch_t"], "n_days": ts["n_days"]}
