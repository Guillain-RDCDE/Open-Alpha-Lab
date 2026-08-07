"""Strategy + inference for Study 814 — Trailing-Sharpe Anomaly.

The claim (risk-adjusted momentum): sort a cross-section on each name's **trailing
12-month Sharpe ratio** — mean of daily returns over the formation window divided by
their standard deviation, **skipping the most recent month** exactly as 12-1 momentum
does. Long the **high-Sharpe** names, short the **low-Sharpe** names. The honest question
is whether risk-adjusting *helps* or whether a Sharpe sort is merely **plain 12-1
momentum + a low-vol tilt repackaged**.

This is distinct from:

* [507-cross-sectional-momentum](../../507-cross-sectional-momentum/) — plain **12-1
  price momentum** (cumulative formation-window return, no risk adjustment); this study
  divides that signal by realized volatility, and we run 507's signal head-to-head as
  the comparator here.
* [8-true-strength](../../8-true-strength/) — a smoothed **trend/oscillator** strength,
  not a moment ratio of the raw return distribution.
* [330-low-volatility-anomaly](../../330-low-volatility-anomaly/) — sorting on
  **volatility alone** (the denominator); a Sharpe sort couples that denominator to a
  momentum numerator, and we decompose which leg does the work.
* [237-residual-momentum](../../237-residual-momentum/) — momentum in **factor-model
  residuals**; this study risk-scales *total* return momentum, not a residual.

Method:

* **Close-to-close returns.** Build a per-name daily simple-return panel from adjusted
  Close.
* **Trailing 12-1 Sharpe.** On each name, the ``lookback``-day (≈12m) mean/std of daily
  returns, formed on the window that **ends ``skip`` (≈1m) days ago** — the value on row
  ``t`` uses returns through ``t-skip``, skipping the most recent month.
* **Plain 12-1 momentum (the comparator).** The same window's cumulative return, no risk
  adjustment — the signal this study is really being graded against.
* **Point-in-time sort.** On each day ``t`` rank the cross-section by the signal known at
  the close of ``t-1`` (one extra ``shift``) and hold day ``t``. Long the **top** ``frac``
  (high signal), short the **bottom** ``frac`` (low signal); equal weight.
* **Inference.** Newey-West (HAC) *t* on the daily long-short spread; a one-sample *t*
  and a pooled Welch *t* (high book vs low book) cross-check; a permutation placebo
  breaks the signal->outcome link; a costed timer charges the round-trip friction; a
  head-to-head against plain momentum and a low-vol sort answers the repackaging question.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# --------------------------------------------------------------------------- #
# Return panel + signals
# --------------------------------------------------------------------------- #
def close_returns(panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Daily simple close-to-close returns (index=date, columns=ticker)."""
    closes = pd.DataFrame({s: panel[s]["Close"] for s in panel}).sort_index()
    return closes.pct_change()


def trailing_sharpe(ret: pd.DataFrame, lookback: int = 252, skip: int = 21) -> pd.DataFrame:
    """Trailing 12-1 **Sharpe** signal: rolling mean/std of daily returns over the
    ``lookback``-day formation window that **ends ``skip`` days ago**.

    Vectorised: the rolling ``lookback``-day mean and (population, ddof=0) std of daily
    returns give the Sharpe of the window *ending on each row*; a ``.shift(skip)`` then
    skips the most recent month, so the value on row ``t`` uses returns through ``t-skip``.
    The sort in :func:`fractile_spreads` shifts by one further day so a day-``t`` position
    is formed on information known at the close of ``t-1`` (zero look-ahead).
    """
    mean = ret.rolling(lookback, min_periods=lookback).mean()
    std = ret.rolling(lookback, min_periods=lookback).std(ddof=0)
    sharpe = (mean / std).where(std > 0)
    return sharpe.shift(skip)


def trailing_momentum(ret: pd.DataFrame, lookback: int = 252, skip: int = 21) -> pd.DataFrame:
    """Plain 12-1 **momentum** comparator: cumulative return over the same window that
    ends ``skip`` days ago (no risk adjustment). Vectorised via a rolling sum of
    ``log1p`` returns, then ``expm1``; value on row ``t`` uses returns through ``t-skip``.
    """
    logret = np.log1p(ret)
    cum = logret.rolling(lookback, min_periods=lookback).sum()
    mom = np.expm1(cum)
    return mom.shift(skip)


def trailing_vol(ret: pd.DataFrame, lookback: int = 252, skip: int = 21) -> pd.DataFrame:
    """Trailing realized **volatility** (the Sharpe denominator alone) — used to sort a
    pure low-vol book (long low-vol / short high-vol via ``long_high=False``). Value on
    row ``t`` uses returns through ``t-skip``."""
    std = ret.rolling(lookback, min_periods=lookback).std(ddof=0)
    return std.shift(skip)


# --------------------------------------------------------------------------- #
# The cross-sectional sort -> long-high / short-low spread
# --------------------------------------------------------------------------- #
def fractile_spreads(
    ret: pd.DataFrame,
    sig: pd.DataFrame,
    frac: float = 0.3,
    min_names: int = 10,
    long_high: bool = True,
) -> pd.DataFrame:
    """Daily equal-weight top-minus-bottom fractile spread for an arbitrary signal.

    On each day ``t`` names are ranked by the signal known at the close of ``t-1`` (one
    ``shift``). With ``long_high=True`` (the Sharpe/momentum default) ``hi`` = mean
    forward day-``t`` return of the top ``frac`` (high signal, the long), ``lo`` = mean of
    the bottom ``frac`` (the short), and ``spread = hi - lo`` (long high, short low). With
    ``long_high=False`` the legs flip (long the bottom, e.g. a low-vol book). Days with
    fewer than ``min_names`` ranked names are dropped.
    """
    S = sig.shift(1).to_numpy(dtype=float)   # known at close t-1
    R = ret.to_numpy(dtype=float)
    idx = ret.index
    out_spread, out_hi, out_lo, out_n, out_t = [], [], [], [], []
    for i in range(len(idx)):
        row = S[i]
        valid = np.where(~np.isnan(row))[0]
        n = len(valid)
        if n < min_names:
            continue
        k = max(1, int(np.floor(n * frac)))
        order = valid[np.argsort(row[valid], kind="stable")]
        bottom = order[:k]      # low signal
        top = order[-k:]        # high signal
        rr = R[i]
        hi = float(np.nanmean(rr[top]))
        lo = float(np.nanmean(rr[bottom]))
        if long_high:
            out_spread.append(hi - lo)
        else:
            out_spread.append(lo - hi)
        out_hi.append(hi); out_lo.append(lo)
        out_n.append(n); out_t.append(idx[i])
    return pd.DataFrame(
        {"spread": out_spread, "hi": out_hi, "lo": out_lo, "n": out_n}, index=out_t
    ).sort_index()


def sharpe_spreads(ret: pd.DataFrame, lookback: int = 252, skip: int = 21,
                   frac: float = 0.3, min_names: int = 10) -> pd.DataFrame:
    """Long-high-Sharpe / short-low-Sharpe daily fractile spread (the headline book)."""
    return fractile_spreads(ret, trailing_sharpe(ret, lookback, skip), frac, min_names,
                            long_high=True)


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
def spread_stats(spreads: pd.DataFrame, nw_lags: int = 10) -> dict:
    sp = spreads["spread"].to_numpy(dtype=float)
    return {
        "n_days": int(len(spreads)),
        "spread_bps": float(np.nanmean(sp) * 1e4),
        "t_nw": newey_west_t(sp, nw_lags),
        "t_1s": one_sample_t(sp),
        "hi_bps": float(np.nanmean(spreads["hi"].to_numpy()) * 1e4),
        "lo_bps": float(np.nanmean(spreads["lo"].to_numpy()) * 1e4),
        "welch_t": welch_t(spreads["hi"].to_numpy(), spreads["lo"].to_numpy()),
        "gross_sharpe": (
            float(np.nanmean(sp) / np.nanstd(sp, ddof=1) * np.sqrt(TRADING_DAYS))
            if np.nanstd(sp, ddof=1) > 0 else float("nan")
        ),
    }


def signal_rank_corr(ret: pd.DataFrame, lookback: int = 252, skip: int = 21) -> dict:
    """How much do the Sharpe and plain-momentum sorts overlap? Average per-day Spearman
    (rank) correlation between the two signals across the cross-section — high overlap
    means a Sharpe sort is largely re-sorting the same names momentum already picks."""
    sh = trailing_sharpe(ret, lookback, skip)
    mo = trailing_momentum(ret, lookback, skip)
    vo = trailing_vol(ret, lookback, skip)
    corr_sm, corr_sv = [], []
    for t in ret.index:
        a, b, c = sh.loc[t], mo.loc[t], vo.loc[t]
        m = a.notna() & b.notna() & c.notna()
        if m.sum() < 10:
            continue
        ra = a[m].rank(); rb = b[m].rank(); rc = c[m].rank()
        corr_sm.append(ra.corr(rb))
        corr_sv.append(ra.corr(rc))
    return {
        "rho_sharpe_mom": float(np.nanmean(corr_sm)) if corr_sm else float("nan"),
        "rho_sharpe_vol": float(np.nanmean(corr_sv)) if corr_sv else float("nan"),
        "n_days": len(corr_sm),
    }


# --------------------------------------------------------------------------- #
# Placebo — is the spread real, or a lucky alignment of the sort?
# --------------------------------------------------------------------------- #
def placebo_pvalue(
    ret: pd.DataFrame,
    lookback: int = 252,
    skip: int = 21,
    frac: float = 0.3,
    min_names: int = 10,
    n_seeds: int = 20,
    n_draws_per_seed: int = 50,
    base_seed: int = 814,
) -> dict:
    """Keep the trailing-Sharpe sort but read each day's forward return from a
    **column-permuted** panel (signal->outcome link broken, each day's cross-sectional
    distribution preserved). p = share of permuted worlds whose long-high/short-low spread
    mean is >= observed (right-tail test)."""
    cols = list(ret.columns)
    ncol = len(cols)
    sig = trailing_sharpe(ret, lookback, skip).shift(1)
    obs = float(sharpe_spreads(ret, lookback, skip, frac, min_names)["spread"].mean())

    ret_mat = ret.to_numpy(dtype=float)
    pos_of = {c: i for i, c in enumerate(cols)}
    rows_idx, tops, bottoms = [], [], []
    row_lookup = {t: r for r, t in enumerate(ret.index)}
    for t in ret.index:
        s = sig.loc[t].dropna()
        if len(s) < min_names:
            continue
        k = max(1, int(np.floor(len(s) * frac)))
        order = s.sort_values()
        rows_idx.append(row_lookup[t])
        bottoms.append(np.array([pos_of[c] for c in order.index[:k]]))
        tops.append(np.array([pos_of[c] for c in order.index[-k:]]))
    rows_idx = np.asarray(rows_idx)

    means = []
    if len(rows_idx):
        M = ret_mat[rows_idx]
        kt = max(len(a) for a in tops)
        kb = max(len(a) for a in bottoms)

        def _pad(books, kmax):
            P = np.zeros((len(books), kmax), dtype=int)
            V = np.zeros((len(books), kmax), dtype=bool)
            for j, a in enumerate(books):
                P[j, :len(a)] = a
                V[j, :len(a)] = True
            return P, V

        TOP, TOPv = _pad(tops, kt)
        BOT, BOTv = _pad(bottoms, kb)
        rows_ar = np.arange(len(rows_idx))[:, None]

        def _masked_mean(pos, valid, perm):
            vals = M[rows_ar, perm[pos]]
            vals = np.where(valid, vals, np.nan)
            return np.nanmean(vals, axis=1)

        for seed in range(n_seeds):
            rng = np.random.default_rng(base_seed + seed)
            for _ in range(n_draws_per_seed):
                perm = rng.permutation(ncol)
                hi_v = _masked_mean(TOP, TOPv, perm)
                lo_v = _masked_mean(BOT, BOTv, perm)
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
    """Cost the long-high-Sharpe / short-low-Sharpe book.

    The signal is a trailing-12m Sharpe that turns over slowly, but names drift across the
    fractile boundary; to stay comparable to the desk's other cross-sectional timers we
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
def synthetic_detect(panel: dict[str, pd.DataFrame], lookback: int = 252, skip: int = 21,
                     frac: float = 0.3) -> dict:
    """Run the headline Sharpe-spread stats on a synthetic panel."""
    ret = close_returns(panel)
    sp = sharpe_spreads(ret, lookback, skip, frac)
    ts = spread_stats(sp)
    return {"spread_bps": ts["spread_bps"], "t_nw": ts["t_nw"],
            "welch_t": ts["welch_t"], "n_days": ts["n_days"]}
