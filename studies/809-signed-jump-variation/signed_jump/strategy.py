"""Strategy + inference for Study 809 — Signed Jump Variation.

The claim (Barndorff-Nielsen, Kinnebrock & Shephard 2010 realized semivariance;
Bollerslev, Li & Zhao 2020): split a name's trailing realized variance into an **upside**
piece ``RS+ = Σ r²·1(r>0)`` and a **downside** piece ``RS- = Σ r²·1(r<0)``. The scaled
**signed jump variation** ``SJ = (RS+ − RS-) / RV`` (``RV = RS+ + RS-``) lives in
``[-1, +1]``: positive when the recent variance is upside-dominated ("good" volatility),
negative when downside-dominated ("bad" volatility). The paper prices it **negatively**:
the downside-dominated (negative-SJ) names carry a **premium** and out-earn, so a long
**low-SJ** / short **high-SJ** book should earn a positive spread.

This is distinct from:

* [803-realized-skewness-reversal](../../803-realized-skewness-reversal/) — the trailing
  **realized skewness** (the standardised third moment ``m3/m2^1.5``), a single asymmetry
  number. Signed jump variation is a **variance decomposition** by the *sign of the return*
  (``RS+`` vs ``RS-``), a different estimator with a different paper.
* [505-left-tail-momentum](../../505-left-tail-momentum/) — a **left-tail** (downside VaR /
  worst-return) *momentum* signal, not the up-vs-down **variance ratio** of the whole month.
* [130-variance-risk-premium](../../130-variance-risk-premium/) — the market-level
  **implied-minus-realized** variance gap (a time-series option signal), not a
  cross-sectional sort on a name's own **realized** up/down variance split.

Method:

* **Close-to-close returns.** Build a per-name daily simple-return panel from adjusted
  Close.
* **Trailing signed jump variation.** On each name compute, over a rolling ``window`` of
  daily returns, ``RS+`` / ``RS-`` and ``SJ = (RS+ − RS-)/RV`` (value on row ``t`` uses
  returns through ``t``).
* **Point-in-time sort.** On each day ``t`` rank the cross-section by the trailing ``SJ``
  known at the close of ``t-1`` (one ``shift``) and hold day ``t``. Long the bottom
  ``frac`` (low / negative SJ — downside-dominated), short the top ``frac`` (high SJ —
  upside-dominated); equal weight.
* **Inference.** Newey-West (HAC) *t* on the daily long-short spread; a one-sample *t* and a
  pooled Welch *t* (bottom book vs top book) cross-check; a permutation placebo breaks the
  signal->outcome link; a costed timer charges the round-trip friction.
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


def _signed_jump(x: np.ndarray) -> float:
    """Scaled signed jump variation ``(RS+ - RS-)/RV`` of a return sample, in [-1, 1]."""
    x = x[~np.isnan(x)]
    if len(x) < 3:
        return float("nan")
    sq = x * x
    rv = float(sq.sum())
    if rv <= 0:
        return float("nan")
    rs_plus = float(sq[x > 0].sum())
    rs_minus = float(sq[x < 0].sum())
    return (rs_plus - rs_minus) / rv


def trailing_signed_jump(ret: pd.DataFrame, window: int = 21) -> pd.DataFrame:
    """Rolling ``window``-day scaled signed jump variation ``(RS+ − RS-)/RV``, per name.

    Vectorised: ``RS+`` is the rolling sum of ``r²`` over up-days, ``RS-`` the same over
    down-days, ``RV = RS+ + RS-`` the rolling realized variance; ``SJ = (RS+ − RS-)/RV``.
    Value on row ``t`` uses returns through ``t`` (inclusive); the sort in
    :func:`signed_jump_spreads` shifts by one day so a day-``t`` position is formed on
    information known at ``t-1``.
    """
    r = ret
    sq = r * r
    up = sq.where(r > 0, 0.0)
    dn = sq.where(r < 0, 0.0)
    rs_plus = up.rolling(window, min_periods=window).sum()
    rs_minus = dn.rolling(window, min_periods=window).sum()
    rv = rs_plus + rs_minus
    out = (rs_plus - rs_minus) / rv
    return out.where(rv > 0)


# --------------------------------------------------------------------------- #
# The cross-sectional sort -> long-low-SJ / short-high-SJ spread
# --------------------------------------------------------------------------- #
def signed_jump_spreads(
    ret: pd.DataFrame,
    window: int = 21,
    frac: float = 0.3,
    min_names: int = 10,
) -> pd.DataFrame:
    """Daily equal-weight bottom-minus-top signed-jump fractile spread.

    On each day ``t`` names are ranked by the trailing signed jump variation known at the
    close of ``t-1`` (one ``shift``). ``lo`` = mean forward day-``t`` return of the bottom
    ``frac`` (low / negative SJ — downside-dominated, the long); ``hi`` = mean of the top
    ``frac`` (high SJ — upside-dominated, the short). ``spread = lo - hi`` (long low-SJ,
    short high-SJ). Days with fewer than ``min_names`` ranked names are dropped.
    """
    sig = trailing_signed_jump(ret, window).shift(1)  # known at close t-1
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
        low = order[:k]        # low / negative SJ (downside) -> long
        high = order[-k:]      # high SJ (upside)             -> short
        rr = R[i]
        lo = float(np.nanmean(rr[low]))
        hi = float(np.nanmean(rr[high]))
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
def signed_jump_stats(spreads: pd.DataFrame, nw_lags: int = 10) -> dict:
    sp = spreads["spread"].to_numpy(dtype=float)
    return {
        "n_days": int(len(spreads)),
        "spread_bps": float(np.nanmean(sp) * 1e4),
        "t_nw": newey_west_t(sp, nw_lags),
        "t_1s": one_sample_t(sp),
        "lo_bps": float(np.nanmean(spreads["lo"].to_numpy()) * 1e4),
        "hi_bps": float(np.nanmean(spreads["hi"].to_numpy()) * 1e4),
        "welch_t": welch_t(spreads["lo"].to_numpy(), spreads["hi"].to_numpy()),
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
    base_seed: int = 809,
) -> dict:
    """Keep the trailing signed-jump sort but read each day's forward return from a
    **column-permuted** panel (signal->outcome link broken, each day's cross-sectional
    distribution preserved). p = share of permuted worlds whose spread mean is >= observed
    (right-tail test on the long-low/short-high spread)."""
    cols = list(ret.columns)
    ncol = len(cols)
    sig = trailing_signed_jump(ret, window).shift(1)
    obs = float(signed_jump_spreads(ret, window, frac, min_names)["spread"].mean())

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
                means.append(np.nanmean(lo_v - hi_v))
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
    """Cost the long-low-SJ / short-high-SJ book.

    The signal is a trailing-window variance split that turns over roughly monthly, but
    names drift across the fractile boundary daily; we charge a conservative daily
    round-trip on the 2x-NAV book. To stay comparable to the desk's other cross-sectional
    timers we charge 2 sides x one-way cost x NAV per day on the long-short book, plus
    borrow on the short leg.
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
    """Run the headline signed-jump stats on a synthetic panel."""
    ret = close_returns(panel)
    sp = signed_jump_spreads(ret, window, frac)
    ts = signed_jump_stats(sp)
    return {"spread_bps": ts["spread_bps"], "t_nw": ts["t_nw"],
            "welch_t": ts["welch_t"], "n_days": ts["n_days"]}
