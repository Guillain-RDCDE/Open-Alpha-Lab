"""Strategy + inference for Study 815 — Variance-Ratio Reversal.

The claim (Lo & MacKinlay 1988): the **variance ratio** ``VR(q) = Var(q-day return) /
(q * Var(1-day return))`` diagnoses departures from a random walk. ``VR < 1`` ⇒ the name
is **mean-reverting** (negative return autocorrelation); ``VR > 1`` ⇒ **trending**. The
cross-sectional test: on each day rank the universe by trailing ``VR(q=5)`` and go
**long the low-VR (mean-reverting) names** / **short the high-VR (trending) names**,
asking whether the mean-reverters offer a tradable **reversal** spread. We measure the
forward spread honestly.

This is distinct from:

* [397-hurst-regime](../../397-hurst-regime/) — the **Hurst exponent** ``H`` from
  rescaled-range / detrended-fluctuation analysis, a *multi-scale* persistence estimate.
  VR(q) is the Lo-MacKinlay *single-horizon* random-walk statistic (``q=5`` here), a
  different (and much more sampling-robust) memory diagnostic.
* [398-entropy-efficiency](../../398-entropy-efficiency/) — a **permutation-entropy**
  market-efficiency score (ordinal-pattern information), not a second-moment ratio.
* [329-one-month-reversal](../../329-one-month-reversal/) — Jegadeesh (1990) sorts on the
  **level of the trailing one-month return** (the classic short-term reversal). VR sorts
  on the **shape of the autocorrelation** (whether a name mean-reverts *at all*), not on
  the sign/size of its recent move.

Method:

* **Close-to-close returns.** Build a per-name daily simple-return panel from adjusted
  Close.
* **Trailing variance ratio.** On each name compute the rolling ``window``-day
  Lo-MacKinlay overlapping-corrected ``VR(q)`` (value on row ``t`` uses returns through
  ``t``), vectorised with rolling sums.
* **Point-in-time sort.** On each day ``t`` rank the cross-section by the trailing VR
  known at the close of ``t-1`` (one ``shift``) and hold day ``t``. Long the bottom
  ``frac`` (low VR, mean-reverting), short the top ``frac`` (high VR, trending); equal
  weight.
* **Inference.** Newey-West (HAC) *t* on the daily long-short spread; a one-sample *t*
  and a pooled Welch *t* (low-VR book vs high-VR book) cross-check; a permutation placebo
  breaks the signal->outcome link; a costed timer charges the round-trip friction.
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


def variance_ratio(r: np.ndarray, q: int = 5) -> float:
    """Lo-MacKinlay (1988) overlapping, bias-corrected variance ratio of one series.

    ``VR(q) = sigma_c2(q) / sigma_a2`` where ``sigma_a2`` is the unbiased 1-day return
    variance and ``sigma_c2(q)`` is the overlapping q-period return variance divided by
    the Lo-MacKinlay bias factor ``m = q*(n-q+1)*(1 - q/n)``. ``VR < 1`` ⇒ mean-reverting,
    ``VR > 1`` ⇒ trending, ``VR = 1`` ⇒ random walk.
    """
    r = np.asarray(r, dtype=float)
    r = r[~np.isnan(r)]
    n = len(r)
    if n < q + 2:
        return float("nan")
    mu = r.mean()
    var1 = float(((r - mu) ** 2).sum()) / (n - 1)
    if var1 <= 0:
        return float("nan")
    cs = np.concatenate([[0.0], np.cumsum(r)])
    rq = cs[q:] - cs[:-q]                       # overlapping q-day returns, length n-q+1
    nq = n - q + 1
    m = q * nq * (1.0 - q / n)
    if m <= 0:
        return float("nan")
    sigma_c2 = float(((rq - q * mu) ** 2).sum()) / m
    return sigma_c2 / var1


def trailing_vr(ret: pd.DataFrame, window: int = 120, q: int = 5) -> pd.DataFrame:
    """Rolling ``window``-day Lo-MacKinlay ``VR(q)`` per name, fully vectorised.

    For each name and each row ``t`` the statistic uses the ``window`` daily returns
    through ``t`` (inclusive). Vectorised via rolling sums:

    * ``sigma_a2`` — unbiased 1-day variance over the window (``* W/(W-1)``);
    * ``Rq`` — the ``q``-day overlapping return, a length-``q`` rolling sum of daily
      returns; the ``Nq = W-q+1`` such q-returns ending at ``t`` all lie inside the
      daily window ``[t-W+1, t]``;
    * ``sigma_c2 = SS / m`` with ``SS = sum (Rq - q*mu)^2`` expanded as
      ``S2 - 2 q mu S1 + Nq (q mu)^2`` (rolling sums ``S1`` of ``Rq`` and ``S2`` of
      ``Rq**2`` over ``Nq``), and ``m = q*Nq*(1 - q/W)``.

    Value on row ``t`` uses returns through ``t``; the sort in :func:`vr_spreads` shifts
    by one day so a day-``t`` position is formed on information known at ``t-1``.
    """
    W, r = window, ret
    Nq = W - q + 1
    mu = r.rolling(W, min_periods=W).mean()
    e2 = (r ** 2).rolling(W, min_periods=W).mean()
    var1 = (e2 - mu ** 2) * (W / (W - 1.0))              # sigma_a2 (unbiased)

    Rq = r.rolling(q, min_periods=q).sum()               # overlapping q-day returns
    S1 = Rq.rolling(Nq, min_periods=Nq).sum()
    S2 = (Rq ** 2).rolling(Nq, min_periods=Nq).sum()
    SS = S2 - 2.0 * q * mu * S1 + Nq * (q * mu) ** 2
    m = q * Nq * (1.0 - q / W)
    sigma_c2 = SS / m
    out = sigma_c2 / var1
    return out.where(var1 > 0)


# --------------------------------------------------------------------------- #
# The cross-sectional sort -> long-low-VR / short-high-VR spread
# --------------------------------------------------------------------------- #
def vr_spreads(
    ret: pd.DataFrame,
    window: int = 120,
    q: int = 5,
    frac: float = 0.3,
    min_names: int = 10,
) -> pd.DataFrame:
    """Daily equal-weight bottom-minus-top variance-ratio fractile spread.

    On each day ``t`` names are ranked by the trailing ``VR(q)`` known at the close of
    ``t-1`` (one ``shift``). ``lo`` = mean forward day-``t`` return of the bottom
    ``frac`` (low VR, mean-reverting, the long); ``hi`` = mean of the top ``frac``
    (high VR, trending, the short). ``spread = lo - hi`` (long low-VR, short high-VR).
    Days with fewer than ``min_names`` ranked names are dropped.
    """
    sig = trailing_vr(ret, window, q).shift(1)   # known at close t-1
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
        low = order[:k]        # low VR (mean-reverting) -> long
        high = order[-k:]      # high VR (trending)      -> short
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
def vr_stats(spreads: pd.DataFrame, nw_lags: int = 10) -> dict:
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
    window: int = 120,
    q: int = 5,
    frac: float = 0.3,
    min_names: int = 10,
    n_seeds: int = 20,
    n_draws_per_seed: int = 50,
    base_seed: int = 815,
) -> dict:
    """Keep the trailing-VR sort but read each day's forward return from a
    **column-permuted** panel (signal->outcome link broken, each day's
    cross-sectional distribution preserved). p = share of permuted worlds whose
    spread mean is >= observed (right-tail test on the long-low/short-high spread)."""
    cols = list(ret.columns)
    ncol = len(cols)
    sig = trailing_vr(ret, window, q).shift(1)
    obs = float(vr_spreads(ret, window, q, frac, min_names)["spread"].mean())

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
    """Cost the long-low-VR / short-high-VR book.

    The signal is a trailing-window variance ratio that turns over slowly, but names
    drift across the fractile boundary daily; we charge a conservative daily round-trip
    on the 2x-NAV long-short book. To stay comparable to the desk's other cross-sectional
    timers we charge 2 sides x one-way cost x NAV per day, plus borrow on the short leg.
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
def synthetic_detect(panel: dict[str, pd.DataFrame], window: int = 120,
                     q: int = 5, frac: float = 0.3) -> dict:
    """Run the headline VR stats on a synthetic panel."""
    ret = close_returns(panel)
    sp = vr_spreads(ret, window, q, frac)
    ts = vr_stats(sp)
    return {"spread_bps": ts["spread_bps"], "t_nw": ts["t_nw"],
            "welch_t": ts["welch_t"], "n_days": ts["n_days"]}
