"""Strategy + inference for Study 810 — Price Delay.

The claim (Hou & Moskowitz 2005): a stock into which market information diffuses
**slowly** — one whose return responds to the market with a lag — commands a **return
premium** over a stock that prices the same information promptly. The measure:

* **Weekly returns.** Resample each name's adjusted Close to weekly (W-FRI) and take
  simple weekly returns. The **market** is the equal-weight cross-sectional mean of the
  panel's weekly returns.
* **Delay regression (trailing 1 year).** On each week ``t``, over a trailing
  ``window``-week window, regress a name's weekly return on the **contemporaneous** market
  return alone (restricted, ``R2_r``) and on the contemporaneous market plus **4 weekly
  lags** of the market (unrestricted, ``R2_u``). The **delay** measure is

      delay = 1 - R2_r / R2_u

  the fraction of explained variance that the *lagged* market terms contribute. Delay ≈ 0
  means the name prices the market instantly; delay → 1 means most of the co-movement
  shows up only in the lags — a slow name.
* **Point-in-time sort.** On each week ``t`` rank the cross-section by the delay known at
  the close of ``t-1`` (one ``shift``) and hold week ``t``. **Long the top ``frac``**
  (HIGH delay), **short the bottom ``frac``** (LOW delay); equal weight.
* **Inference.** Newey-West (HAC) *t* on the weekly long-short spread; a one-sample *t*
  and a pooled Welch *t* (high-delay book vs low-delay book) cross-check; a permutation
  placebo breaks the signal->outcome link; a costed timer charges the round-trip friction.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

TRADING_WEEKS = 52


# --------------------------------------------------------------------------- #
# Return panel + market + signal
# --------------------------------------------------------------------------- #
def weekly_returns(panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Weekly (W-FRI) simple returns per name (index=week, columns=ticker)."""
    closes = pd.DataFrame({s: panel[s]["Close"] for s in panel}).sort_index()
    wk = closes.resample("W-FRI").last()
    return wk.pct_change()


def market_weekly(weekly: pd.DataFrame) -> pd.Series:
    """Equal-weight cross-sectional mean of the panel's weekly returns (the market)."""
    return weekly.mean(axis=1, skipna=True)


def _r2_multi(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Ordinary-least-squares R^2 of each column of ``Y`` on the common design ``X``.

    ``X`` is (L, k) full-rank (an intercept column included); ``Y`` is (L, N) with no NaN.
    Solved once for all N columns via a single least-squares call (the design is shared,
    so this is a single small factorisation, not one per name)."""
    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    resid = Y - X @ beta
    ss_res = np.sum(resid ** 2, axis=0)
    ybar = Y.mean(axis=0)
    ss_tot = np.sum((Y - ybar) ** 2, axis=0)
    return 1.0 - ss_res / np.where(ss_tot > 0, ss_tot, np.nan)


def price_delay(weekly: pd.DataFrame, window: int = 52, n_lags: int = 4) -> pd.DataFrame:
    """Rolling Hou-Moskowitz **delay** measure per name (index=week, columns=ticker).

    For each week ``t`` and each name with a complete trailing ``window``-week history,
    ``delay = 1 - R2(contemporaneous market only) / R2(contemporaneous + n_lags weekly
    lags)``. The market is the equal-weight cross-sectional mean of the panel's weekly
    returns. Value on row ``t`` uses the window **ending at** ``t`` (inclusive); the sort
    in :func:`delay_spreads` shifts by one week so a week-``t`` position is formed on
    information known at ``t-1``. Vectorised across names inside each window (the market
    design is common to all names, so one factorisation serves the whole cross-section).
    """
    W = weekly.to_numpy(dtype=float)
    T, N = W.shape
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mkt = np.nanmean(W, axis=1)  # all-NaN weeks (e.g. the first) -> NaN, harmless
    delay = np.full((T, N), np.nan)
    L = window
    ones = np.ones(L)
    for t in range(L - 1 + n_lags, T):
        s0 = t - L + 1
        contemp = mkt[s0:t + 1]
        cols = [ones, contemp]
        for k in range(1, n_lags + 1):
            cols.append(mkt[s0 - k:t + 1 - k])
        Xu = np.column_stack(cols)
        if not np.all(np.isfinite(Xu)):
            continue
        Xr = np.column_stack([ones, contemp])
        Yw = W[s0:t + 1, :]
        good = np.all(np.isfinite(Yw), axis=0)
        if not good.any():
            continue
        Yg = Yw[:, good]
        r2u = _r2_multi(Xu, Yg)
        r2r = _r2_multi(Xr, Yg)
        d = 1.0 - r2r / np.where(r2u > 1e-6, r2u, np.nan)
        delay[t, good] = d
    return pd.DataFrame(delay, index=weekly.index, columns=weekly.columns)


# --------------------------------------------------------------------------- #
# The cross-sectional sort -> long-high-delay / short-low-delay spread
# --------------------------------------------------------------------------- #
def delay_spreads(
    weekly: pd.DataFrame,
    window: int = 52,
    n_lags: int = 4,
    frac: float = 0.3,
    min_names: int = 10,
) -> pd.DataFrame:
    """Weekly equal-weight top-minus-bottom price-delay fractile spread.

    On each week ``t`` names are ranked by the delay known at the close of ``t-1`` (one
    ``shift``). ``long`` = mean forward week-``t`` return of the top ``frac`` (HIGH delay);
    ``short`` = mean of the bottom ``frac`` (LOW delay). ``spread = long - short`` (long
    high-delay, short low-delay). Weeks with fewer than ``min_names`` ranked names dropped.
    """
    sig = price_delay(weekly, window, n_lags).shift(1)  # known at close t-1
    S = sig.to_numpy(dtype=float)
    R = weekly.to_numpy(dtype=float)
    idx = weekly.index
    out_spread, out_long, out_short, out_n, out_t = [], [], [], [], []
    for i in range(len(idx)):
        row = S[i]
        valid = np.where(~np.isnan(row))[0]
        n = len(valid)
        if n < min_names:
            continue
        k = max(1, int(np.floor(n * frac)))
        order = valid[np.argsort(row[valid], kind="stable")]
        low = order[:k]         # low delay  -> short
        high = order[-k:]       # high delay -> long
        rr = R[i]
        lg = float(np.nanmean(rr[high]))
        sh = float(np.nanmean(rr[low]))
        out_spread.append(lg - sh); out_long.append(lg); out_short.append(sh)
        out_n.append(n); out_t.append(idx[i])
    return pd.DataFrame(
        {"spread": out_spread, "long": out_long, "short": out_short, "n": out_n},
        index=out_t,
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
def delay_stats(spreads: pd.DataFrame, nw_lags: int = 6) -> dict:
    sp = spreads["spread"].to_numpy(dtype=float)
    return {
        "n_weeks": int(len(spreads)),
        "spread_bps": float(np.nanmean(sp) * 1e4),
        "t_nw": newey_west_t(sp, nw_lags),
        "t_1s": one_sample_t(sp),
        "long_bps": float(np.nanmean(spreads["long"].to_numpy()) * 1e4),
        "short_bps": float(np.nanmean(spreads["short"].to_numpy()) * 1e4),
        "welch_t": welch_t(spreads["long"].to_numpy(), spreads["short"].to_numpy()),
    }


# --------------------------------------------------------------------------- #
# Placebo — is the spread real, or a lucky alignment of the sort?
# --------------------------------------------------------------------------- #
def placebo_pvalue(
    weekly: pd.DataFrame,
    window: int = 52,
    n_lags: int = 4,
    frac: float = 0.3,
    min_names: int = 10,
    n_seeds: int = 20,
    n_draws_per_seed: int = 50,
    base_seed: int = 810,
) -> dict:
    """Keep the delay sort but read each week's forward return from a **column-permuted**
    panel (signal->outcome link broken, each week's cross-sectional distribution
    preserved). p = share of permuted worlds whose spread mean is >= observed (right-tail
    test on the long-high-delay / short-low-delay spread)."""
    cols = list(weekly.columns)
    ncol = len(cols)
    sig = price_delay(weekly, window, n_lags).shift(1)
    obs = float(delay_spreads(weekly, window, n_lags, frac, min_names)["spread"].mean())

    ret_mat = weekly.to_numpy(dtype=float)
    pos_of = {c: i for i, c in enumerate(cols)}
    rows_idx, lows, highs = [], [], []
    row_lookup = {t: r for r, t in enumerate(weekly.index)}
    for t in weekly.index:
        s = sig.loc[t].dropna()
        if len(s) < min_names:
            continue
        k = max(1, int(np.floor(len(s) * frac)))
        order = s.sort_values()
        rows_idx.append(row_lookup[t])
        lows.append(np.array([pos_of[c] for c in order.index[:k]]))     # low delay  -> short
        highs.append(np.array([pos_of[c] for c in order.index[-k:]]))   # high delay -> long
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
    """Cost the long-high-delay / short-low-delay book.

    The delay signal turns over on a weekly rebalance. We charge 2 sides x one-way cost x
    NAV per week on the long-short book, plus borrow on the short leg — the same
    convention as the desk's other cross-sectional timers.
    """
    sp = spreads["spread"].to_numpy(dtype=float)
    sp = sp[~np.isnan(sp)]
    n = len(sp)
    round_trip_cost = 2.0 * cost_bps / 1e4
    borrow_weekly = (borrow_bps_yr / 1e4) / TRADING_WEEKS
    net = sp - round_trip_cost - borrow_weekly
    gross_mean = float(sp.mean())
    net_mean = float(net.mean())
    sd = float(net.std(ddof=1)) if n > 1 else float("nan")
    sharpe = net_mean / sd * np.sqrt(TRADING_WEEKS) if sd and sd > 0 else float("nan")
    return {
        "n_weeks": n,
        "gross_bps": gross_mean * 1e4,
        "net_bps": net_mean * 1e4,
        "cost_bps_per_week": (round_trip_cost + borrow_weekly) * 1e4,
        "ann_net_pct": net_mean * TRADING_WEEKS * 100,
        "sharpe_net": sharpe,
        "t_net": one_sample_t(net),
    }


# --------------------------------------------------------------------------- #
# Synthetic-control detector (the machinery proof)
# --------------------------------------------------------------------------- #
def synthetic_detect(panel: dict[str, pd.DataFrame], window: int = 52,
                     n_lags: int = 4, frac: float = 0.3) -> dict:
    """Run the headline delay stats on a synthetic panel."""
    weekly = weekly_returns(panel)
    sp = delay_spreads(weekly, window, n_lags, frac)
    ts = delay_stats(sp)
    return {"spread_bps": ts["spread_bps"], "t_nw": ts["t_nw"],
            "welch_t": ts["welch_t"], "n_weeks": ts["n_weeks"]}
