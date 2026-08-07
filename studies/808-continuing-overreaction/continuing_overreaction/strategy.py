"""Strategy + inference for Study 808 — Continuing Overreaction.

The claim (Byun, Lim & Yun 2016): build a **weighted signed-momentum** score from a
name's trailing monthly returns — weight the *signs* of the more recent months more
heavily — and it predicts the cross-section of returns **positively**. A name on a
persistent recent up-streak (high "continuing overreaction", CO) keeps rising: a
long-high-CO / short-low-CO book should earn a positive spread (continuation, later
followed by reversal at longer horizons).

The measure, on each name at the end of month ``i``:

    CO_i = sum_{p} w_p * sign(r_{i, m})

over the ``n`` monthly returns of months ``i-(n+skip) .. i-(skip+1)`` (the trailing
``n`` months, **skipping the most recent** ``skip`` month(s) to sidestep the 1-month
reversal), with weights ``w_p`` that **increase toward the recent months** and are
normalised to sum to 1 (the Byun-Lim-Yun ``w_j = (n-j)`` shape — recent months count
most). Only the *sign* of each monthly return enters, so CO measures the *consistency*
of the recent run, not its magnitude.

This is distinct from:

* [507-cross-sectional-momentum](../../507-cross-sectional-momentum/) — plain
  (12,1) past-**return** momentum, which sorts on the *magnitude* of the cumulative
  trailing return, not the weighted **signs** of the monthly steps;
* [508-momentum-crashes](../../508-momentum-crashes/) — the *conditional* crash risk
  of the momentum factor, not a signed-consistency signal;
* [196-long-term-reversal](../../196-long-term-reversal/) — the 3-5y **reversal**
  (De Bondt-Thaler), the opposite horizon and sign;
* [510-frog-in-the-pan](../../510-frog-in-the-pan/) — information *discreteness* (how
  smoothly the past return arrived), a path-smoothness modifier of momentum, not a
  weighted count of monthly signs.

Method:

* **Monthly returns.** Resample adjusted Close to month-end simple returns.
* **CO signal.** The normalised recent-weighted sum of monthly-return signs over the
  trailing ``n`` months, skipping the most recent ``skip``.
* **Point-in-time sort.** On each month ``i`` rank the cross-section by the CO known
  through month ``i-1-skip``; long the top ``frac`` (high CO), short the bottom
  ``frac`` (low CO); equal weight; hold month ``i`` (the skipped month ``i-1`` is the
  documented lag). Forward return is month ``i``'s realised return.
* **Inference.** Newey-West (HAC) *t* on the monthly long-short spread; a one-sample
  *t* and a pooled Welch *t* (high-CO book vs low-CO book) cross-check; a permutation
  placebo breaks the signal->outcome link; a costed timer charges the monthly friction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MONTHS_PER_YEAR = 12


# --------------------------------------------------------------------------- #
# Return panel + signal
# --------------------------------------------------------------------------- #
def monthly_returns(panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Month-end simple returns (index=month-end, columns=ticker)."""
    closes = pd.DataFrame({s: panel[s]["Close"] for s in panel}).sort_index()
    monthly_close = closes.resample("ME").last()
    return monthly_close.pct_change()


def co_signal(
    monthly: pd.DataFrame,
    n: int = 12,
    skip: int = 1,
) -> pd.DataFrame:
    """Weighted signed-momentum ("continuing overreaction") score, per name.

    For each month row ``i`` the score reads the **signs** of the ``n`` monthly returns
    of months ``i-(n+skip) .. i-(skip+1)`` — the trailing ``n`` months **skipping the
    most recent** ``skip`` — weighted by ``w_p`` that increase toward the recent months
    (``w_p ∝ p+1``, oldest→newest), normalised to sum 1. Value on row ``i`` therefore
    uses only information known at the end of month ``i-1-skip``; the sort in
    :func:`co_spreads` holds month ``i`` itself, so the skipped month ``i-1`` is a
    documented execution lag with zero look-ahead.

    Vectorised as a fixed length-``n`` linear filter over the sign matrix (an ``n``-step
    loop over the weights, never a per-date loop).
    """
    sgn = np.sign(monthly.to_numpy(dtype=float))
    T, N = sgn.shape
    offset = n + skip
    w = np.arange(1, n + 1, dtype=float)   # oldest=1 ... newest=n (increase toward recent)
    w /= w.sum()
    co = np.full((T, N), np.nan, dtype=float)
    if T > offset:
        acc = np.zeros((T - offset, N), dtype=float)
        for p in range(n):
            acc = acc + w[p] * sgn[p:T - offset + p]
        co[offset:] = acc
    return pd.DataFrame(co, index=monthly.index, columns=monthly.columns)


# --------------------------------------------------------------------------- #
# The cross-sectional sort -> long-high-CO / short-low-CO spread
# --------------------------------------------------------------------------- #
def co_spreads(
    monthly: pd.DataFrame,
    n: int = 12,
    skip: int = 1,
    frac: float = 0.3,
    min_names: int = 10,
) -> pd.DataFrame:
    """Monthly equal-weight top-minus-bottom CO fractile spread.

    On each month ``i`` names are ranked by the CO known through month ``i-1-skip``.
    ``high`` = mean forward month-``i`` return of the top ``frac`` (high CO, the long);
    ``low`` = mean of the bottom ``frac`` (low CO, the short). ``spread = high - low``
    (long high-CO, short low-CO — the Byun-Lim-Yun continuation). Months with fewer than
    ``min_names`` ranked names are dropped.
    """
    sig = co_signal(monthly, n, skip)
    S = sig.to_numpy(dtype=float)
    R = monthly.to_numpy(dtype=float)
    idx = monthly.index
    out_spread, out_hi, out_lo, out_n, out_t = [], [], [], [], []
    for i in range(len(idx)):
        row = S[i]
        rr = R[i]
        valid = np.where(~np.isnan(row) & ~np.isnan(rr))[0]
        m = len(valid)
        if m < min_names:
            continue
        k = max(1, int(np.floor(m * frac)))
        order = valid[np.argsort(row[valid], kind="stable")]
        low = order[:k]         # low CO  -> short
        high = order[-k:]       # high CO -> long
        hi = float(np.nanmean(rr[high]))
        lo = float(np.nanmean(rr[low]))
        out_spread.append(hi - lo); out_hi.append(hi); out_lo.append(lo)
        out_n.append(m); out_t.append(idx[i])
    return pd.DataFrame(
        {"spread": out_spread, "high": out_hi, "low": out_lo, "n": out_n}, index=out_t
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
def co_stats(spreads: pd.DataFrame, nw_lags: int = 6) -> dict:
    sp = spreads["spread"].to_numpy(dtype=float)
    return {
        "n_months": int(len(spreads)),
        "spread_bps": float(np.nanmean(sp) * 1e4),
        "t_nw": newey_west_t(sp, nw_lags),
        "t_1s": one_sample_t(sp),
        "hi_bps": float(np.nanmean(spreads["high"].to_numpy()) * 1e4),
        "lo_bps": float(np.nanmean(spreads["low"].to_numpy()) * 1e4),
        "welch_t": welch_t(spreads["high"].to_numpy(), spreads["low"].to_numpy()),
    }


# --------------------------------------------------------------------------- #
# Placebo — is the spread real, or a lucky alignment of the sort?
# --------------------------------------------------------------------------- #
def placebo_pvalue(
    monthly: pd.DataFrame,
    n: int = 12,
    skip: int = 1,
    frac: float = 0.3,
    min_names: int = 10,
    n_seeds: int = 20,
    n_draws_per_seed: int = 50,
    base_seed: int = 808,
) -> dict:
    """Keep the CO sort but read each month's forward return from a **column-permuted**
    panel (signal->outcome link broken, each month's cross-sectional distribution
    preserved). p = share of permuted worlds whose spread mean is >= observed
    (right-tail test on the long-high/short-low spread)."""
    cols = list(monthly.columns)
    ncol = len(cols)
    sig = co_signal(monthly, n, skip)
    obs = float(co_spreads(monthly, n, skip, frac, min_names)["spread"].mean())

    ret_mat = monthly.to_numpy(dtype=float)
    pos_of = {c: i for i, c in enumerate(cols)}
    rows_idx, lows, highs = [], [], []
    row_lookup = {t: r for r, t in enumerate(monthly.index)}
    for t in monthly.index:
        s = sig.loc[t]
        rr = monthly.loc[t]
        s = s[~s.isna() & ~rr.isna()]
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
                hi_v = _masked_mean(HIGH, HIGHv, perm)
                lo_v = _masked_mean(LOW, LOWv, perm)
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
    """Cost the long-high-CO / short-low-CO book.

    The CO sort rebalances monthly, so we charge a conservative per-rebalance round-trip
    on the 2x-NAV long-short book — 2 sides x one-way cost x NAV per month — plus borrow
    on the short leg (prorated monthly). This is the honest test of whether a monthly
    cross-sectional spread survives friction.
    """
    sp = spreads["spread"].to_numpy(dtype=float)
    sp = sp[~np.isnan(sp)]
    n = len(sp)
    round_trip_cost = 2.0 * cost_bps / 1e4
    borrow_monthly = (borrow_bps_yr / 1e4) / MONTHS_PER_YEAR
    net = sp - round_trip_cost - borrow_monthly
    gross_mean = float(sp.mean())
    net_mean = float(net.mean())
    sd = float(net.std(ddof=1)) if n > 1 else float("nan")
    sharpe = net_mean / sd * np.sqrt(MONTHS_PER_YEAR) if sd and sd > 0 else float("nan")
    return {
        "n_months": n,
        "gross_bps": gross_mean * 1e4,
        "net_bps": net_mean * 1e4,
        "cost_bps_per_reb": (round_trip_cost + borrow_monthly) * 1e4,
        "ann_net_pct": net_mean * MONTHS_PER_YEAR * 100,
        "sharpe_net": sharpe,
        "t_net": one_sample_t(net),
    }


# --------------------------------------------------------------------------- #
# Synthetic-control detector (the machinery proof)
# --------------------------------------------------------------------------- #
def synthetic_detect(panel: dict[str, pd.DataFrame], n: int = 12, skip: int = 1,
                     frac: float = 0.3) -> dict:
    """Run the headline CO stats on a synthetic panel."""
    monthly = monthly_returns(panel)
    sp = co_spreads(monthly, n, skip, frac)
    ts = co_stats(sp)
    return {"spread_bps": ts["spread_bps"], "t_nw": ts["t_nw"],
            "welch_t": ts["welch_t"], "n_months": ts["n_months"]}
