"""Strategy + inference for Study 805 — Cokurtosis Premium.

The claim (Fang & Lai 1997; four-moment CAPM): a security's **systematic kurtosis** — its
**cokurtosis** with the market — is a priced risk. Sort a cross-section on trailing
cokurtosis; the high-cokurtosis names (those that amplify the market's fat-tailed moves)
should earn a **positive** risk premium, so a book that is **long high-cokurtosis / short
low-cokurtosis** should earn a positive spread.

This is distinct from:

* [504-coskewness](../../504-coskewness/) — **systematic co-skewness**, the *third*-order
  co-moment ``E[(r_i-μ_i)(r_m-μ_m)^2]`` (one power on the name, **two** on the market). This
  study uses the *fourth*-order co-moment with the market cubed, ``(r_m-μ_m)^3`` — a
  different tail exposure and a different paper.
* [238-betting-against-beta](../../238-betting-against-beta/) — the low-beta / BAB trade, a
  sort on the *second*-order co-moment (co-**variance**, i.e. beta). Cokurtosis is a
  higher-moment exposure orthogonal in spirit to plain beta.
* [803-realized-skewness-reversal](../../803-realized-skewness-reversal/) — a name's **own**
  total realized skewness (a stand-alone third moment), not its **co**-movement with the
  market at all.

Method:

* **Close-to-close returns.** Build a per-name daily simple-return panel from adjusted Close.
* **Equal-weight market.** ``r_m`` = the cross-sectional equal-weight mean of the panel's
  daily returns.
* **Trailing cokurtosis.** On each name compute the rolling ``window``-day standardised
  fourth co-moment ``E[(r_i-μ_i)(r_m-μ_m)^3]/(σ_i·σ_m^3)`` (value on row ``t`` uses returns
  through ``t``), vectorised via rolling raw moments — no per-date python loop for the signal.
* **Point-in-time sort.** On each day ``t`` rank the cross-section by the trailing cokurtosis
  known at the close of ``t-1`` (one ``shift``) and hold day ``t``. Long the top ``frac``
  (high cokurtosis), short the bottom ``frac`` (low cokurtosis); equal weight. ``spread =
  hi - lo`` (a positive claim ⇒ positive spread).
* **Inference.** Newey-West (HAC) *t* on the daily long-short spread; a one-sample *t* and a
  pooled Welch *t* (high book vs low book) cross-check; a permutation placebo breaks the
  signal→outcome link; a costed timer charges the round-trip friction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# --------------------------------------------------------------------------- #
# Return panel + market + signal
# --------------------------------------------------------------------------- #
def close_returns(panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Daily simple close-to-close returns (index=date, columns=ticker)."""
    closes = pd.DataFrame({s: panel[s]["Close"] for s in panel}).sort_index()
    return closes.pct_change()


def market_return(ret: pd.DataFrame) -> pd.Series:
    """Equal-weight market return = cross-sectional mean across names (per day)."""
    return ret.mean(axis=1)


def trailing_cokurt(ret: pd.DataFrame, window: int = 252) -> pd.DataFrame:
    """Rolling ``window``-day standardised cokurtosis with the equal-weight market, per name.

    ``cokurt_i = E[(r_i-μ_i)(r_m-μ_m)^3] / (σ_i·σ_m^3)`` over the trailing window, where
    ``r_m`` is the equal-weight panel mean. Vectorised via rolling raw moments (the market
    is a single broadcast series), so there is **no** per-date python loop here:

        E[(r_i-μ_i)(r_m-μ_m)^3] = E[r_i(r_m-μ_m)^3] - μ_i·m3_m
        E[r_i(r_m-μ_m)^3] = A - 3μ_m·B + 3μ_m²·Cc - μ_m³·μ_i
        m3_m = E[r_m³] - 3μ_m·E[r_m²] + 2μ_m³

    with ``A = E[r_i·r_m³]``, ``B = E[r_i·r_m²]``, ``Cc = E[r_i·r_m]`` (rolling means,
    market broadcast across name columns). ``σ_i``/``σ_m`` are population (ddof=0) window
    std. Value on row ``t`` uses returns through ``t`` (inclusive); :func:`cokurt_spreads`
    shifts by one day so a day-``t`` position is formed on information known at ``t-1``.
    """
    r = ret
    rm = market_return(r)
    W = window
    kw = dict(window=W, min_periods=W)

    mu_i = r.rolling(**kw).mean()
    e_ii = (r ** 2).rolling(**kw).mean()
    sig_i = (e_ii - mu_i ** 2).clip(lower=0.0).pow(0.5)          # ddof=0 std of name

    mu_m = rm.rolling(**kw).mean()
    e_m2 = (rm ** 2).rolling(**kw).mean()
    e_m3 = (rm ** 3).rolling(**kw).mean()
    var_m = (e_m2 - mu_m ** 2).clip(lower=0.0)
    sig_m = var_m.pow(0.5)
    m3_m = e_m3 - 3 * mu_m * e_m2 + 2 * mu_m ** 3                # 3rd central moment (market)

    # rolling cross raw moments A=E[r_i r_m^3], B=E[r_i r_m^2], Cc=E[r_i r_m]
    A = r.mul(rm ** 3, axis=0).rolling(**kw).mean()
    B = r.mul(rm ** 2, axis=0).rolling(**kw).mean()
    Cc = r.mul(rm, axis=0).rolling(**kw).mean()

    e_ri_wm3 = (A
                .sub(3 * B.mul(mu_m, axis=0))
                .add(3 * Cc.mul(mu_m ** 2, axis=0))
                .sub(mu_i.mul(mu_m ** 3, axis=0)))
    numerator = e_ri_wm3.sub(mu_i.mul(m3_m, axis=0))
    denom = sig_i.mul(sig_m ** 3, axis=0)
    out = numerator.div(denom)
    return out.where(denom > 0)


# --------------------------------------------------------------------------- #
# The cross-sectional sort -> long-high-cokurt / short-low-cokurt spread
# --------------------------------------------------------------------------- #
def cokurt_spreads(
    ret: pd.DataFrame,
    window: int = 252,
    frac: float = 0.3,
    min_names: int = 10,
) -> pd.DataFrame:
    """Daily equal-weight top-minus-bottom cokurtosis fractile spread.

    On each day ``t`` names are ranked by the trailing cokurtosis known at the close of
    ``t-1`` (one ``shift``). ``hi`` = mean forward day-``t`` return of the top ``frac``
    (high cokurtosis, the long); ``lo`` = mean of the bottom ``frac`` (low cokurtosis, the
    short). ``spread = hi - lo`` (long high-cokurtosis, short low). Days with fewer than
    ``min_names`` ranked names are dropped.
    """
    sig = trailing_cokurt(ret, window).shift(1)  # known at close t-1
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
        low = order[:k]        # low cokurtosis  -> short
        high = order[-k:]      # high cokurtosis -> long
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
def cokurt_stats(spreads: pd.DataFrame, nw_lags: int = 10) -> dict:
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
    window: int = 252,
    frac: float = 0.3,
    min_names: int = 10,
    n_seeds: int = 20,
    n_draws_per_seed: int = 50,
    base_seed: int = 805,
) -> dict:
    """Keep the trailing-cokurtosis sort but read each day's forward return from a
    **column-permuted** panel (signal->outcome link broken, each day's cross-sectional
    distribution preserved). p = share of permuted worlds whose spread mean is >= observed
    (right-tail test on the long-high/short-low spread)."""
    cols = list(ret.columns)
    ncol = len(cols)
    sig = trailing_cokurt(ret, window).shift(1)
    obs = float(cokurt_spreads(ret, window, frac, min_names)["spread"].mean())

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
    """Cost the long-high-cokurt / short-low-cokurt book.

    The signal is a slow trailing-window cokurtosis, but names drift across the fractile
    boundary daily; to stay comparable to the desk's other cross-sectional timers we charge
    2 sides x one-way cost x NAV per day on the long-short book, plus borrow on the short leg.
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
def synthetic_detect(panel: dict[str, pd.DataFrame], window: int = 252,
                     frac: float = 0.3) -> dict:
    """Run the headline cokurtosis stats on a synthetic panel."""
    ret = close_returns(panel)
    sp = cokurt_spreads(ret, window, frac)
    ts = cokurt_stats(sp)
    return {"spread_bps": ts["spread_bps"], "t_nw": ts["t_nw"],
            "welch_t": ts["welch_t"], "n_days": ts["n_days"]}
