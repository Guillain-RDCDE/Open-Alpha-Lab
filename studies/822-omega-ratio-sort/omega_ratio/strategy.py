"""Strategy + inference for Study 822 — Omega-Ratio Sort.

The claim (Keating & Shadwick 2002): the **Omega ratio** at threshold 0,
``Ω(0) = E[max(r,0)] / E[max(−r,0)]`` — the ratio of a name's average gain to its average
loss — is a distribution-aware performance measure that uses *every* moment of the return
distribution, not just the mean and variance a Sharpe ratio uses. Rank a cross-section on
each name's trailing-year Omega and go **long high-Omega / short low-Omega**; the pitch is
that this richer sort out-performs a plain trailing-Sharpe sort.

This is distinct from / directly compared with:

* [814-trailing-sharpe-anomaly](../../814-trailing-sharpe-anomaly/) — the same
  cross-sectional sort machinery on trailing **Sharpe** (mean/std, only the first two
  moments). This study's whole point is the **head-to-head**: does the full gain/loss
  ratio beat mean/std, or re-pick the same names?
* [330-low-volatility-anomaly](../../330-low-volatility-anomaly/) — the **low-vol** tilt
  (sort on trailing volatility alone). Omega's denominator is a downside-loss measure, so
  a high-Omega book is partly a low-vol book; we measure the rank overlap to name the
  confound.

Method:

* **Close-to-close returns.** Build a per-name daily simple-return panel from adjusted
  Close.
* **Trailing Omega(0).** On each name compute, over a rolling ``lookback``-day formation
  window ending ``skip`` (~1 month) days ago (12-1 style, à la momentum), the ratio of
  the average positive daily return to the average absolute negative daily return.
* **Comparators.** Trailing 12-1 Sharpe (mean/std) and trailing volatility on the
  identical window/lag/sort, plus the average per-day rank correlation of the signals.
* **Point-in-time sort.** On each day ``t`` rank the cross-section by the trailing Omega
  known at the close of ``t-1`` (one further ``shift``) and hold day ``t``. Long the top
  ``frac`` (high Omega), short the bottom ``frac`` (low Omega); equal weight.
* **Inference.** Newey-West (HAC) *t* on the daily long-short spread; a one-sample *t*
  and a pooled Welch *t* (high book vs low book) cross-check; a permutation placebo
  breaks the signal->outcome link; a costed timer charges the round-trip friction.
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


def trailing_omega(ret: pd.DataFrame, lookback: int = 252, skip: int = 21,
                   threshold: float = 0.0) -> pd.DataFrame:
    """Trailing 12-1 **Omega(0)** signal: over the rolling ``lookback``-day formation
    window that **ends ``skip`` days ago**, the ratio of the average gain above
    ``threshold`` to the average loss below it,

        Ω(L) = E[max(r − L, 0)] / E[max(L − r, 0)].

    At ``L = 0`` this is the average positive daily return divided by the average
    absolute negative daily return — a gain/loss ratio that reflects the whole
    distribution. Vectorised via two rolling means of the clipped return series. A
    ``.shift(skip)`` then skips the most recent month, so the value on row ``t`` uses
    returns through ``t-skip``. The sort in :func:`fractile_spreads` shifts by one further
    day so a day-``t`` position is formed on information known at the close of ``t-1``.
    """
    excess = ret - threshold
    gain = excess.clip(lower=0.0)
    loss = (-excess).clip(lower=0.0)
    up = gain.rolling(lookback, min_periods=lookback).mean()
    down = loss.rolling(lookback, min_periods=lookback).mean()
    omega = (up / down).where(down > 0)
    return omega.shift(skip)


def trailing_sharpe(ret: pd.DataFrame, lookback: int = 252, skip: int = 21) -> pd.DataFrame:
    """Trailing 12-1 **Sharpe** comparator (study 814): rolling mean ÷ population std
    (ddof=0) of daily returns over the window ending ``skip`` days ago. Value on row
    ``t`` uses returns through ``t-skip``."""
    mean = ret.rolling(lookback, min_periods=lookback).mean()
    std = ret.rolling(lookback, min_periods=lookback).std(ddof=0)
    sharpe = (mean / std).where(std > 0)
    return sharpe.shift(skip)


def trailing_vol(ret: pd.DataFrame, lookback: int = 252, skip: int = 21) -> pd.DataFrame:
    """Trailing realized **volatility** comparator (study 330): rolling population std
    over the window ending ``skip`` days ago. A pure low-vol book sorts on this via
    ``long_high=False``. Value on row ``t`` uses returns through ``t-skip``."""
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
    ``shift``). With ``long_high=True`` (the Omega/Sharpe default) ``hi`` = mean forward
    day-``t`` return of the top ``frac`` (high signal, the long), ``lo`` = mean of the
    bottom ``frac`` (the short), and ``spread = hi - lo`` (long high, short low). With
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
        out_spread.append(hi - lo if long_high else lo - hi)
        out_hi.append(hi); out_lo.append(lo)
        out_n.append(n); out_t.append(idx[i])
    return pd.DataFrame(
        {"spread": out_spread, "hi": out_hi, "lo": out_lo, "n": out_n}, index=out_t
    ).sort_index()


def omega_spreads(ret: pd.DataFrame, lookback: int = 252, skip: int = 21,
                  frac: float = 0.3, min_names: int = 10,
                  threshold: float = 0.0) -> pd.DataFrame:
    """Long-high-Omega / short-low-Omega daily fractile spread (the headline book)."""
    return fractile_spreads(ret, trailing_omega(ret, lookback, skip, threshold),
                            frac, min_names, long_high=True)


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


# --------------------------------------------------------------------------- #
# Head-to-head — Omega vs Sharpe vs low-vol (the honesty question)
# --------------------------------------------------------------------------- #
def _spearman_rows(A: np.ndarray, B: np.ndarray) -> list[float]:
    """Per-row Spearman rank correlation between two signal matrices (NaN-aware)."""
    out = []
    for a, b in zip(A, B):
        mask = ~np.isnan(a) & ~np.isnan(b)
        if mask.sum() < 5:
            continue
        ra = pd.Series(a[mask]).rank().to_numpy()
        rb = pd.Series(b[mask]).rank().to_numpy()
        if ra.std() == 0 or rb.std() == 0:
            continue
        out.append(float(np.corrcoef(ra, rb)[0, 1]))
    return out


def signal_rank_corr(ret: pd.DataFrame, lookback: int = 252, skip: int = 21,
                     threshold: float = 0.0) -> dict:
    """Average per-day cross-sectional rank correlation of the Omega signal with the
    Sharpe comparator and with (minus) volatility — how much the sorts overlap."""
    om = trailing_omega(ret, lookback, skip, threshold).to_numpy(dtype=float)
    sh = trailing_sharpe(ret, lookback, skip).to_numpy(dtype=float)
    nv = (-trailing_vol(ret, lookback, skip)).to_numpy(dtype=float)
    corr_os = _spearman_rows(om, sh)
    corr_ov = _spearman_rows(om, nv)
    return {
        "rho_omega_sharpe": float(np.mean(corr_os)) if corr_os else float("nan"),
        "rho_omega_negvol": float(np.mean(corr_ov)) if corr_ov else float("nan"),
        "n_days": len(corr_os),
    }


def head_to_head(ret: pd.DataFrame, lookback: int = 252, skip: int = 21,
                 frac: float = 0.3, min_names: int = 10,
                 threshold: float = 0.0) -> dict:
    """The three sibling sorts on the identical universe/dates/machinery."""
    om = spread_stats(omega_spreads(ret, lookback, skip, frac, min_names, threshold))
    sh = spread_stats(fractile_spreads(ret, trailing_sharpe(ret, lookback, skip),
                                       frac, min_names, long_high=True))
    lv = spread_stats(fractile_spreads(ret, trailing_vol(ret, lookback, skip),
                                       frac, min_names, long_high=False))
    return {
        "omega_bps": om["spread_bps"], "omega_t": om["t_nw"],
        "sharpe_bps": sh["spread_bps"], "sharpe_t": sh["t_nw"],
        "lowvol_bps": lv["spread_bps"], "lowvol_t": lv["t_nw"],
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
    threshold: float = 0.0,
    n_seeds: int = 20,
    n_draws_per_seed: int = 50,
    base_seed: int = 822,
) -> dict:
    """Keep the trailing-Omega sort but read each day's forward return from a
    **column-permuted** panel (signal->outcome link broken, each day's cross-sectional
    distribution preserved). p = share of permuted worlds whose spread mean is >= observed
    (right-tail test on the long-high/short-low spread)."""
    cols = list(ret.columns)
    ncol = len(cols)
    sig = trailing_omega(ret, lookback, skip, threshold).shift(1)
    obs = float(omega_spreads(ret, lookback, skip, frac, min_names, threshold)["spread"].mean())

    ret_mat = ret.to_numpy(dtype=float)
    pos_of = {c: i for i, c in enumerate(cols)}
    rows_idx, highs, lows = [], [], []
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
    """Cost the long-high-Omega / short-low-Omega book.

    The signal is a trailing-year Omega that turns over slowly, but names drift across the
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
    """Run the headline Omega-spread stats on a synthetic panel."""
    ret = close_returns(panel)
    sp = omega_spreads(ret, lookback, skip, frac)
    ts = spread_stats(sp)
    return {"spread_bps": ts["spread_bps"], "t_nw": ts["t_nw"],
            "welch_t": ts["welch_t"], "n_days": ts["n_days"]}
