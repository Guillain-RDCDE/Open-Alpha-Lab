"""Strategy + inference for Study 819 — Abnormal-Volume Shock.

The claim (Garfinkel & Sokobin 2006): **unusual trading volume** proxies attention /
opinion divergence; names printing abnormally high volume go on to earn a **positive
subsequent drift**. Sort a cross-section on its recent **standardised abnormal volume**;
a long **high-abnormal-volume** / short **low-abnormal-volume** book should earn a
positive spread.

This is distinct from:

* [512-high-volume-return-premium](../../512-high-volume-return-premium/) — the **level**
  of trading volume / dollar volume (a liquidity-premium sort), not volume *relative to a
  name's own trailing norm*. A perennially heavy name (AAPL) is high-*level* every day but
  is only high-*abnormal* on the days it spikes.
* [141-turnover-anomaly](../../141-turnover-anomaly/) — **share turnover** (volume ÷
  shares outstanding), a slow liquidity/attention state variable, not a short-window
  standardised **shock** off the recent mean.
* [254-wsb-mentions](../../254-wsb-mentions/) — an **exogenous** social-media attention
  proxy (Reddit mention counts). Here the attention proxy is **endogenous** to the tape:
  volume abnormal versus its own 60-day benchmark, no alt-data feed.

Method:

* **Standardised abnormal volume.** For each name, on each day, ``z = (Volume − trailing
  60-day mean Volume) / trailing 60-day std Volume`` — how many sigmas today's volume is
  above its own recent norm.
* **Formation window.** The signal is the mean of ``z`` over the last ``form`` (~5) days —
  a short abnormal-volume *shock*, not a one-day blip.
* **Point-in-time sort.** On each day ``t`` rank the cross-section by the abnormal-volume
  signal known at the close of ``t-1`` (one ``shift``) and hold day ``t``. Long the top
  ``frac`` (high abnormal volume), short the bottom ``frac`` (low abnormal volume); equal
  weight.
* **Inference.** Newey-West (HAC) *t* on the daily long-short spread; a one-sample *t*
  and a pooled Welch *t* (high book vs low book) cross-check; a permutation placebo
  breaks the signal->outcome link; a costed timer charges the round-trip friction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# --------------------------------------------------------------------------- #
# Return / volume panels + signal
# --------------------------------------------------------------------------- #
def close_returns(panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Daily simple close-to-close returns (index=date, columns=ticker)."""
    closes = pd.DataFrame({s: panel[s]["Close"] for s in panel}).sort_index()
    return closes.pct_change()


def volume_panel(panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Daily traded volume (index=date, columns=ticker)."""
    return pd.DataFrame({s: panel[s]["Volume"] for s in panel}).sort_index()


def std_abnormal_volume(vol: pd.DataFrame, lookback: int = 60) -> pd.DataFrame:
    """Standardised abnormal volume per name: ``(V − mean_60) / std_60``.

    The trailing ``lookback``-day mean and std form each name's own volume benchmark;
    the value on row ``t`` uses volume through ``t`` (inclusive). The sort in
    :func:`avol_spreads` shifts by one day so a day-``t`` position is formed on
    information known at ``t-1``."""
    m = vol.rolling(lookback, min_periods=lookback).mean()
    s = vol.rolling(lookback, min_periods=lookback).std(ddof=0)
    z = (vol - m) / s.where(s > 0)
    return z


def abnormal_volume_signal(
    vol: pd.DataFrame, lookback: int = 60, form: int = 5
) -> pd.DataFrame:
    """Formation-window abnormal-volume signal: the ``form``-day mean of the standardised
    abnormal volume. High = a sustained attention/disagreement shock."""
    z = std_abnormal_volume(vol, lookback)
    return z.rolling(form, min_periods=form).mean()


# --------------------------------------------------------------------------- #
# The cross-sectional sort -> long-high-avol / short-low-avol spread
# --------------------------------------------------------------------------- #
def avol_spreads(
    ret: pd.DataFrame,
    vol: pd.DataFrame,
    lookback: int = 60,
    form: int = 5,
    frac: float = 0.3,
    min_names: int = 10,
) -> pd.DataFrame:
    """Daily equal-weight top-minus-bottom abnormal-volume fractile spread.

    On each day ``t`` names are ranked by the abnormal-volume signal known at the close of
    ``t-1`` (one ``shift``). ``hi`` = mean forward day-``t`` return of the top ``frac``
    (high abnormal volume, the long); ``lo`` = mean of the bottom ``frac`` (low abnormal
    volume, the short). ``spread = hi - lo`` (long high-abnormal-volume, short low). Days
    with fewer than ``min_names`` ranked names are dropped. The signal is aligned to the
    return index so both share the same rows."""
    sig = abnormal_volume_signal(vol, lookback, form).shift(1)  # known at close t-1
    sig = sig.reindex(index=ret.index, columns=ret.columns)
    S = sig.to_numpy(dtype=float)
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
        low = order[:k]        # low abnormal volume  -> short
        high = order[-k:]      # high abnormal volume -> long
        rr = R[i]
        hi = float(np.nanmean(rr[high]))
        lo = float(np.nanmean(rr[low]))
        out_spread.append(hi - lo); out_hi.append(hi); out_lo.append(lo)
        out_n.append(n); out_t.append(idx[i])
    return pd.DataFrame(
        {"spread": out_spread, "hi": out_hi, "lo": out_lo, "n": out_n}, index=out_t
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
def avol_stats(spreads: pd.DataFrame, nw_lags: int = 10) -> dict:
    sp = spreads["spread"].to_numpy(dtype=float)
    return {
        "n_days": int(len(spreads)),
        "spread_bps": float(np.nanmean(sp) * 1e4),
        "t_nw": newey_west_t(sp, nw_lags),
        "t_1s": one_sample_t(sp),
        "hi_bps": float(np.nanmean(spreads["hi"].to_numpy()) * 1e4),
        "lo_bps": float(np.nanmean(spreads["lo"].to_numpy()) * 1e4),
        "welch_t": welch_t(spreads["hi"].to_numpy(), spreads["lo"].to_numpy()),
    }


# --------------------------------------------------------------------------- #
# Placebo — is the spread real, or a lucky alignment of the sort?
# --------------------------------------------------------------------------- #
def placebo_pvalue(
    ret: pd.DataFrame,
    vol: pd.DataFrame,
    lookback: int = 60,
    form: int = 5,
    frac: float = 0.3,
    min_names: int = 10,
    n_seeds: int = 20,
    n_draws_per_seed: int = 50,
    base_seed: int = 819,
) -> dict:
    """Keep the abnormal-volume sort but read each day's forward return from a
    **column-permuted** panel (signal->outcome link broken, each day's cross-sectional
    distribution preserved). p = share of permuted worlds whose spread mean is >= observed
    (right-tail test on the long-high/short-low spread)."""
    cols = list(ret.columns)
    ncol = len(cols)
    sig = abnormal_volume_signal(vol, lookback, form).shift(1)
    sig = sig.reindex(index=ret.index, columns=ret.columns)
    obs = float(avol_spreads(ret, vol, lookback, form, frac, min_names)["spread"].mean())

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
    """Cost the long-high-avol / short-low-avol book.

    The signal is a short abnormal-volume shock that rotates roughly weekly, but names
    drift across the fractile boundary daily; to stay comparable to the desk's other
    cross-sectional timers we charge 2 sides × one-way cost × NAV per day on the
    long-short book, plus borrow on the short leg.
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
def synthetic_detect(panel: dict[str, pd.DataFrame], lookback: int = 60, form: int = 5,
                     frac: float = 0.3) -> dict:
    """Run the headline abnormal-volume stats on a synthetic panel."""
    ret = close_returns(panel)
    vol = volume_panel(panel)
    sp = avol_spreads(ret, vol, lookback, form, frac)
    ts = avol_stats(sp)
    return {"spread_bps": ts["spread_bps"], "t_nw": ts["t_nw"],
            "welch_t": ts["welch_t"], "n_days": ts["n_days"]}
