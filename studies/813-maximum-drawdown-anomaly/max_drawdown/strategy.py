"""Strategy + inference for Study 813 — the Maximum-Drawdown Anomaly.

The claim: sort a cross-section of stocks on each name's **trailing 12-month maximum
drawdown** (the largest peak-to-trough decline of its cumulative total return) and ask
whether the recently distressed names — deepest past drawdown — subsequently
**under-earn** (distress premium) or **rebound** (reversal). We take no prior on the
sign; the sort into MaxDD fractiles reports the forward long-short spread and we stamp it
by the sign and significance we actually find.

This is distinct from:

* [333-recovery-speed](../../333-recovery-speed/) — how *fast* a name climbs back out of
  a drawdown (a duration/recovery measure), not the **depth** of the worst decline;
* [816-drawdown-duration](../../816-drawdown-duration/) — how *long* a name spends
  underwater (time-in-drawdown), the horizontal axis, not the vertical **depth** here;
* [540-distress-risk](../../540-distress-risk/) — a fundamental default/distress score
  (Campbell-Hilscher-Szilagyi), not a purely price-based trailing drawdown;
* [332-downside-beta](../../332-downside-beta/) — a name's **beta in down markets** (a
  systematic co-movement), not its own realized peak-to-trough decline.

Method:

* **Close-to-close returns.** Build a per-name daily simple-return panel from adjusted
  Close (total-return prices), for measuring the *forward* book return.
* **Trailing maximum drawdown.** On each name compute the rolling ``window``-day maximum
  drawdown of the cumulative price — the deepest peak-to-trough decline inside the
  trailing window (a positive magnitude; larger = more distressed). Fully vectorised via
  a sliding-window view + running peak.
* **Point-in-time sort.** On each day ``t`` rank the cross-section by the trailing MaxDD
  known at the close of ``t-1`` (one ``shift``) and hold day ``t``. **Long the bottom**
  ``frac`` (calmest, shallow drawdown), **short the top** ``frac`` (deepest drawdown,
  the distressed names); equal weight. ``spread = lo - hi`` is therefore
  *long-calm / short-distressed*: a positive spread means the distressed names
  under-earned (distress premium), a negative spread means they rebounded.
* **Inference.** Newey-West (HAC) *t* on the daily long-short spread; a one-sample *t*
  and a pooled Welch *t* (calm book vs distressed book) cross-check; a permutation placebo
  breaks the signal->outcome link; a costed timer charges the round-trip friction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

TRADING_DAYS = 252


# --------------------------------------------------------------------------- #
# Return panel + signal
# --------------------------------------------------------------------------- #
def close_frame(panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Aligned adjusted-Close frame (index=date, columns=ticker)."""
    return pd.DataFrame({s: panel[s]["Close"] for s in panel}).sort_index()


def close_returns(panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Daily simple close-to-close returns (index=date, columns=ticker)."""
    return close_frame(panel).pct_change()


def _maxdd_1d(c: np.ndarray, window: int) -> np.ndarray:
    """Rolling ``window``-day maximum drawdown magnitude of a price series ``c``.

    For the window ending at each day, the running peak inside the window is
    ``maximum.accumulate``; the drawdown is ``price/peak - 1`` (<= 0); the window's
    MaxDD is the most negative value, returned as a **positive magnitude**. Fully
    vectorised over the (n_windows, window) sliding view. NaN for the leading
    ``window-1`` rows that lack a full window.
    """
    n = len(c)
    out = np.full(n, np.nan)
    if n >= window:
        w = sliding_window_view(c, window)            # (n-window+1, window)
        peak = np.maximum.accumulate(w, axis=1)
        dd = w / peak - 1.0                            # <= 0 within each window
        out[window - 1:] = -dd.min(axis=1)             # positive magnitude of worst decline
    return out


def trailing_maxdd(closes: pd.DataFrame, window: int = 252) -> pd.DataFrame:
    """Rolling ``window``-day maximum drawdown magnitude, per name.

    Value on row ``t`` uses prices through ``t`` (inclusive); the sort in
    :func:`maxdd_spreads` shifts by one day so a day-``t`` position is formed on
    information known at ``t-1``. Larger value = deeper recent drawdown = more distressed.
    """
    out = {}
    for s in closes.columns:
        ser = closes[s].dropna()
        m = _maxdd_1d(ser.to_numpy(dtype=float), window)
        out[s] = pd.Series(m, index=ser.index)
    return pd.DataFrame(out).reindex(closes.index)


# --------------------------------------------------------------------------- #
# The cross-sectional sort -> long-calm / short-distressed spread
# --------------------------------------------------------------------------- #
def maxdd_spreads(
    closes: pd.DataFrame,
    window: int = 252,
    frac: float = 0.3,
    min_names: int = 10,
) -> pd.DataFrame:
    """Daily equal-weight bottom-minus-top trailing-MaxDD fractile spread.

    On each day ``t`` names are ranked by the trailing maximum drawdown known at the
    close of ``t-1`` (one ``shift``). ``lo`` = mean forward day-``t`` return of the
    bottom ``frac`` (shallow drawdown, the *calm* long); ``hi`` = mean of the top
    ``frac`` (deepest drawdown, the *distressed* short). ``spread = lo - hi``
    (long calm, short distressed) — positive => distressed under-earned. Days with
    fewer than ``min_names`` ranked names are dropped.
    """
    sig = trailing_maxdd(closes, window).shift(1)   # known at close t-1
    ret = closes.pct_change()
    S = sig.to_numpy(dtype=float)
    R = ret.to_numpy(dtype=float)
    idx = closes.index
    out_spread, out_lo, out_hi, out_n, out_t = [], [], [], [], []
    for i in range(len(idx)):
        row = S[i]
        valid = np.where(~np.isnan(row))[0]
        n = len(valid)
        if n < min_names:
            continue
        k = max(1, int(np.floor(n * frac)))
        order = valid[np.argsort(row[valid], kind="stable")]
        low = order[:k]        # shallow drawdown -> long (calm)
        high = order[-k:]      # deep drawdown    -> short (distressed)
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
def maxdd_stats(spreads: pd.DataFrame, nw_lags: int = 10) -> dict:
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
    closes: pd.DataFrame,
    window: int = 252,
    frac: float = 0.3,
    min_names: int = 10,
    n_seeds: int = 20,
    n_draws_per_seed: int = 50,
    base_seed: int = 813,
) -> dict:
    """Keep the trailing-MaxDD sort but read each day's forward return from a
    **column-permuted** panel (signal->outcome link broken, each day's cross-sectional
    distribution preserved). p = share of permuted worlds whose spread mean is >=
    observed (right-tail test on the long-calm / short-distressed spread)."""
    cols = list(closes.columns)
    ncol = len(cols)
    ret = closes.pct_change()
    sig = trailing_maxdd(closes, window).shift(1)
    obs = float(maxdd_spreads(closes, window, frac, min_names)["spread"].mean())

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
    """Cost the long-calm / short-distressed book.

    The signal is a trailing-12-month drawdown that turns over slowly, but names drift
    across the fractile boundary daily; to stay comparable to the desk's other
    cross-sectional timers we charge 2 sides x one-way cost x NAV per day on the
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
def synthetic_detect(panel: dict[str, pd.DataFrame], window: int = 252,
                     frac: float = 0.3) -> dict:
    """Run the headline MaxDD stats on a synthetic panel."""
    closes = close_frame(panel)
    sp = maxdd_spreads(closes, window, frac)
    ts = maxdd_stats(sp)
    return {"spread_bps": ts["spread_bps"], "t_nw": ts["t_nw"],
            "welch_t": ts["welch_t"], "n_days": ts["n_days"]}
