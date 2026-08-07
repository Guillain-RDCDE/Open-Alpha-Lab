"""Strategy + inference for Study 817 — Realized-Volatility Trend.

The claim under test: sort a cross-section of stocks on their **vol trend** — the
*change* in realized volatility, distinct from the low-vol *level* effect (study 330).
Each name's vol trend is ``(trailing 21d realized vol) / (trailing 63d realized vol) -
1``: **positive** = short-window vol running above its longer average (vol **rising**),
**negative** = vol **falling**. The story says rising-vol names keep **de-rating** and
falling-vol names **re-rate**, so a long **falling-vol** / short **rising-vol** book
should earn a positive spread.

This is distinct from:

* [330-low-volatility](../../330-low-volatility/) — the low-vol **level** anomaly
  (low-vol names out-earn high-vol names). This study sorts on the **trend** (change) in
  vol, not the level; the two are near-orthogonal by construction (a name can be
  low-vol-and-rising or high-vol-and-falling).
* [501-idiosyncratic-volatility](../../501-idiosyncratic-volatility/) — the level of
  **residual** (market-model) vol, still a level not a trend.
* [6-clockwork-volatility](../../6-clockwork-volatility/) — the *calendar seasonality*
  of aggregate vol (a time-series clock), not a **cross-sectional** name-by-name trend.

Method:

* **Close-to-close returns.** Build a per-name daily simple-return panel from adjusted
  Close.
* **Vol trend.** On each name compute the rolling 21-day and 63-day realized vol
  (std of daily returns) and form ``vol21/vol63 - 1`` (value on row ``t`` uses returns
  through ``t``).
* **Point-in-time sort.** On each day ``t`` rank the cross-section by the vol trend
  known at the close of ``t-1`` (one ``shift``) and hold day ``t``. Long the bottom
  ``frac`` (most falling vol), short the top ``frac`` (most rising vol); equal weight.
* **Inference.** Newey-West (HAC) *t* on the daily long-short spread; a one-sample *t*
  and a pooled Welch *t* (falling book vs rising book) cross-check; a permutation placebo
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


def trailing_vol(ret: pd.DataFrame, window: int) -> pd.DataFrame:
    """Rolling ``window``-day realized volatility (std, ddof=0) of daily returns."""
    return ret.rolling(window, min_periods=window).std(ddof=0)


def vol_trend(ret: pd.DataFrame, short_w: int = 21, long_w: int = 63) -> pd.DataFrame:
    """Vol trend ``vol(short_w)/vol(long_w) - 1``, per name.

    Positive = short-window realized vol running above its longer-window average (vol
    **rising**); negative = vol **falling**. Value on row ``t`` uses returns through
    ``t`` (inclusive); the sort in :func:`vol_spreads` shifts by one day so a day-``t``
    position is formed on information known at ``t-1``.
    """
    vs = trailing_vol(ret, short_w)
    vl = trailing_vol(ret, long_w)
    out = vs / vl - 1.0
    return out.where(vl > 0)


# --------------------------------------------------------------------------- #
# The cross-sectional sort -> long-falling-vol / short-rising-vol spread
# --------------------------------------------------------------------------- #
def vol_spreads(
    ret: pd.DataFrame,
    short_w: int = 21,
    long_w: int = 63,
    frac: float = 0.3,
    min_names: int = 10,
) -> pd.DataFrame:
    """Daily equal-weight bottom-minus-top vol-trend fractile spread.

    On each day ``t`` names are ranked by the vol trend known at the close of ``t-1``
    (one ``shift``). ``lo`` = mean forward day-``t`` return of the bottom ``frac`` (most
    falling vol, the long); ``hi`` = mean of the top ``frac`` (most rising vol, the
    short). ``spread = lo - hi`` (long falling-vol, short rising-vol). Days with fewer
    than ``min_names`` ranked names are dropped.
    """
    sig = vol_trend(ret, short_w, long_w).shift(1)  # known at close t-1
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
        low = order[:k]        # falling vol -> long
        high = order[-k:]      # rising vol  -> short
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
def vol_stats(spreads: pd.DataFrame, nw_lags: int = 10) -> dict:
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
    short_w: int = 21,
    long_w: int = 63,
    frac: float = 0.3,
    min_names: int = 10,
    n_seeds: int = 20,
    n_draws_per_seed: int = 50,
    base_seed: int = 817,
) -> dict:
    """Keep the vol-trend sort but read each day's forward return from a
    **column-permuted** panel (signal->outcome link broken, each day's
    cross-sectional distribution preserved). p = share of permuted worlds whose
    spread mean is >= observed (right-tail test on the long-falling/short-rising
    spread)."""
    cols = list(ret.columns)
    ncol = len(cols)
    sig = vol_trend(ret, short_w, long_w).shift(1)
    obs = float(vol_spreads(ret, short_w, long_w, frac, min_names)["spread"].mean())

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
# Additivity vs the low-vol LEVEL effect (study 330)
# --------------------------------------------------------------------------- #
def level_spreads(
    ret: pd.DataFrame,
    long_w: int = 63,
    frac: float = 0.3,
    min_names: int = 10,
) -> pd.DataFrame:
    """The plain low-vol LEVEL sort — long low realized vol, short high realized vol.

    Same machinery as :func:`vol_spreads`, but the ranking signal is the *level* of
    trailing ``long_w``-day realized vol (not its trend). Used to ask whether the vol
    *trend* is additive to the classic low-vol level anomaly (study 330).
    """
    sig = trailing_vol(ret, long_w).shift(1)
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
        low = order[:k]        # low vol  -> long
        high = order[-k:]      # high vol -> short
        rr = R[i]
        lo = float(np.nanmean(rr[low]))
        hi = float(np.nanmean(rr[high]))
        out_spread.append(lo - hi); out_lo.append(lo); out_hi.append(hi)
        out_n.append(n); out_t.append(idx[i])
    return pd.DataFrame(
        {"spread": out_spread, "lo": out_lo, "hi": out_hi, "n": out_n}, index=out_t
    ).sort_index()


def additivity(ret: pd.DataFrame, short_w: int = 21, long_w: int = 63,
               frac: float = 0.3, nw_lags: int = 10) -> dict:
    """Regress the trend spread on the level spread; is the trend's alpha additive?

    Fits ``trend_spread_t = a + b * level_spread_t + e_t`` on the overlapping days.
    ``a`` (intercept) is the part of the trend spread orthogonal to the level anomaly —
    a Newey-West *t* on the residual ``trend - b*level`` says whether the trend carries
    anything the level does not. Also reports the correlation of the two spreads.
    """
    tr = vol_spreads(ret, short_w, long_w, frac)
    lv = level_spreads(ret, long_w, frac)
    joined = tr[["spread"]].join(lv[["spread"]], how="inner", lsuffix="_tr", rsuffix="_lv")
    y = joined["spread_tr"].to_numpy(dtype=float)
    x = joined["spread_lv"].to_numpy(dtype=float)
    n = len(y)
    if n < 3:
        return {"n_days": n, "corr": float("nan"), "beta": float("nan"),
                "alpha_bps": float("nan"), "alpha_t_nw": float("nan")}
    X = np.column_stack([np.ones(n), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    a, b = float(coef[0]), float(coef[1])
    resid = y - b * x            # trend spread net of its level-explained part
    return {
        "n_days": n,
        "corr": float(np.corrcoef(x, y)[0, 1]),
        "beta": b,
        "alpha_bps": a * 1e4,
        "alpha_t_nw": newey_west_t(resid, nw_lags),
        "level_bps": float(np.nanmean(x) * 1e4),
        "level_t_nw": newey_west_t(x, nw_lags),
    }


# --------------------------------------------------------------------------- #
# The costed timer
# --------------------------------------------------------------------------- #
def timer_stats(
    spreads: pd.DataFrame,
    cost_bps: float = 5.0,
    borrow_bps_yr: float = 50.0,
) -> dict:
    """Cost the long-falling-vol / short-rising-vol book.

    The signal is a trailing vol ratio that turns over roughly monthly, but names drift
    across the fractile boundary daily; we charge a conservative daily round-trip on the
    fraction of the 2x-NAV book that rotates. To stay comparable to the desk's other
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
def synthetic_detect(panel: dict[str, pd.DataFrame], short_w: int = 21,
                     long_w: int = 63, frac: float = 0.3) -> dict:
    """Run the headline vol-trend stats on a synthetic panel."""
    ret = close_returns(panel)
    sp = vol_spreads(ret, short_w, long_w, frac)
    ts = vol_stats(sp)
    return {"spread_bps": ts["spread_bps"], "t_nw": ts["t_nw"],
            "welch_t": ts["welch_t"], "n_days": ts["n_days"]}
