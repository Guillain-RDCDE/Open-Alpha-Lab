"""Strategy + inference for Study 903 — Sector-Neutral Low-Vol.

The claim (low-volatility anomaly, Baker-Bradley-Wurgler 2011; Frazzini-Pedersen 2014):
calm stocks out-earn wild ones risk-adjusted, so a long-low-vol / short-high-vol book earns
a positive spread. The **critique** this study tests: a naive low-vol sort quietly loads the
structurally calm **sectors** (staples, health care) and shorts the wild ones (tech, energy)
— so part of the "edge" is a defensive-**sector** bet, not a stock-level effect.

We build two books on the *same* trailing-volatility signal and the *same* panel:

* **Raw low-vol.** Rank the whole cross-section by trailing volatility; long the bottom
  ``frac`` (calmest), short the top ``frac`` (wildest), equal-weight. This carries the
  sector tilt.
* **Sector-neutral low-vol.** First **demean each name's volatility by its sector's
  cross-sectional median** (so a name is judged calm/wild *relative to its own sector*),
  then run the identical bottom-minus-top sort on the residual. The extreme-low and
  extreme-high residuals are drawn ~evenly across sectors, so the book is ~sector-neutral —
  the sector bet is removed and only a genuine *stock-level* low-vol effect can survive.

Method:

* **Close-to-close returns.** Per-name daily simple-return panel from adjusted Close.
* **Trailing volatility.** Rolling ``window``-day standard deviation of daily returns per
  name (value on row ``t`` uses returns through ``t``).
* **Sector demean.** On each day subtract, from each name's trailing vol, the *median*
  trailing vol of its own sector that day (the sector-neutral signal).
* **Point-in-time sort.** On each day ``t`` rank on the signal **known at the close of
  ``t-1``** (one ``shift``) and hold day ``t``. Long the low-vol book, short the high-vol
  book; equal weight.
* **Inference.** Newey-West (HAC) *t* on the daily long-short spread; a one-sample *t* and a
  pooled Welch *t* (low book vs high book) cross-check; a permutation placebo breaks the
  signal->outcome link; a costed timer charges the round-trip friction; a sector-exposure
  diagnostic measures the defensive tilt each book carries.
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


def trailing_vol(ret: pd.DataFrame, window: int = 63) -> pd.DataFrame:
    """Rolling ``window``-day realized volatility (std of daily returns), per name.

    Value on row ``t`` uses returns through ``t`` (inclusive); the sort in
    :func:`vol_spreads` shifts by one day so a day-``t`` position is formed on information
    known at the close of ``t-1``.
    """
    return ret.rolling(window, min_periods=window).std(ddof=0)


def sector_demean(signal: pd.DataFrame, sectors: pd.Series) -> pd.DataFrame:
    """Subtract, from each name's signal, the cross-sectional **median** of its own sector,
    per day. Vectorised (group the columns by sector, take a per-day median, broadcast).

    The result is the *sector-relative* signal: a name's volatility net of how volatile its
    whole sector is that day. Sorting on this residual is sector-neutral by construction.
    """
    sec = sectors.reindex(signal.columns)
    out = signal.copy()
    for label, cols in sec.groupby(sec).groups.items():
        cols = list(cols)
        med = signal[cols].median(axis=1)
        out[cols] = signal[cols].sub(med, axis=0)
    return out


# --------------------------------------------------------------------------- #
# The cross-sectional sort -> long-low-vol / short-high-vol spread
# --------------------------------------------------------------------------- #
def vol_spreads(
    ret: pd.DataFrame,
    sectors: pd.Series | None = None,
    window: int = 63,
    frac: float = 0.3,
    neutral: bool = True,
    min_names: int = 10,
) -> pd.DataFrame:
    """Daily equal-weight low-minus-high trailing-vol fractile spread.

    ``neutral=True`` demeans the trailing vol within sector first (sector-neutral book);
    ``neutral=False`` sorts the raw vol (the sector-tilted book). On each day ``t`` names are
    ranked by the signal known at the close of ``t-1`` (one ``shift``). ``lo`` = mean forward
    day-``t`` return of the bottom ``frac`` (**low** vol, the long); ``hi`` = mean of the top
    ``frac`` (**high** vol, the short). ``spread = lo - hi``. Days with fewer than
    ``min_names`` ranked names are dropped.
    """
    sig = trailing_vol(ret, window)
    if neutral:
        if sectors is None:
            raise ValueError("neutral=True requires a `sectors` series")
        sig = sector_demean(sig, sectors)
    sig = sig.shift(1)  # known at close t-1
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


def defensive_tilt(
    ret: pd.DataFrame,
    sectors: pd.Series,
    defensive: tuple[str, ...],
    window: int = 63,
    frac: float = 0.3,
    neutral: bool = True,
    min_names: int = 10,
) -> dict:
    """How much of the long (low-vol) book sits in *defensive* sectors, on average.

    Runs the same point-in-time sort as :func:`vol_spreads` and, each day, measures the
    share of the long book (bottom ``frac``) whose names belong to ``defensive`` sectors,
    minus the share of the short book. A large positive number means the book is a
    defensive-sector bet; a sector-neutral book should sit near the universe's defensive
    weight with a small long-minus-short gap.
    """
    sig = trailing_vol(ret, window)
    if neutral:
        sig = sector_demean(sig, sectors)
    sig = sig.shift(1)
    S = sig.to_numpy(dtype=float)
    cols = list(ret.columns)
    is_def = np.array([sectors.get(c, "") in defensive for c in cols])
    base = float(is_def.mean())
    long_def, short_def = [], []
    for i in range(len(ret.index)):
        row = S[i]
        valid = np.where(~np.isnan(row))[0]
        if len(valid) < min_names:
            continue
        k = max(1, int(np.floor(len(valid) * frac)))
        order = valid[np.argsort(row[valid], kind="stable")]
        long_def.append(float(is_def[order[:k]].mean()))
        short_def.append(float(is_def[order[-k:]].mean()))
    long_def = np.asarray(long_def)
    short_def = np.asarray(short_def)
    return {
        "universe_defensive_share": base,
        "long_defensive_share": float(long_def.mean()) if len(long_def) else float("nan"),
        "short_defensive_share": float(short_def.mean()) if len(short_def) else float("nan"),
        "long_minus_short_defensive": float((long_def - short_def).mean())
        if len(long_def) else float("nan"),
    }


# --------------------------------------------------------------------------- #
# Inference primitives  (copied from Study 803)
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


def sharpe_ci_bootstrap(
    x: np.ndarray, n_boot: int = 2000, alpha: float = 0.05, seed: int = 903,
) -> tuple[float, float]:
    """Circular block-bootstrap CI for the annualised Sharpe of a daily return series."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = x.size
    block = 10
    if n < block + 1:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    sh = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel() % n
        s = x[idx[:n]]
        sd = s.std(ddof=1)
        sh[b] = s.mean() / sd * np.sqrt(TRADING_DAYS) if sd > 0 else np.nan
    lo, hi = np.nanquantile(sh, [alpha / 2, 1 - alpha / 2])
    return (float(lo), float(hi))


# --------------------------------------------------------------------------- #
# Headline stats
# --------------------------------------------------------------------------- #
def vol_stats(spreads: pd.DataFrame, nw_lags: int = 10) -> dict:
    sp = spreads["spread"].to_numpy(dtype=float)
    sp_clean = sp[~np.isnan(sp)]
    sd = sp_clean.std(ddof=1) if len(sp_clean) > 1 else float("nan")
    sharpe = (sp_clean.mean() / sd * np.sqrt(TRADING_DAYS)) if sd and sd > 0 else float("nan")
    return {
        "n_days": int(len(spreads)),
        "spread_bps": float(np.nanmean(sp) * 1e4),
        "t_nw": newey_west_t(sp, nw_lags),
        "t_1s": one_sample_t(sp),
        "lo_bps": float(np.nanmean(spreads["lo"].to_numpy()) * 1e4),
        "hi_bps": float(np.nanmean(spreads["hi"].to_numpy()) * 1e4),
        "welch_t": welch_t(spreads["lo"].to_numpy(), spreads["hi"].to_numpy()),
        "gross_sharpe": float(sharpe),
    }


# --------------------------------------------------------------------------- #
# Placebo — is the spread real, or a lucky alignment of the sort?
# --------------------------------------------------------------------------- #
def placebo_pvalue(
    ret: pd.DataFrame,
    sectors: pd.Series | None = None,
    window: int = 63,
    frac: float = 0.3,
    neutral: bool = True,
    min_names: int = 10,
    n_seeds: int = 20,
    n_draws_per_seed: int = 50,
    base_seed: int = 903,
) -> dict:
    """Keep the trailing-vol sort but read each day's forward return from a
    **column-permuted** panel (signal->outcome link broken, each day's cross-sectional
    distribution preserved). p = share of permuted worlds whose spread mean is >= observed
    (right-tail test on the long-low-vol / short-high-vol spread)."""
    cols = list(ret.columns)
    ncol = len(cols)
    sig = trailing_vol(ret, window)
    if neutral:
        sig = sector_demean(sig, sectors)
    sig = sig.shift(1)
    obs = float(
        vol_spreads(ret, sectors, window, frac, neutral, min_names)["spread"].mean()
    )

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
    """Cost the long-low-vol / short-high-vol book.

    Trailing volatility turns over slowly, but names drift across the fractile boundary; to
    stay comparable to the desk's other cross-sectional timers we charge 2 sides x one-way
    cost x NAV per day on the long-short book, plus borrow on the short (high-vol) leg.
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
def synthetic_detect(
    panel: dict[str, pd.DataFrame],
    sectors: pd.Series,
    window: int = 63,
    frac: float = 0.3,
    neutral: bool = True,
) -> dict:
    """Run the headline vol stats on a synthetic panel (sector-neutral by default)."""
    ret = close_returns(panel)
    sp = vol_spreads(ret, sectors, window, frac, neutral)
    ts = vol_stats(sp)
    return {"spread_bps": ts["spread_bps"], "t_nw": ts["t_nw"],
            "welch_t": ts["welch_t"], "n_days": ts["n_days"]}
