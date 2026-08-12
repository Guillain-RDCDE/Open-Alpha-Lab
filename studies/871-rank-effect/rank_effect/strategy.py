"""Strategy + inference for Study 871 — The Rank Effect.

The claim (Hartzmark 2015): investors disproportionately **sell the best- and
worst-ranked positions** in their portfolio, so the names an investor ranks at the top
and the bottom carry predictable **selling pressure**. Cross-sectional proxy: each day
**rank the names by trailing return**; test whether the **extreme-ranked** names
(rank 1 = best, rank N = worst) go on to **underperform the middle** of the ranking next
period — a *rank-extremity short* (long the middle band, short both tails) — while
**controlling for the raw trailing-return level** so the test isolates rank *position*
from plain momentum / reversal.

This is distinct from:

* [327-disposition-effect](../../327-disposition-effect/) — the tendency to sell
  **winners** and ride **losers** relative to a *purchase-price* reference. The rank
  effect is reference-free: it is the **rank position within the portfolio** (top *and*
  bottom), not the sign of the gain, that drives the trade.
* [365-lottery-max-effect](../../365-lottery-max-effect/) — sorts on the single
  **maximum daily return** (a right-tail lottery proxy). Here the signal is a name's
  **rank extremity** among its peers, symmetric across *both* tails.
* [806-prospect-theory-value](../../806-prospect-theory-value/) — a prospect-theory
  **value** of the whole gain/loss distribution. The rank effect uses only the *ordinal
  position* of a name in the cross-section, not a valuation of its return path.
* [202-fifty-two-week-low](../../202-fifty-two-week-low/) — nearness to a **52-week
  extreme price**, an anchor relative to a name's own history; the rank effect is a
  **cross-sectional** position among peers this period.

Method:

* **Close-to-close returns.** Build a per-name daily simple-return panel from adjusted
  Close.
* **Trailing return + cross-sectional rank.** On each name compute the rolling
  ``window``-day trailing return (the ranking variable *and* the level control). Each
  day rank the cross-section; the fractional rank ``u ∈ [0,1]`` gives an **extremity**
  score ``ext = |2u − 1|`` (0 in the middle, 1 at either tail).
* **Point-in-time sort.** On each day ``t`` the ranking uses trailing returns known at
  the close of ``t−1`` (one ``shift``) and the book is held day ``t``. The
  rank-extremity spread longs the central band, shorts both tails, equal weight.
* **Controlling for the raw return level.** A daily Fama-MacBeth cross-sectional
  regression of the forward return on the standardised **trailing-return level** *and*
  the **extremity** score; the extremity slope's time-series (Newey-West) *t* is the
  level-controlled headline. The both-tails spread is itself approximately level-neutral
  by construction (a big winner and a big loser average toward the middle).
* **Inference.** Newey-West (HAC) *t* on the daily spread; a one-sample *t* and a pooled
  Welch *t* (middle book vs extremes book) cross-check; a permutation placebo breaks the
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


def trailing_return(ret: pd.DataFrame, window: int = 42) -> pd.DataFrame:
    """Rolling ``window``-day cumulative simple return per name.

    The ranking variable *and* the raw-level control. Value on row ``t`` uses returns
    through ``t`` (inclusive); the sorts below shift by one day so a day-``t`` position
    is formed on information known at ``t−1``. Vectorised via log-return cumulation.
    """
    logr = np.log1p(ret)
    out = np.expm1(logr.rolling(window, min_periods=window).sum())
    return out


# --------------------------------------------------------------------------- #
# The rank-extremity spread -> long-middle / short-extremes
# --------------------------------------------------------------------------- #
def extremity_spreads(
    ret: pd.DataFrame,
    window: int = 42,
    tail_frac: float = 0.2,
    min_names: int = 10,
) -> pd.DataFrame:
    """Daily equal-weight middle-minus-extremes rank-extremity spread.

    On each day ``t`` names are ranked by the trailing return known at the close of
    ``t−1`` (one ``shift``). ``k = floor(n * tail_frac)`` names from *each* tail form the
    **extremes** book (rank 1 winners + rank N losers, ``2k`` names); the ``2k`` names
    nearest the centre of the ranking form the **middle** book. ``spread = mid − ext``
    (long the middle, short both extremes). ``lvl_*`` report each book's mean trailing
    return — the level-neutrality diagnostic (both-tails extremes average toward the
    middle). Days with fewer than ``min_names`` ranked names are dropped.
    """
    sig = trailing_return(ret, window).shift(1)  # known at close t-1
    S = sig.to_numpy(dtype=float)
    R = ret.to_numpy(dtype=float)
    idx = ret.index
    out = {"spread": [], "mid": [], "ext": [], "lvl_mid": [], "lvl_ext": [], "n": []}
    ts = []
    for i in range(len(idx)):
        row = S[i]
        valid = np.where(~np.isnan(row))[0]
        n = len(valid)
        if n < min_names:
            continue
        k = max(1, int(np.floor(n * tail_frac)))
        if 2 * k >= n:                      # need a distinct middle band
            continue
        order = valid[np.argsort(row[valid], kind="stable")]   # ascending trailing ret
        extremes = np.concatenate([order[:k], order[-k:]])     # both tails
        c = n // 2
        half = k                             # 2k centred names
        mid = order[max(0, c - half):c - half + 2 * k]
        rr = R[i]
        m = float(np.nanmean(rr[mid]))
        e = float(np.nanmean(rr[extremes]))
        out["spread"].append(m - e); out["mid"].append(m); out["ext"].append(e)
        out["lvl_mid"].append(float(np.nanmean(row[mid])))
        out["lvl_ext"].append(float(np.nanmean(row[extremes])))
        out["n"].append(n); ts.append(idx[i])
    return pd.DataFrame(out, index=ts).sort_index()


# --------------------------------------------------------------------------- #
# Level-controlled spread — the explicit "raw return level" control
# --------------------------------------------------------------------------- #
def level_controlled_spreads(
    ret: pd.DataFrame,
    window: int = 42,
    tail_frac: float = 0.2,
    min_names: int = 10,
) -> pd.DataFrame:
    """Middle-minus-extremes spread on **level-residualised** forward returns.

    The both-tails raw spread (:func:`extremity_spreads`) is *approximately*
    level-neutral by construction, but a smooth momentum / reversal curve in the raw
    return level can still leak in. Here, each day, the forward returns are first
    **residualised** against a quadratic in the standardised trailing-return level
    ``z`` (``y = a + b·z + c·z² + resid``, so *any* smooth function of the raw level is
    removed), and the middle-minus-extremes spread is measured on the **residuals**.
    What survives is the return to rank *extremity* **holding the raw return level
    fixed** — the level-controlled rank-extremity effect. Point-in-time (signal at
    ``t−1``); NaNs dropped; days with too few names skipped.
    """
    sig = trailing_return(ret, window).shift(1)
    S = sig.to_numpy(dtype=float)
    R = ret.to_numpy(dtype=float)
    idx = ret.index
    spread, ts, nn = [], [], []
    for i in range(len(idx)):
        row = S[i]
        valid = np.where(~np.isnan(row))[0]
        if len(valid) < min_names:
            continue
        lv = row[valid]
        y = R[i][valid]
        keep = ~np.isnan(y)
        lv, y = lv[keep], y[keep]
        n = len(y)
        if n < min_names:
            continue
        sd = lv.std(ddof=0)
        if sd <= 0:
            continue
        k = max(1, int(np.floor(n * tail_frac)))
        if 2 * k >= n:
            continue
        z = (lv - lv.mean()) / sd
        X = np.column_stack([np.ones_like(y), z, z * z])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        order = np.argsort(lv, kind="stable")           # ascending trailing return
        extremes = np.concatenate([order[:k], order[-k:]])
        c = n // 2
        mid = order[max(0, c - k):c - k + 2 * k]
        spread.append(float(resid[mid].mean() - resid[extremes].mean()))
        ts.append(idx[i]); nn.append(n)
    return pd.DataFrame({"spread": spread, "n": nn}, index=ts).sort_index()


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
def rank_stats(spreads: pd.DataFrame, nw_lags: int = 10) -> dict:
    sp = spreads["spread"].to_numpy(dtype=float)
    return {
        "n_days": int(len(spreads)),
        "spread_bps": float(np.nanmean(sp) * 1e4),
        "t_nw": newey_west_t(sp, nw_lags),
        "t_1s": one_sample_t(sp),
        "mid_bps": float(np.nanmean(spreads["mid"].to_numpy()) * 1e4),
        "ext_bps": float(np.nanmean(spreads["ext"].to_numpy()) * 1e4),
        "welch_t": welch_t(spreads["mid"].to_numpy(), spreads["ext"].to_numpy()),
        "lvl_mid": float(np.nanmean(spreads["lvl_mid"].to_numpy())),
        "lvl_ext": float(np.nanmean(spreads["lvl_ext"].to_numpy())),
    }


def lc_stats(lc: pd.DataFrame, nw_lags: int = 10) -> dict:
    """Headline stats on the level-controlled (residualised) middle-minus-extremes spread."""
    sp = lc["spread"].to_numpy(dtype=float)
    return {
        "n_days": int(len(lc)),
        "spread_bps": float(np.nanmean(sp) * 1e4),
        "t_nw": newey_west_t(sp, nw_lags),
        "t_1s": one_sample_t(sp),
    }


# --------------------------------------------------------------------------- #
# Placebo — is the spread real, or a lucky alignment of the sort?
# --------------------------------------------------------------------------- #
def placebo_pvalue(
    ret: pd.DataFrame,
    window: int = 42,
    tail_frac: float = 0.2,
    min_names: int = 10,
    n_seeds: int = 20,
    n_draws_per_seed: int = 50,
    base_seed: int = 871,
) -> dict:
    """Keep the rank-extremity sort but read each day's forward return from a
    **column-permuted** panel (signal->outcome link broken, each day's cross-sectional
    distribution preserved). p = share of permuted worlds whose spread mean is >=
    observed (right-tail test on the long-middle / short-extremes spread)."""
    cols = list(ret.columns)
    ncol = len(cols)
    sig = trailing_return(ret, window).shift(1)
    obs = float(extremity_spreads(ret, window, tail_frac, min_names)["spread"].mean())

    ret_mat = ret.to_numpy(dtype=float)
    pos_of = {c: i for i, c in enumerate(cols)}
    rows_idx, mids, exts = [], [], []
    row_lookup = {t: r for r, t in enumerate(ret.index)}
    for t in ret.index:
        s = sig.loc[t].dropna()
        n = len(s)
        if n < min_names:
            continue
        k = max(1, int(np.floor(n * tail_frac)))
        if 2 * k >= n:
            continue
        order = s.sort_values()
        names = list(order.index)
        extremes = names[:k] + names[-k:]
        c = n // 2
        mid = names[max(0, c - k):c - k + 2 * k]
        rows_idx.append(row_lookup[t])
        mids.append(np.array([pos_of[x] for x in mid]))
        exts.append(np.array([pos_of[x] for x in extremes]))
    rows_idx = np.asarray(rows_idx)

    means = []
    if len(rows_idx):
        M = ret_mat[rows_idx]
        km = max(len(a) for a in mids)
        ke = max(len(a) for a in exts)

        def _pad(books, kmax):
            P = np.zeros((len(books), kmax), dtype=int)
            V = np.zeros((len(books), kmax), dtype=bool)
            for j, a in enumerate(books):
                P[j, :len(a)] = a
                V[j, :len(a)] = True
            return P, V

        MID, MIDv = _pad(mids, km)
        EXT, EXTv = _pad(exts, ke)
        rows_ar = np.arange(len(rows_idx))[:, None]

        def _masked_mean(pos, valid, perm):
            vals = M[rows_ar, perm[pos]]
            vals = np.where(valid, vals, np.nan)
            return np.nanmean(vals, axis=1)

        for seed in range(n_seeds):
            rng = np.random.default_rng(base_seed + seed)
            for _ in range(n_draws_per_seed):
                perm = rng.permutation(ncol)
                mid_v = _masked_mean(MID, MIDv, perm)
                ext_v = _masked_mean(EXT, EXTv, perm)
                means.append(np.nanmean(mid_v - ext_v))
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
    """Cost the long-middle / short-extremes book.

    The signal is a trailing-window rank that turns over as names drift across the
    tail / middle boundary; to stay comparable to the desk's other cross-sectional
    timers we charge 2 sides x one-way cost x NAV per day on the long-short book, plus
    borrow on the short (extremes) leg.
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
def synthetic_detect(panel: dict[str, pd.DataFrame], window: int = 42,
                     tail_frac: float = 0.2) -> dict:
    """Run the headline rank-extremity stats on a synthetic panel."""
    ret = close_returns(panel)
    sp = extremity_spreads(ret, window, tail_frac)
    ts = rank_stats(sp)
    lc = lc_stats(level_controlled_spreads(ret, window, tail_frac))
    return {"spread_bps": ts["spread_bps"], "t_nw": ts["t_nw"],
            "welch_t": ts["welch_t"], "n_days": ts["n_days"],
            "lc_spread_bps": lc["spread_bps"], "lc_t_nw": lc["t_nw"]}
