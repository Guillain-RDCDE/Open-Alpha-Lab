"""The engine and its honest controls — Study 836 (Rebalance Timing Luck).

The claim, at full strength (Hoffstein, Sober & Vezeris 2019): take **one** monthly
cross-sectional momentum book and rebalance it on a **different day of the month** each
time you run it. Nothing about the rule changes — same lookback, same universe, same
fractile — yet each *rebalance offset* (day 0, day 1, ... day 20 of the period) traces a
**materially different equity curve** with a **materially different Sharpe**. That
spread is **rebalance timing luck**: a phantom dispersion driven entirely by the
arbitrary choice of *when* to rebalance, not by any difference in skill or signal.

The honest diagnosis and the fix:

* **The dispersion is luck, not skill.** If the offset-to-offset Sharpe gap were real
  information, the *lucky* offset would stay lucky. It does not: split the sample and
  the rank of offsets by Sharpe in the first half is ~uncorrelated with the second half
  — the winner is unforecastable, the signature of pure luck.
* **The fix is tranching / overlapping portfolios.** Instead of betting the whole book
  on one arbitrary rebalance day, run ``period`` sub-portfolios each rebalanced on a
  different offset with ``1/period`` of the capital (equivalently: rebalance a slice
  every day). Their average is a **single** overlapping portfolio — there is nothing
  left to be lucky *about*, so the phantom dispersion collapses to zero while the
  strategy's genuine content (if any) is preserved.

Inference primitives (``one_sample_t`` / ``welch_t`` / ``newey_west_t`` /
``wilson_interval`` / ``spearman_rank_corr``) and a costed ``timer`` grade the tranched
book; a synthetic positive control proves the machinery detects a *planted* momentum
premium and stays silent on the null. All vectorised numpy — no per-row date loops.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

TRADING_DAYS = 252


# --------------------------------------------------------------------------- #
# Signal — trailing cross-sectional momentum
# --------------------------------------------------------------------------- #
def trailing_return(returns: np.ndarray, lookback: int) -> np.ndarray:
    """Trailing ``lookback``-day compounded return per name (vectorised via log-cumsum).

    Value on row ``t`` uses returns through ``t`` (inclusive) over the trailing window
    ``[t-lookback+1, t]``; rows before ``lookback`` are ``nan``. The sort in
    :func:`offset_portfolio` reads the signal at row ``d-1`` (one lag), so a position
    formed on rebalance day ``d`` uses only information known at the close of ``d-1``.
    """
    R = np.asarray(returns, dtype=float)
    lp = np.cumsum(np.log1p(R), axis=0)
    out = np.full_like(R, np.nan)
    out[lookback:] = lp[lookback:] - lp[:-lookback]
    return out


def _ls_weights(sig_row: np.ndarray, top_frac: float, min_names: int) -> np.ndarray:
    """Dollar-neutral long-short weights from one cross-sectional signal row.

    Long the top ``top_frac`` (highest momentum), short the bottom ``top_frac``, equal
    weight; the long leg sums to +1 and the short leg to −1 (gross 2, net 0). Returns an
    all-zero vector if fewer than ``min_names`` names carry a signal.
    """
    n_total = sig_row.shape[0]
    w = np.zeros(n_total)
    valid = np.where(np.isfinite(sig_row))[0]
    if valid.size < min_names:
        return w
    k = max(1, int(np.floor(valid.size * top_frac)))
    order = valid[np.argsort(sig_row[valid], kind="stable")]
    short = order[:k]     # lowest momentum -> short
    long = order[-k:]     # highest momentum -> long
    w[long] = 1.0 / len(long)
    w[short] = -1.0 / len(short)
    return w


# --------------------------------------------------------------------------- #
# One rebalance-offset portfolio
# --------------------------------------------------------------------------- #
def offset_portfolio(
    returns: np.ndarray,
    mom: np.ndarray,
    offset: int,
    period: int = 21,
    top_frac: float = 0.3,
    min_names: int = 6,
) -> tuple[np.ndarray, int]:
    """Daily returns of a momentum long-short rebalanced every ``period`` days starting
    at ``offset``.

    Rebalance days are ``offset, offset+period, offset+2*period, ...``. On each rebalance
    day ``d`` the book is set from the momentum signal known at the close of ``d-1`` and
    held **fixed** until the next rebalance. Returns ``(port, first_active)`` where
    ``port`` is the length-``T`` daily portfolio-return array (zero before the first
    filled rebalance) and ``first_active`` is the first day the book is live.
    """
    R = np.asarray(returns, dtype=float)
    T, N = R.shape
    W = np.zeros((T, N))
    reb_days = np.arange(offset, T, period)
    reb_days = reb_days[reb_days >= 1]  # need a signal at d-1
    first_active = T
    for j, d in enumerate(reb_days):
        w = _ls_weights(mom[d - 1], top_frac, min_names)
        end = reb_days[j + 1] if j + 1 < len(reb_days) else T
        W[d:end] = w
        if np.any(w) and first_active == T:
            first_active = d
    port = np.einsum("tn,tn->t", W, R)
    return port, first_active


# --------------------------------------------------------------------------- #
# Inference primitives (ported house style)
# --------------------------------------------------------------------------- #
def sharpe(x: np.ndarray, periods: int = TRADING_DAYS) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 3:
        return float("nan")
    sd = x.std(ddof=1)
    return float(x.mean() / sd * np.sqrt(periods)) if sd > 0 else float("nan")


def one_sample_t(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return float("nan")
    se = x.std(ddof=1) / np.sqrt(x.size)
    return float(x.mean() / se) if se > 0 else float("nan")


def welch_t(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if a.size < 2 or b.size < 2:
        return float("nan")
    se = np.sqrt(a.var(ddof=1) / a.size + b.var(ddof=1) / b.size)
    return float((a.mean() - b.mean()) / se) if se > 0 else float("nan")


def newey_west_t(x: np.ndarray, lags: int = 10) -> float:
    """HAC (Newey-West, Bartlett kernel) t of mean(x) vs 0."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
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


def spearman_rank_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return float("nan")
    return float(stats.spearmanr(a[m], b[m]).correlation)


# --------------------------------------------------------------------------- #
# The headline — Sharpe dispersion across every rebalance offset
# --------------------------------------------------------------------------- #
def timing_luck(
    returns: pd.DataFrame | np.ndarray,
    period: int = 21,
    lookback: int = 126,
    top_frac: float = 0.3,
    min_names: int = 6,
) -> dict:
    """Run the SAME momentum book on every rebalance offset ``0..period-1`` and quantify
    the phantom Sharpe dispersion.

    Every offset trades the identical rule on the identical data — the only difference is
    *which* day of the ``period``-day cycle it rebalances. Returns the per-offset Sharpe
    array plus the dispersion summary (spread = luckiest − unluckiest, sd) and the
    offset means. Evaluation starts once every offset is live (``lookback + period``) so
    all offsets are compared on a common window.
    """
    R = returns.to_numpy(dtype=float) if isinstance(returns, pd.DataFrame) else np.asarray(returns, float)
    mom = trailing_return(R, lookback)
    T = R.shape[0]
    start = lookback + period
    sharpes, means = [], []
    for off in range(period):
        port, _ = offset_portfolio(R, mom, off, period, top_frac, min_names)
        seg = port[start:]
        sharpes.append(sharpe(seg))
        means.append(float(np.mean(seg)))
    sharpes = np.asarray(sharpes)
    means = np.asarray(means)
    return {
        "period": period,
        "sharpes": sharpes,
        "means_bps": means * 1e4,
        "sharpe_best": float(np.nanmax(sharpes)),
        "sharpe_worst": float(np.nanmin(sharpes)),
        "sharpe_spread": float(np.nanmax(sharpes) - np.nanmin(sharpes)),
        "sharpe_sd": float(np.nanstd(sharpes, ddof=1)),
        "sharpe_mean": float(np.nanmean(sharpes)),
        "best_offset": int(np.nanargmax(sharpes)),
        "worst_offset": int(np.nanargmin(sharpes)),
        "n_days": int(T - start),
    }


# --------------------------------------------------------------------------- #
# The fix — tranched / overlapping portfolio
# --------------------------------------------------------------------------- #
def tranched_portfolio(
    returns: pd.DataFrame | np.ndarray,
    period: int = 21,
    lookback: int = 126,
    top_frac: float = 0.3,
    min_names: int = 6,
) -> dict:
    """The overlapping-portfolio fix: average all ``period`` offset books into one curve.

    Running ``period`` sub-portfolios each rebalanced on a different offset with
    ``1/period`` of the capital is equivalent to rebalancing a slice of the book every
    day. There is exactly **one** such combined portfolio, so the timing-luck dispersion
    is gone by construction. Returns the tranched daily-return series and its stats.
    """
    R = returns.to_numpy(dtype=float) if isinstance(returns, pd.DataFrame) else np.asarray(returns, float)
    mom = trailing_return(R, lookback)
    T = R.shape[0]
    start = lookback + period
    acc = np.zeros(T)
    for off in range(period):
        port, _ = offset_portfolio(R, mom, off, period, top_frac, min_names)
        acc += port
    tranched = acc / period
    seg = tranched[start:]
    return {
        "tranched": tranched,
        "seg": seg,
        "sharpe": sharpe(seg),
        "mean_bps": float(np.mean(seg) * 1e4),
        "t_nw": newey_west_t(seg, lags=10),
        "t_1s": one_sample_t(seg),
        "n_days": int(seg.size),
        "start": start,
    }


# --------------------------------------------------------------------------- #
# Is the lucky offset skill or luck? Out-of-sample persistence of the ranking
# --------------------------------------------------------------------------- #
def offset_persistence(
    returns: pd.DataFrame | np.ndarray,
    period: int = 21,
    lookback: int = 126,
    top_frac: float = 0.3,
    min_names: int = 6,
) -> dict:
    """Rank offsets by Sharpe in the first half of the sample and the second half; the
    rank correlation tells you whether the *lucky* offset stays lucky.

    A high positive correlation would mean the offset choice carries real, forecastable
    information (skill). A correlation near zero means the winner in-sample is a coin-flip
    out-of-sample — the dispersion is **pure luck**. Returns both half Sharpe arrays and
    their Spearman rank correlation.
    """
    R = returns.to_numpy(dtype=float) if isinstance(returns, pd.DataFrame) else np.asarray(returns, float)
    mom = trailing_return(R, lookback)
    T = R.shape[0]
    start = lookback + period
    mid = start + (T - start) // 2
    sh1, sh2 = [], []
    for off in range(period):
        port, _ = offset_portfolio(R, mom, off, period, top_frac, min_names)
        sh1.append(sharpe(port[start:mid]))
        sh2.append(sharpe(port[mid:]))
    sh1 = np.asarray(sh1); sh2 = np.asarray(sh2)
    return {
        "sharpe_h1": sh1,
        "sharpe_h2": sh2,
        "rank_corr": spearman_rank_corr(sh1, sh2),
    }


# --------------------------------------------------------------------------- #
# The costed timer on the tranched book
# --------------------------------------------------------------------------- #
def timer_stats(
    returns: pd.DataFrame | np.ndarray,
    period: int = 21,
    lookback: int = 126,
    top_frac: float = 0.3,
    min_names: int = 6,
    cost_bps: float = 5.0,
    borrow_bps_yr: float = 50.0,
) -> dict:
    """Cost the tranched long-short book.

    Each day the tranched book rebalances one slice (``1/period`` of a 2×-NAV long-short
    book), turning over roughly ``2 * top_frac`` of that slice's gross; we charge a
    conservative daily round-trip on the traded slice, plus borrow on the (half-NAV)
    short leg. Returns gross/net daily means, the net Sharpe and net *t*.
    """
    tr = tranched_portfolio(returns, period, lookback, top_frac, min_names)
    seg = tr["seg"]
    # traded fraction per day: one slice (1/period) of a gross-2 book fully rotated ->
    # 2/period of NAV turns over; round-trip = 2 sides.
    daily_turnover = 2.0 / period
    round_trip_cost = 2.0 * (cost_bps / 1e4) * daily_turnover
    borrow_daily = (borrow_bps_yr / 1e4) / 365.0  # short leg = 1x NAV
    net = seg - round_trip_cost - borrow_daily
    return {
        "gross_bps": float(np.mean(seg) * 1e4),
        "net_bps": float(np.mean(net) * 1e4),
        "cost_bps_per_day": (round_trip_cost + borrow_daily) * 1e4,
        "sharpe_gross": sharpe(seg),
        "sharpe_net": sharpe(net),
        "t_net": one_sample_t(net),
        "n_days": int(seg.size),
    }


# --------------------------------------------------------------------------- #
# Multi-seed robustness — the house >=20-seed rule for synthetic claims
# --------------------------------------------------------------------------- #
def seed_robust(
    data_mod,
    mom_edge: float,
    n_seeds: int = 25,
    base_seed: int = 836,
    period: int = 21,
    lookback: int = 126,
    top_frac: float = 0.3,
    min_names: int = 6,
    n_days: int = 2600,
) -> dict:
    """Average the key numbers over ``n_seeds`` synthetic worlds with a planted
    ``mom_edge``.

    Reports, per world and then averaged:

    - ``sharpe_spread`` — luckiest − unluckiest offset Sharpe (the phantom dispersion),
    - ``rank_corr`` — out-of-sample persistence of the offset ranking (≈ 0 = pure luck),
    - ``tranched_sharpe`` / ``tranched_t_nw`` — the dispersion-free book (should be ~0 on
      the null, robustly positive when a real momentum premium is planted).
    """
    spreads, corrs, tsh, ttnw, best_off = [], [], [], [], []
    for s in range(base_seed, base_seed + n_seeds):
        ret, _ = data_mod.synthetic_panel(mom_edge=mom_edge, seed=s, n_days=n_days)
        tl = timing_luck(ret, period, lookback, top_frac, min_names)
        pr = offset_persistence(ret, period, lookback, top_frac, min_names)
        tr = tranched_portfolio(ret, period, lookback, top_frac, min_names)
        spreads.append(tl["sharpe_spread"])
        corrs.append(pr["rank_corr"])
        tsh.append(tr["sharpe"])
        ttnw.append(tr["t_nw"])
        best_off.append(tl["best_offset"])
    tsh = np.asarray(tsh); ttnw = np.asarray(ttnw)
    return {
        "mom_edge": mom_edge,
        "n_seeds": n_seeds,
        "mean_sharpe_spread": float(np.mean(spreads)),
        "mean_rank_corr": float(np.mean(corrs)),
        "mean_tranched_sharpe": float(np.mean(tsh)),
        "mean_tranched_t_nw": float(np.mean(ttnw)),
        "tranched_t_fires": int(np.sum(np.abs(ttnw) >= 2.0)),
        "tranched_sharpe_arr": tsh,
        "best_offsets": np.asarray(best_off),
    }
