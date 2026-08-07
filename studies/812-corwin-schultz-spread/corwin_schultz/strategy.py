"""Strategy + inference for Study 812 — Corwin-Schultz Spread.

The claim (Corwin & Schultz 2012): a stock's effective bid-ask **spread** can be
estimated from its daily **high** and **low** prices alone; that estimated spread proxies
illiquidity, so a long **high-spread** (illiquid) / short **low-spread** (liquid) book
should harvest an **illiquidity premium** — a positive cross-sectional spread.

This is distinct from:

* [377-bid-ask-bounce](../../377-bid-ask-bounce/) — the short-horizon **mean-reversion**
  induced by the spread's own bounce (a return-autocorrelation timing signal), not a
  cross-sectional *level* of illiquidity used to sort names.
* [140-amihud-illiquidity](../../140-amihud-illiquidity/) — Amihud's
  |return| / dollar-volume **price-impact** ratio, a *volume*-based illiquidity proxy.
  Corwin-Schultz needs no volume at all — it reads the spread straight off the high-low
  range.
* [811-zero-return-days](../../811-zero-return-days/) — the *count of zero-return days*
  (Lesmond-Ogden-Trzcinka) as an illiquidity proxy, a frequency measure, not a
  high-low range estimator of the spread.

The Corwin-Schultz estimator. Over two consecutive days ``t-1, t`` with highs ``H`` and
lows ``L``:

    beta  = [ln(H_{t-1}/L_{t-1})]^2 + [ln(H_t/L_t)]^2                 (sum of 2 single-day ranges)
    gamma = [ln(max(H_{t-1},H_t) / min(L_{t-1},L_t))]^2               (the 2-day range)
    alpha = (sqrt(2*beta) - sqrt(beta)) / (3 - 2*sqrt(2)) - sqrt(gamma / (3 - 2*sqrt(2)))
    S     = 2 * (exp(alpha) - 1) / (1 + exp(alpha))                   (proportional spread)

Negative daily ``S`` (volatility swamps the spread that day) is floored at 0. The name's
**estimated spread** is the trailing-``window``-day average of daily ``S``.

Method:

* **Daily CS spread, then a trailing month.** Compute daily ``S`` per name (vectorised
  over the whole panel), floor negatives, average over a trailing ``window`` (≈ a month).
* **Point-in-time sort.** On each day ``t`` rank the cross-section by the trailing spread
  known at the close of ``t-1`` (one ``shift``) and hold day ``t``. Long the top ``frac``
  (high spread, illiquid), short the bottom ``frac`` (low spread, liquid); equal weight.
* **Inference.** Newey-West (HAC) *t* on the daily long-short spread; a one-sample *t*
  and a pooled Welch *t* (high-spread book vs low-spread book) cross-check; a permutation
  placebo breaks the signal->outcome link; a costed timer charges the round-trip friction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252
_K = 3.0 - 2.0 * np.sqrt(2.0)   # the Corwin-Schultz constant 3 - 2*sqrt(2)


# --------------------------------------------------------------------------- #
# Return panel + signal
# --------------------------------------------------------------------------- #
def close_returns(panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Daily simple close-to-close returns (index=date, columns=ticker)."""
    closes = pd.DataFrame({s: panel[s]["Close"] for s in panel}).sort_index()
    return closes.pct_change()


def daily_cs_spread(panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Daily Corwin-Schultz high-low spread estimate ``S`` per name (negatives floored).

    Fully vectorised over the whole panel. Row ``t`` uses days ``t-1`` and ``t``; the
    first row of each name is NaN. ``S`` is a *proportional* spread (e.g. 0.004 = 40 bps).
    """
    highs = pd.DataFrame({s: panel[s]["High"] for s in panel}).sort_index()
    lows = pd.DataFrame({s: panel[s]["Low"] for s in panel}).sort_index()

    H = highs.to_numpy(dtype=float)
    L = lows.to_numpy(dtype=float)
    Hp = np.roll(H, 1, axis=0); Hp[0] = np.nan   # previous day's high
    Lp = np.roll(L, 1, axis=0); Lp[0] = np.nan   # previous day's low

    with np.errstate(invalid="ignore", divide="ignore"):
        hl_t = np.log(H / L) ** 2
        hl_p = np.log(Hp / Lp) ** 2
        beta = hl_t + hl_p
        two_hi = np.maximum(H, Hp)
        two_lo = np.minimum(L, Lp)
        gamma = np.log(two_hi / two_lo) ** 2
        alpha = (np.sqrt(2.0 * beta) - np.sqrt(beta)) / _K - np.sqrt(gamma / _K)
        S = 2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha))

    S = np.where(np.isfinite(S), S, np.nan)
    S = np.where(S < 0.0, 0.0, S)                # Corwin-Schultz: floor negatives at 0
    return pd.DataFrame(S, index=highs.index, columns=highs.columns)


def trailing_spread(panel: dict[str, pd.DataFrame], window: int = 21) -> pd.DataFrame:
    """Trailing-``window``-day mean of the daily Corwin-Schultz spread, per name.

    The row-``t`` value uses daily ``S`` through ``t`` (inclusive); the sort in
    :func:`cs_spreads` shifts by one day so a day-``t`` position is formed on information
    known at ``t-1``.
    """
    S = daily_cs_spread(panel)
    return S.rolling(window, min_periods=window).mean()


# --------------------------------------------------------------------------- #
# The cross-sectional sort -> long-high-spread / short-low-spread
# --------------------------------------------------------------------------- #
def cs_spreads(
    panel: dict[str, pd.DataFrame],
    window: int = 21,
    frac: float = 0.3,
    min_names: int = 10,
) -> pd.DataFrame:
    """Daily equal-weight top-minus-bottom estimated-spread fractile spread.

    On each day ``t`` names are ranked by the trailing CS spread known at the close of
    ``t-1`` (one ``shift``). ``long_ret`` = mean forward day-``t`` return of the top
    ``frac`` (high spread, illiquid, the long); ``short_ret`` = mean of the bottom
    ``frac`` (low spread, liquid, the short). ``spread = long_ret - short_ret`` (long
    high-spread, short low-spread). Days with fewer than ``min_names`` ranked names drop.
    """
    ret = close_returns(panel)
    sig = trailing_spread(panel, window).reindex_like(ret).shift(1)  # known at close t-1
    Sg = sig.to_numpy(dtype=float)
    R = ret.to_numpy(dtype=float)
    idx = ret.index
    out_spread, out_long, out_short, out_n, out_t = [], [], [], [], []
    for i in range(len(idx)):
        row = Sg[i]
        valid = np.where(~np.isnan(row))[0]
        n = len(valid)
        if n < min_names:
            continue
        k = max(1, int(np.floor(n * frac)))
        order = valid[np.argsort(row[valid], kind="stable")]
        low = order[:k]        # low spread  -> short
        high = order[-k:]      # high spread -> long
        rr = R[i]
        lg = float(np.nanmean(rr[high]))
        sh = float(np.nanmean(rr[low]))
        out_spread.append(lg - sh); out_long.append(lg); out_short.append(sh)
        out_n.append(n); out_t.append(idx[i])
    return pd.DataFrame(
        {"spread": out_spread, "long_ret": out_long, "short_ret": out_short,
         "n": out_n}, index=out_t
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
def cs_stats(spreads: pd.DataFrame, nw_lags: int = 10) -> dict:
    sp = spreads["spread"].to_numpy(dtype=float)
    return {
        "n_days": int(len(spreads)),
        "spread_bps": float(np.nanmean(sp) * 1e4),
        "t_nw": newey_west_t(sp, nw_lags),
        "t_1s": one_sample_t(sp),
        "long_bps": float(np.nanmean(spreads["long_ret"].to_numpy()) * 1e4),
        "short_bps": float(np.nanmean(spreads["short_ret"].to_numpy()) * 1e4),
        "welch_t": welch_t(spreads["long_ret"].to_numpy(), spreads["short_ret"].to_numpy()),
    }


# --------------------------------------------------------------------------- #
# Placebo — is the spread real, or a lucky alignment of the sort?
# --------------------------------------------------------------------------- #
def placebo_pvalue(
    panel: dict[str, pd.DataFrame],
    window: int = 21,
    frac: float = 0.3,
    min_names: int = 10,
    n_seeds: int = 20,
    n_draws_per_seed: int = 50,
    base_seed: int = 812,
) -> dict:
    """Keep the trailing-spread sort but read each day's forward return from a
    **column-permuted** panel (signal->outcome link broken, each day's cross-sectional
    distribution preserved). p = share of permuted worlds whose spread mean is >= observed
    (right-tail test on the long-high/short-low spread)."""
    ret = close_returns(panel)
    cols = list(ret.columns)
    ncol = len(cols)
    sig = trailing_spread(panel, window).reindex_like(ret).shift(1)
    obs = float(cs_spreads(panel, window, frac, min_names)["spread"].mean())

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
        lows.append(np.array([pos_of[c] for c in order.index[:k]]))    # low spread -> short
        highs.append(np.array([pos_of[c] for c in order.index[-k:]]))  # high spread -> long
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
                means.append(np.nanmean(hi_v - lo_v))   # long high-spread minus short low-spread
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
    """Cost the long-high-spread / short-low-spread book.

    The signal is a trailing-window spread that turns over roughly monthly, but names
    drift across the fractile boundary daily; to stay comparable to the desk's other
    cross-sectional timers we charge 2 sides x one-way cost x NAV per day on the
    long-short book, plus borrow on the short leg. (Ironically the illiquid long leg is
    exactly where real-world spread costs would bite hardest — this flat charge is
    generous to the strategy.)
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
def synthetic_detect(panel: dict[str, pd.DataFrame], window: int = 21,
                     frac: float = 0.3) -> dict:
    """Run the headline CS spread stats on a synthetic panel."""
    sp = cs_spreads(panel, window, frac)
    ts = cs_stats(sp)
    return {"spread_bps": ts["spread_bps"], "t_nw": ts["t_nw"],
            "welch_t": ts["welch_t"], "n_days": ts["n_days"]}
