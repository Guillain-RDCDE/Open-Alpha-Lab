"""Strategy + inference for Study 806 — Prospect-Theory Value.

The claim (Barberis, Mukherjee & Wang 2016): compute, for each stock, the
**cumulative-prospect-theory (TK) value** of its recent return distribution — the
value a Tversky-Kahneman prospect-theory investor would place on holding the stock as
a one-period gamble. A right-skewed, lottery-like tape scores a **high** TK value;
prospect-theory investors overweight that good gamble and over-pay, so high-TK names
go on to earn **lower** returns. A long **low-TK** / short **high-TK** book should
therefore earn a positive spread.

This is distinct from:

* [365-lottery-max-effect](../../365-lottery-max-effect/) — the single **maximum daily
  return** (MAX), a one-number tail proxy, not the whole probability-weighted value of
  the return distribution;
* [327-disposition-overhang](../../327-disposition-overhang/) — the **capital-gains
  overhang** driving disposition-effect selling pressure, a reference-point *holding*
  story, not the prospect-theory value of the past *return distribution*;
* [503-expected-idiosyncratic-skewness](../../503-expected-idiosyncratic-skewness/) —
  the **ex-ante / modelled** idiosyncratic skewness (Boyer-Mitton-Vorkink), a single
  moment forecast, not the full TK functional (value function + probability weighting).

Method (the canonical Barberis-Mukherjee-Wang design):

* **Return distribution.** For each name, take the trailing ``win_days`` daily simple
  returns known at a month-end ``t`` as the empirical distribution of the "gamble".
* **The TK value.** Sort the outcomes, treat each as equally likely (``1/n``), apply
  the Tversky-Kahneman **value function** ``v(x)=x^0.88`` for gains and
  ``-2.25*(-x)^0.88`` for losses, and **rank-dependent probability weighting**
  ``w+``/``w-`` (the inverse-S that overweights both tails). ``TK = Σ v(outcome) ×
  decision_weight`` where the decision weights are successive differences of the
  weighted cumulative probabilities (from the bottom for losses, from the top for
  gains).
* **Point-in-time monthly sort.** On each month-end ``t`` rank the cross-section by the
  TK value known at the close of ``t`` and hold the **next** month (one documented
  execution lag). Long the bottom ``frac`` (low TK), short the top ``frac`` (high TK);
  equal weight.
* **Inference.** Newey-West (HAC) *t* on the monthly long-short spread; a one-sample
  *t* and a pooled Welch *t* (bottom book vs top book) cross-check; a permutation
  placebo breaks the signal->outcome link; a costed timer charges the friction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252
MONTHS_YR = 12

# Tversky-Kahneman (1992) prospect-theory parameters.
TK_ALPHA = 0.88     # value-function curvature (gains and losses)
TK_LAMBDA = 2.25    # loss aversion
TK_GAMMA = 0.61     # probability-weighting curvature, gains  (w+)
TK_DELTA = 0.69     # probability-weighting curvature, losses (w-)


# --------------------------------------------------------------------------- #
# Return panel + the TK value of a return distribution
# --------------------------------------------------------------------------- #
def close_returns(panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Daily simple close-to-close returns (index=date, columns=ticker)."""
    closes = pd.DataFrame({s: panel[s]["Close"] for s in panel}).sort_index()
    return closes.pct_change()


def _weight(p: np.ndarray, gamma: float) -> np.ndarray:
    """Tversky-Kahneman probability-weighting function evaluated at cumulative ``p``.

    ``w(p) = p^g / (p^g + (1-p)^g)^(1/g)`` — the inverse-S that overweights small
    probabilities (the tails). ``p`` is a cumulative probability in ``[0, 1]``.
    """
    p = np.clip(np.asarray(p, dtype=float), 0.0, 1.0)
    num = p ** gamma
    den = (p ** gamma + (1.0 - p) ** gamma) ** (1.0 / gamma)
    out = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
    return out


def tk_value(returns: np.ndarray, min_obs: int = 24) -> float:
    """Cumulative-prospect-theory (TK) value of an empirical return distribution.

    ``returns`` is a 1-D array of outcomes, each treated as equally likely (``1/n``).
    Sort ascending ``r_(1) <= ... <= r_(n)``. Losses occupy the low ranks, gains the
    high ranks. The **decision weight** on the outcome at ascending rank ``i`` is

    * a **loss** (``r_(i) < 0``): ``w-(i/n) - w-((i-1)/n)``   — cumulated from the
      worst outcome up (the left tail);
    * a **gain** (``r_(i) >= 0``): ``w+((n-i+1)/n) - w+((n-i)/n)``  — cumulated from
      the best outcome down (the right tail).

    The value function is ``v(x)=x^0.88`` for ``x>=0`` and ``-2.25*(-x)^0.88`` else.
    ``TK = Σ_i v(r_(i)) × weight_i``. A right-skewed / lottery-like distribution puts
    mass in the overweighted right tail and scores **high**.
    """
    x = np.asarray(returns, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < min_obs:
        return float("nan")
    xs = np.sort(x)                      # ascending: losses first, gains last
    i = np.arange(1, n + 1)              # ascending rank
    # loss decision weights (cumulated from the bottom tail)
    loss_w = _weight(i / n, TK_DELTA) - _weight((i - 1) / n, TK_DELTA)
    # gain decision weights (cumulated from the top tail): rank-from-top = n-i+1
    j = n - i + 1
    gain_w = _weight(j / n, TK_GAMMA) - _weight((j - 1) / n, TK_GAMMA)
    is_gain = xs >= 0.0
    w = np.where(is_gain, gain_w, loss_w)
    v = np.where(is_gain, np.abs(xs) ** TK_ALPHA, -TK_LAMBDA * np.abs(xs) ** TK_ALPHA)
    return float(np.sum(w * v))


# --------------------------------------------------------------------------- #
# Monthly rebalance scaffolding + the cross-sectional TK sort
# --------------------------------------------------------------------------- #
def _month_end_positions(index: pd.DatetimeIndex) -> np.ndarray:
    """Integer positions of the last trading row in each calendar month."""
    keys = index.year * 12 + index.month
    last = np.where(keys[1:] != keys[:-1])[0]     # position before a month change
    return np.append(last, len(index) - 1)


def tk_spreads(
    ret: pd.DataFrame,
    win_days: int = 1260,
    frac: float = 0.3,
    min_names: int = 10,
    min_obs: int = 24,
) -> pd.DataFrame:
    """Monthly equal-weight bottom-minus-top TK-value fractile spread.

    At each month-end ``t`` each name's TK value is computed on its trailing
    ``win_days`` daily returns (the empirical gamble). Names are ranked; ``lo`` = mean
    **next-month** return of the bottom ``frac`` (low TK, the long); ``hi`` = mean of
    the top ``frac`` (high TK, the short). ``spread = lo - hi`` (long low-TK, short
    high-TK). The signal is known at the close of month ``t`` and the book is held
    over month ``t+1`` — one documented execution lag, zero look-ahead. Months with
    fewer than ``min_names`` valued names are dropped. A name is eligible only once a
    **full** ``win_days`` trailing window is available (no truncated early windows), so
    the TK value is always measured over the same-length distribution.
    """
    R = ret.to_numpy(dtype=float)
    idx = ret.index
    me = _month_end_positions(idx)

    # TK value per name at each month-end (uses the full trailing window through it).
    tk = np.full((len(me), R.shape[1]), np.nan)
    for m, pos in enumerate(me):
        if pos < win_days:                       # require a full trailing window
            continue
        window = R[pos - win_days + 1:pos + 1]
        for j in range(R.shape[1]):
            tk[m, j] = tk_value(window[:, j], min_obs=min_obs)

    # Forward (next-month) simple return per name, from month-end m to month-end m+1.
    close = (1.0 + np.nan_to_num(R)).cumprod(axis=0)
    me_close = close[me]
    fwd = np.full_like(tk, np.nan)
    fwd[:-1] = me_close[1:] / me_close[:-1] - 1.0     # return over the held month

    out_spread, out_lo, out_hi, out_n, out_t = [], [], [], [], []
    for m in range(len(me) - 1):                       # last month has no forward
        sig = tk[m]
        rr = fwd[m]
        valid = np.where(~np.isnan(sig) & ~np.isnan(rr))[0]
        n = len(valid)
        if n < min_names:
            continue
        k = max(1, int(np.floor(n * frac)))
        order = valid[np.argsort(sig[valid], kind="stable")]
        low = order[:k]        # low TK  -> long
        high = order[-k:]      # high TK -> short
        lo = float(np.mean(rr[low]))
        hi = float(np.mean(rr[high]))
        out_spread.append(lo - hi); out_lo.append(lo); out_hi.append(hi)
        out_n.append(n); out_t.append(idx[me[m]])
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
def tk_stats(spreads: pd.DataFrame, nw_lags: int = 6) -> dict:
    sp = spreads["spread"].to_numpy(dtype=float)
    return {
        "n_months": int(len(spreads)),
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
    win_days: int = 1260,
    frac: float = 0.3,
    min_names: int = 10,
    min_obs: int = 24,
    n_seeds: int = 20,
    n_draws_per_seed: int = 50,
    base_seed: int = 806,
) -> dict:
    """Keep the TK sort but read each month's forward return from a **column-permuted**
    panel (signal->outcome link broken, each month's cross-sectional distribution
    preserved). p = share of permuted worlds whose spread mean is >= observed
    (right-tail test on the long-low/short-high spread)."""
    R = ret.to_numpy(dtype=float)
    idx = ret.index
    me = _month_end_positions(idx)
    ncol = R.shape[1]

    tk = np.full((len(me), ncol), np.nan)
    for m, pos in enumerate(me):
        if pos < win_days:
            continue
        window = R[pos - win_days + 1:pos + 1]
        for j in range(ncol):
            tk[m, j] = tk_value(window[:, j], min_obs=min_obs)
    close = (1.0 + np.nan_to_num(R)).cumprod(axis=0)
    me_close = close[me]
    fwd = np.full_like(tk, np.nan)
    fwd[:-1] = me_close[1:] / me_close[:-1] - 1.0

    obs = float(tk_spreads(ret, win_days, frac, min_names, min_obs)["spread"].mean())

    lows, highs, rows = [], [], []
    for m in range(len(me) - 1):
        sig = tk[m]
        rr = fwd[m]
        valid = np.where(~np.isnan(sig) & ~np.isnan(rr))[0]
        if len(valid) < min_names:
            continue
        k = max(1, int(np.floor(len(valid) * frac)))
        order = valid[np.argsort(sig[valid], kind="stable")]
        lows.append(order[:k]); highs.append(order[-k:]); rows.append(m)
    rows = np.asarray(rows)

    means = []
    if len(rows):
        F = fwd[rows]                      # forward-return rows aligned to sort months
        kl = max(len(a) for a in lows)
        kh = max(len(a) for a in highs)

        def _pad(books, kmax):
            P = np.zeros((len(books), kmax), dtype=int)
            V = np.zeros((len(books), kmax), dtype=bool)
            for jx, a in enumerate(books):
                P[jx, :len(a)] = a
                V[jx, :len(a)] = True
            return P, V

        LOW, LOWv = _pad(lows, kl)
        HIGH, HIGHv = _pad(highs, kh)
        rows_ar = np.arange(len(rows))[:, None]

        def _masked_mean(pos, valid, perm):
            vals = F[rows_ar, perm[pos]]
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
    """Cost the long-low-TK / short-high-TK book.

    The TK signal turns over roughly monthly (the book is re-formed every month), so we
    charge, per monthly rebalance, 2 sides × one-way cost × NAV on the long-short book,
    plus one month of borrow on the short leg. Comparable to the desk's other
    cross-sectional timers.
    """
    sp = spreads["spread"].to_numpy(dtype=float)
    sp = sp[~np.isnan(sp)]
    n = len(sp)
    round_trip_cost = 2.0 * cost_bps / 1e4
    borrow_monthly = (borrow_bps_yr / 1e4) / MONTHS_YR
    net = sp - round_trip_cost - borrow_monthly
    gross_mean = float(sp.mean())
    net_mean = float(net.mean())
    sd = float(net.std(ddof=1)) if n > 1 else float("nan")
    sharpe = net_mean / sd * np.sqrt(MONTHS_YR) if sd and sd > 0 else float("nan")
    return {
        "n_months": n,
        "gross_bps": gross_mean * 1e4,
        "net_bps": net_mean * 1e4,
        "cost_bps_per_reb": (round_trip_cost + borrow_monthly) * 1e4,
        "ann_net_pct": net_mean * MONTHS_YR * 100,
        "sharpe_net": sharpe,
        "t_net": one_sample_t(net),
    }


# --------------------------------------------------------------------------- #
# Synthetic-control detector (the machinery proof)
# --------------------------------------------------------------------------- #
def synthetic_detect(panel: dict[str, pd.DataFrame], win_days: int = 252,
                     frac: float = 0.3, min_obs: int = 24) -> dict:
    """Run the headline TK stats on a synthetic panel (short window for speed)."""
    ret = close_returns(panel)
    sp = tk_spreads(ret, win_days=win_days, frac=frac, min_obs=min_obs)
    ts = tk_stats(sp)
    return {"spread_bps": ts["spread_bps"], "t_nw": ts["t_nw"],
            "welch_t": ts["welch_t"], "n_months": ts["n_months"]}
