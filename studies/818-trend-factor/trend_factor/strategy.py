"""Strategy + inference for Study 818 — Trend Factor.

The claim (Han, Zhou & Zhu 2016): a stock's expected return is better forecast by a
**blend** of moving-average signals across *all* horizons than by any single moving-average
timing rule or by momentum alone. The construction:

* **Normalized MA signals.** For each name and each horizon ``L`` in
  {3, 5, 10, 20, 50, 100, 200} days, ``A_L = MA_L(price) / price`` — the L-day moving
  average divided by today's price (a number near 1; ``> 1`` when price sits *below* its
  moving average, ``< 1`` in a sustained up-trend).
* **Rolling Fama-MacBeth-lite slopes.** Each day, regress the realized next-day cross-
  section of returns on the ``A_L`` vector, giving a time series of predictive slope
  vectors ``beta_t`` (one per horizon). The **expected** slope ``E_t[beta]`` is the rolling
  average of the *past* slopes (known through the close of ``t``).
* **The trend factor.** The fitted expected return of each name is
  ``trend_j,t = A_j,t · E_t[beta]`` — the data-weighted blend of its horizon signals.
* **Point-in-time sort.** On each day ``t`` rank the cross-section by the trend factor
  **known at the close of ``t-1``** (one ``shift``) and hold day ``t``. Long the top
  ``frac`` (high trend), short the bottom ``frac`` (low trend); equal weight.

This is contrasted (see :func:`single_ma_spreads`, :func:`momentum_spreads`) with the two
things the paper says the trend factor beats:

* **Single-MA timing** — the Faber-style rule of sorting on *one* ``A_L`` (cf.
  [110-faber-timing](../../110-faber-timing/), [438-triple-ma-crossover](../../438-triple-ma-crossover/));
* **Cross-sectional momentum** — the 12-1 trailing return sort (cf.
  [507-momentum](../../507-momentum/), [518-tsmom](../../518-tsmom/)).

Inference: a Newey-West (HAC) *t* on the daily long-short spread; a one-sample *t* and a
pooled Welch *t* (high-trend book vs low-trend book) cross-check; a permutation placebo
breaks the signal->outcome link; a costed timer charges the round-trip friction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252
MA_LAGS = (3, 5, 10, 20, 50, 100, 200)


# --------------------------------------------------------------------------- #
# Return + price panels
# --------------------------------------------------------------------------- #
def close_prices(panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Adjusted close price panel (index=date, columns=ticker)."""
    return pd.DataFrame({s: panel[s]["Close"] for s in panel}).sort_index()


def close_returns(panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Daily simple close-to-close returns (index=date, columns=ticker)."""
    return close_prices(panel).pct_change()


# --------------------------------------------------------------------------- #
# The trend-factor signal
# --------------------------------------------------------------------------- #
def ma_signals(prices: pd.DataFrame, lags=MA_LAGS) -> np.ndarray:
    """Normalized moving-average signals ``A_L = MA_L / price``.

    Returns a float array of shape ``(T, N, K)`` (dates × names × horizons). ``A[t]`` is
    known at the close of day ``t``. Rows without a full ``max(lags)`` window are NaN.
    """
    P = prices.to_numpy(dtype=float)
    T, N = P.shape
    K = len(lags)
    A = np.full((T, N, K), np.nan, dtype=float)
    for k, L in enumerate(lags):
        ma = prices.rolling(L, min_periods=L).mean().to_numpy(dtype=float)
        A[:, :, k] = ma / P
    return A


def _xs_betas(A: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Rolling cross-sectional predictive slopes (the Fama-MacBeth-lite step).

    ``beta[t]`` is the OLS slope vector (length ``K``) of the realized day-``t`` return
    cross-section ``R[t]`` on the signal vector ``A[t-1]`` known at the close of ``t-1``
    (with an intercept). So ``beta[t]`` is known at the close of ``t``. Vectorised per day
    via ``np.linalg.lstsq`` on the (names × horizons) design — a handful of tiny solves.
    """
    T, N, K = A.shape
    betas = np.full((T, K), np.nan, dtype=float)
    for t in range(1, T):
        x = A[t - 1]           # (N, K) — signal known at close t-1
        y = R[t]               # (N,)   — realized return over day t
        ok = np.all(np.isfinite(x), axis=1) & np.isfinite(y)
        if ok.sum() < K + 2:
            continue
        Xd = np.column_stack([np.ones(ok.sum()), x[ok]])   # intercept + K
        coef, *_ = np.linalg.lstsq(Xd, y[ok], rcond=None)
        betas[t] = coef[1:]    # drop intercept -> K slopes
    return betas


def trend_factor(prices: pd.DataFrame, ret: pd.DataFrame,
                 lags=MA_LAGS, beta_window: int = 250) -> pd.DataFrame:
    """The fitted expected return (trend factor) per name, ``A_j,t · E_t[beta]``.

    ``E_t[beta]`` is the rolling ``beta_window``-day average of the past cross-sectional
    slopes (through the close of ``t``); dotting it into the horizon signals ``A[t]`` (also
    known at close ``t``) gives ``trend[t]``, the model's forecast of the day-``t+1`` return.
    The value on row ``t`` is therefore known at the close of ``t``; :func:`trend_spreads`
    shifts by one day so a day-``t`` position is formed on information known at ``t-1``.
    """
    A = ma_signals(prices, lags)
    R = ret.to_numpy(dtype=float)
    betas = _xs_betas(A, R)
    ebeta = (
        pd.DataFrame(betas, index=prices.index)
        .rolling(beta_window, min_periods=beta_window).mean()
        .to_numpy(dtype=float)
    )                                           # (T, K), E_t[beta] known at close t
    trend = np.einsum("tnk,tk->tn", A, ebeta)   # (T, N)
    return pd.DataFrame(trend, index=prices.index, columns=prices.columns)


# --------------------------------------------------------------------------- #
# Contrast signals — single-MA timing and momentum
# --------------------------------------------------------------------------- #
def single_ma_signal(prices: pd.DataFrame, lag: int = 200) -> pd.DataFrame:
    """One horizon's timing signal, sorted *high price-vs-MA* = up-trend.

    Faber/110-style: ``price / MA_L - 1`` (positive when price is above its L-day MA).
    Higher = stronger single-horizon up-trend. Known at the close of ``t``.
    """
    ma = prices.rolling(lag, min_periods=lag).mean()
    return prices / ma - 1.0


def momentum_signal(prices: pd.DataFrame, lookback: int = 252, skip: int = 21) -> pd.DataFrame:
    """Cross-sectional 12-1 momentum: trailing ``lookback`` return skipping the last
    ``skip`` days. Known at the close of ``t``."""
    return prices.shift(skip) / prices.shift(lookback) - 1.0


# --------------------------------------------------------------------------- #
# The cross-sectional sort -> long-high / short-low spread
# --------------------------------------------------------------------------- #
def _fractile_spreads(sig: pd.DataFrame, ret: pd.DataFrame,
                      frac: float, min_names: int) -> pd.DataFrame:
    """Daily equal-weight top-minus-bottom fractile spread on a signal ``sig``.

    ``sig`` carries the value known at the close of ``t`` on row ``t``; it is shifted one
    day here so a day-``t`` position is formed on information known at ``t-1``. ``hi`` =
    mean forward day-``t`` return of the top ``frac`` (long), ``lo`` = mean of the bottom
    ``frac`` (short). ``spread = hi - lo``. Days with fewer than ``min_names`` ranked names
    are dropped.
    """
    S = sig.shift(1).to_numpy(dtype=float)      # known at close t-1
    R = ret.to_numpy(dtype=float)
    idx = ret.index
    out_spread, out_lo, out_hi, out_n, out_t = [], [], [], [], []
    for i in range(len(idx)):
        row = S[i]
        valid = np.where(np.isfinite(row))[0]
        n = len(valid)
        if n < min_names:
            continue
        k = max(1, int(np.floor(n * frac)))
        order = valid[np.argsort(row[valid], kind="stable")]
        low = order[:k]         # low trend  -> short
        high = order[-k:]       # high trend -> long
        rr = R[i]
        lo = float(np.nanmean(rr[low]))
        hi = float(np.nanmean(rr[high]))
        out_spread.append(hi - lo); out_lo.append(lo); out_hi.append(hi)
        out_n.append(n); out_t.append(idx[i])
    return pd.DataFrame(
        {"spread": out_spread, "lo": out_lo, "hi": out_hi, "n": out_n}, index=out_t
    ).sort_index()


def trend_spreads(prices: pd.DataFrame, ret: pd.DataFrame, lags=MA_LAGS,
                  beta_window: int = 250, frac: float = 0.3,
                  min_names: int = 10) -> pd.DataFrame:
    """Daily equal-weight long-high-trend / short-low-trend spread on the trend factor."""
    sig = trend_factor(prices, ret, lags, beta_window)
    return _fractile_spreads(sig, ret, frac, min_names)


def single_ma_spreads(prices: pd.DataFrame, ret: pd.DataFrame, lag: int = 200,
                       frac: float = 0.3, min_names: int = 10) -> pd.DataFrame:
    """Contrast: the long-short spread from sorting on ONE moving-average horizon."""
    return _fractile_spreads(single_ma_signal(prices, lag), ret, frac, min_names)


def momentum_spreads(prices: pd.DataFrame, ret: pd.DataFrame, lookback: int = 252,
                     skip: int = 21, frac: float = 0.3, min_names: int = 10) -> pd.DataFrame:
    """Contrast: the long-short spread from a 12-1 cross-sectional momentum sort."""
    return _fractile_spreads(momentum_signal(prices, lookback, skip), ret, frac, min_names)


# --------------------------------------------------------------------------- #
# Inference primitives  (copied from study 803)
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
def trend_stats(spreads: pd.DataFrame, nw_lags: int = 10) -> dict:
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
def placebo_pvalue(prices: pd.DataFrame, ret: pd.DataFrame, lags=MA_LAGS,
                   beta_window: int = 250, frac: float = 0.3, min_names: int = 10,
                   n_seeds: int = 20, n_draws_per_seed: int = 50,
                   base_seed: int = 818) -> dict:
    """Keep the trend-factor sort but read each day's forward return from a
    **column-permuted** panel (signal->outcome link broken, each day's cross-sectional
    distribution preserved). p = share of permuted worlds whose spread mean is >= observed
    (right-tail test on the long-high/short-low spread)."""
    cols = list(ret.columns)
    ncol = len(cols)
    sig = trend_factor(prices, ret, lags, beta_window).shift(1)
    obs = float(trend_spreads(prices, ret, lags, beta_window, frac, min_names)["spread"].mean())

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
def timer_stats(spreads: pd.DataFrame, cost_bps: float = 5.0,
                borrow_bps_yr: float = 50.0) -> dict:
    """Cost the long-high-trend / short-low-trend book.

    The trend factor is a slow, blended signal that turns over gradually, but names drift
    across the fractile boundary daily; to stay comparable to the desk's other cross-
    sectional timers we charge 2 sides × one-way cost × NAV per day on the long-short book,
    plus borrow on the short leg.
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
def synthetic_detect(panel: dict[str, pd.DataFrame], lags=MA_LAGS,
                     beta_window: int = 120, frac: float = 0.3) -> dict:
    """Run the headline trend stats on a synthetic panel."""
    prices = close_prices(panel)
    ret = close_returns(panel)
    sp = trend_spreads(prices, ret, lags, beta_window, frac)
    ts = trend_stats(sp)
    return {"spread_bps": ts["spread_bps"], "t_nw": ts["t_nw"],
            "welch_t": ts["welch_t"], "n_days": ts["n_days"]}
