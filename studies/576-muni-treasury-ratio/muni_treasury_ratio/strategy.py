"""The engine and its honest controls — Study 576 (Muni-Treasury-Ratio).

The claim, at full strength (muni-market folklore; e.g. the "10Y AAA muni / 10Y Treasury ratio"
quoted daily by every muni desk): the muni-Treasury yield **ratio** is a rich/cheap gauge that
**times** muni or duration exposure. When the ratio is *high* (munis yield an unusually large
fraction of Treasuries) munis are *cheap* and should outperform; when it is *low*, munis are
*rich* and should lag. The tradable expression is a ratio-conditioned tilt: overweight munis (vs
comparable Treasuries) when the ratio is high, underweight when low.

This module measures that, honestly, on what a retail stack can reach — the MUB/IEF distribution-
yield ratio and their daily total returns:

1. **The signal.** The M/T ratio, z-scored over a trailing window (so "high"/"low" are relative
   to recent history, not a fixed level). Higher z = munis cheaper.

2. **The forward target.** The muni-minus-Treasury excess return over the next ``horizon`` days —
   the thing the timer is supposed to predict. A *positive* relation to the ratio z is the
   folklore (high ratio -> munis outperform).

3. **The headline test.** A predictive OLS of forward excess on the ratio z, with a Newey-West
   (HAC) standard error (overlapping forward windows are strongly autocorrelated, so a naive t is
   inflated). The HAC *t* on the slope is the headline statistic. Sign > 0 and |t| >= 2 = the
   folklore certifies.

4. **The tradable long-short.** Sort days into ratio quintiles; go long-muni/short-Treasury on the
   top quintile (cheap) and the reverse on the bottom quintile. Report the mean forward excess
   spread (Q5 - Q1), gross and net of costs + a short borrow on the Treasury leg.

5. **A label-shuffle placebo** null and a **multi-horizon robustness sweep** (does the sign hold
   across 21/63/126/252-day horizons?).

6. **A synthetic positive control.** A deterministic world where the timing effect is planted
   (``timing_beta > 0``); the engine must recover a positive HAC-t slope and stay flat at the
   null, averaged over >= 20 seeds (house rule).

Execution convention: the ratio at date *t* uses only data up to and including *t*; the position
is entered at that same close and held over the forward window (*t*, *t*+h]. No future data enters
the signal (no look-ahead). The same-close fill is a single documented convention — on a multi-week
hold a one-bar-later entry moves the headline slope-t by a hair and changes no stamp (see
METHODOLOGY). The robustness sweep and synthetic control use the identical convention.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# Signal + forward target
# ---------------------------------------------------------------------------
def ratio_z(df: pd.DataFrame, lookback: int = 252) -> pd.Series:
    """Trailing z-score of the M/T ratio (higher = munis cheaper relative to recent history).

    Uses a trailing rolling mean/std ending at each date (no look-ahead).
    """
    r = df["mt_ratio"]
    mu = r.rolling(lookback, min_periods=lookback // 2).mean()
    sd = r.rolling(lookback, min_periods=lookback // 2).std(ddof=1)
    z = (r - mu) / sd
    return z.rename("ratio_z")


def forward_excess(df: pd.DataFrame, horizon: int = 63) -> pd.Series:
    """Forward muni-minus-Treasury excess return over the next ``horizon`` trading days.

    Compounded daily returns from t+1 through t+horizon for each leg; the difference is the
    excess. Indexed at the *signal* date t (the position entered at t's close).
    """
    muni_fwd = (1 + df["muni_ret"]).shift(-1).rolling(horizon).apply(np.prod, raw=True).shift(
        -(horizon - 1)
    ) - 1
    tsy_fwd = (1 + df["tsy_ret"]).shift(-1).rolling(horizon).apply(np.prod, raw=True).shift(
        -(horizon - 1)
    ) - 1
    return (muni_fwd - tsy_fwd).rename("fwd_excess")


def aligned(df: pd.DataFrame, lookback: int = 252, horizon: int = 63) -> pd.DataFrame:
    """Signal (ratio z at t) aligned with the forward excess (t, t+h], both fully public at t."""
    z = ratio_z(df, lookback)
    fe = forward_excess(df, horizon)
    out = pd.concat([z, fe], axis=1).dropna()
    return out


# ---------------------------------------------------------------------------
# Newey-West (HAC) OLS — the headline predictive regression
# ---------------------------------------------------------------------------
def _nw_lags(n: int, horizon: int) -> int:
    """Sensible HAC lag: the overlap horizon (minus 1) plus a small cushion."""
    return max(horizon - 1, int(round(4 * (n / 100.0) ** (2.0 / 9.0))))


def hac_regression(x: np.ndarray, y: np.ndarray, lags: int) -> dict:
    """OLS of y on [1, x] with a Newey-West HAC standard error on the slope."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = len(y)
    if n < 10 or np.nanstd(x) == 0:
        return {"slope": float("nan"), "t": float("nan"), "n": n, "r2": float("nan")}
    X = np.column_stack([np.ones(n), x])
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta
    # Newey-West meat matrix.
    S = (X * resid[:, None]).T @ (X * resid[:, None])
    for L in range(1, lags + 1):
        w = 1.0 - L / (lags + 1.0)
        g = (X[L:] * resid[L:, None]).T @ (X[:-L] * resid[:-L, None])
        S += w * (g + g.T)
    cov = XtX_inv @ S @ XtX_inv
    se_slope = float(np.sqrt(cov[1, 1])) if cov[1, 1] > 0 else float("nan")
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float((resid**2).sum()) / ss_tot if ss_tot > 0 else float("nan")
    return {
        "slope": float(beta[1]),
        "t": float(beta[1] / se_slope) if se_slope and se_slope > 0 else float("nan"),
        "n": n,
        "r2": r2,
    }


def headline_test(df: pd.DataFrame, lookback: int = 252, horizon: int = 63) -> dict:
    """The predictive regression: forward excess ~ ratio z, with a HAC t on the slope.

    Slope > 0 and |t| >= 2 = the folklore (high ratio -> munis outperform) certifies on the tape.
    """
    a = aligned(df, lookback, horizon)
    if len(a) < 30:
        return {"slope": float("nan"), "t": float("nan"), "n": len(a), "r2": float("nan")}
    lags = _nw_lags(len(a), horizon)
    return hac_regression(a["ratio_z"].to_numpy(), a["fwd_excess"].to_numpy(), lags)


# ---------------------------------------------------------------------------
# The tradable quintile long-short
# ---------------------------------------------------------------------------
def quintile_spread(df: pd.DataFrame, lookback: int = 252, horizon: int = 63) -> dict:
    """Mean forward excess in the top vs bottom ratio quintile (Q5 - Q1).

    Q5 = munis cheapest (highest ratio z) -> the folklore says overweight muni; Q1 = richest.
    Returns the two bucket means, the spread, and the per-observation forward-excess arrays.
    """
    a = aligned(df, lookback, horizon)
    if len(a) < 20:
        return {}
    z = a["ratio_z"]
    q_hi = z.quantile(0.8)
    q_lo = z.quantile(0.2)
    top = a.loc[z >= q_hi, "fwd_excess"]
    bot = a.loc[z <= q_lo, "fwd_excess"]
    return {
        "n": int(len(a)),
        "q5_mean": float(top.mean()),
        "q1_mean": float(bot.mean()),
        "spread": float(top.mean() - bot.mean()),
        "q5_ret": top,
        "q1_ret": bot,
    }


def spread_tstat(buckets: dict) -> dict:
    """Welch two-sample t on the Q5 vs Q1 forward-excess buckets."""
    if not buckets:
        return {"spread": float("nan"), "t": float("nan"), "n": 0}
    a = buckets["q5_ret"].to_numpy(float)
    b = buckets["q1_ret"].to_numpy(float)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return {"spread": float("nan"), "t": float("nan"), "n": na + nb}
    se = np.sqrt(a.var(ddof=1) / na + b.var(ddof=1) / nb)
    t = (a.mean() - b.mean()) / se if se > 0 else float("nan")
    return {"spread": float(a.mean() - b.mean()), "t": float(t), "n": na + nb}


# ---------------------------------------------------------------------------
# Placebo / label-shuffle null
# ---------------------------------------------------------------------------
def placebo_pvalue(df: pd.DataFrame, lookback: int = 252, horizon: int = 63,
                   n_perm: int = 2000, seed: int = 576) -> float:
    """Label-shuffle placebo p-value for the quintile spread.

    Shuffle the ratio-z labels against the forward excess ``n_perm`` times; the p-value is the
    fraction of shuffles whose |spread| >= the observed |spread|. A real timing signal sits in the
    tail; noise does not. (Shuffling breaks the signal->forward link while preserving the marginal
    distributions, so it is conservative about the overlap autocorrelation.)
    """
    a = aligned(df, lookback, horizon)
    if len(a) < 20:
        return float("nan")
    obs = abs(quintile_spread(df, lookback, horizon)["spread"])
    z = a["ratio_z"].to_numpy(float)
    y = a["fwd_excess"].to_numpy(float)
    n = len(y)
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_perm):
        zp = z[rng.permutation(n)]
        hi = zp >= np.quantile(zp, 0.8)
        lo = zp <= np.quantile(zp, 0.2)
        if hi.sum() < 2 or lo.sum() < 2:
            continue
        spread = y[hi].mean() - y[lo].mean()
        if abs(spread) >= obs - 1e-15:
            count += 1
    return (count + 1) / (n_perm + 1)


# ---------------------------------------------------------------------------
# Costs + borrow
# ---------------------------------------------------------------------------
def net_spread(buckets: dict, cost_bps: float = 3.0, borrow_ann_bps: float = 50.0,
               horizon: int = 63, turns_per_year: float | None = None) -> float:
    """Q5 - Q1 forward-excess spread net of round-trip costs and a Treasury-leg short borrow.

    The long-short book is a paired muni-long / Treasury-short overlay rebalanced each holding
    period. Charge a round-trip one-way cost on each of the two legs (entry + exit) plus an annual
    borrow on the short Treasury leg, pro-rated over the horizon. ``turns_per_year`` scales the
    frictions to a realistic rebalance cadence (default: one turn per horizon).
    """
    if not buckets:
        return float("nan")
    gross = buckets["spread"]
    holding_years = horizon / TRADING_DAYS
    cost = 4.0 * cost_bps * 1e-4          # entry + exit, two legs
    borrow = borrow_ann_bps * 1e-4 * holding_years
    return float(gross - cost - borrow)


# ---------------------------------------------------------------------------
# Robustness — multi-horizon sweep
# ---------------------------------------------------------------------------
def horizon_sweep(df: pd.DataFrame, lookback: int = 252,
                  horizons=(21, 63, 126, 252)) -> list[tuple]:
    """(horizon, slope, HAC-t, quintile spread%) for each forward horizon."""
    out = []
    for h in horizons:
        reg = headline_test(df, lookback, h)
        b = quintile_spread(df, lookback, h)
        spr = b.get("spread", float("nan")) if b else float("nan")
        out.append((int(h), reg["slope"], reg["t"], spr))
    return out


# ---------------------------------------------------------------------------
# Seed-robust synthetic control (house rule for synthetic claims)
# ---------------------------------------------------------------------------
def synthetic_mean_t(data_mod, timing_beta: float, horizon: int = 63,
                     n_seeds: int = 25, base_seed: int = 576) -> float:
    """Average the headline HAC slope-t over ``n_seeds`` synthetic worlds for a planted beta.

    House rule: any synthetic-dependent claim averages the HAC stat over >= 20 seeds so no single
    lucky seed can manufacture significance.
    """
    ts = []
    for s in range(base_seed, base_seed + n_seeds):
        frame, _ = data_mod.synthetic_series(timing_beta=timing_beta, horizon=horizon, seed=s)
        ts.append(headline_test(frame, lookback=252, horizon=horizon)["t"])
    return float(np.nanmean(ts))
