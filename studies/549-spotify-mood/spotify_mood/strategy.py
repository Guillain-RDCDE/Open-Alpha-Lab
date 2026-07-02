"""The engine and its honest controls — Study 549 (Spotify-Mood).

The claim, at full strength (the music-sentiment literature): the aggregate musical **valence** of
the top-streamed songs is a national/global mood proxy, and mood *leads* the tape — a happier month
of streaming forecasts a higher market return *next* month, a sadder month a lower one. The
tradable expression is a **mood-timing** rule: go long the market when last month's valence is
above its median, else sit in cash (or short).

This module measures that, honestly, on a synthetic mood tape × the real monthly market:

1. **The predictive regression.** OLS of *next*-month market return on *this*-month valence
   (z-scored), with a **Newey-West / HAC** standard error (overlapping-mood autocorrelation would
   otherwise inflate the naive t). The HAC t on the slope is the headline inference statistic — a
   REAL signal needs |t| >= 2 *on a real valence tape*, which does not exist for free (see data.py).

2. **A lag sweep.** Regress return[t] on valence[t-k] for k = 1..5 — the Granger-lag
   multiple-comparisons trap. If the "signal" only shows at one cherry-picked lag, it is noise; a
   real lead-lag survives a Bonferroni-style bar.

3. **A time-shuffle placebo null.** Circularly shift the valence series against returns many times
   and read where the observed |slope| sits in the shuffled distribution.

4. **The trading rule, gross AND net.** A one-month execution lag (trade on *last* month's valence,
   which is fully public), one-way costs × NAV on each rebalance, and — for the long-short variant —
   the short leg pays borrow. Gross and net are both labelled.

5. **A seed-robust synthetic positive control.** Plant a genuine predictive edge
   (``predictive_beta > 0``) and average the HAC t over >= 20 seeds; the engine must recover it and
   stay ~flat at the null.

Execution convention: the rule at month t uses valence known at the *close of month t-1* (last
month's completed streaming), so the signal is fully public before the return it is meant to
predict — the single, documented one-month execution lag. No future data enters the signal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MONTHS_PER_YEAR = 12


def _z(x: np.ndarray) -> np.ndarray:
    sd = x.std(ddof=1)
    return (x - x.mean()) / sd if sd and sd > 0 else x * 0.0


# ---------------------------------------------------------------------------
# Newey-West / HAC OLS of return[t] on valence[t-lag]
# ---------------------------------------------------------------------------
def predictive_regression(frame: pd.DataFrame, lag: int = 1, hac_lags: int | None = None) -> dict:
    """OLS of market return on *lagged* valence (z-scored) with a Newey-West HAC t on the slope.

    ``lag`` shifts valence forward by ``lag`` months so that ``valence[t-lag]`` predicts
    ``mkt_ret[t]`` — the lead-lag the claim asserts (lag = 1 is the headline). ``hac_lags`` is the
    Newey-West bandwidth (default ~ n**(1/4)). Returns slope, its plain and HAC t, R², and n.
    """
    if frame.empty or len(frame) <= lag + 4:
        return {"slope": np.nan, "t": np.nan, "hac_t": np.nan, "r2": np.nan, "n": 0}
    v = frame["valence"].to_numpy(dtype=float)
    r = frame["mkt_ret"].to_numpy(dtype=float)
    x = _z(v[:-lag]) if lag > 0 else _z(v)
    y = r[lag:] if lag > 0 else r
    n = len(y)
    X = np.column_stack([np.ones(n), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    xtx_inv = np.linalg.inv(X.T @ X)

    # plain OLS se
    dof = max(n - 2, 1)
    sigma2 = float(resid @ resid) / dof
    se_ols = float(np.sqrt(sigma2 * xtx_inv[1, 1]))

    # Newey-West HAC covariance
    if hac_lags is None:
        hac_lags = int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0)))
        hac_lags = max(hac_lags, 1)
    S = np.zeros((2, 2))
    Xe = X * resid[:, None]
    S += Xe.T @ Xe
    for L in range(1, hac_lags + 1):
        w = 1.0 - L / (hac_lags + 1.0)
        G = Xe[L:].T @ Xe[:-L]
        S += w * (G + G.T)
    cov_hac = xtx_inv @ S @ xtx_inv
    se_hac = float(np.sqrt(cov_hac[1, 1])) if cov_hac[1, 1] > 0 else np.nan

    r2 = 1.0 - float(resid @ resid) / float(((y - y.mean()) ** 2).sum() + 1e-15)
    return {
        "slope": float(beta[1]),
        "t": float(beta[1] / se_ols) if se_ols > 0 else np.nan,
        "hac_t": float(beta[1] / se_hac) if se_hac and se_hac > 0 else np.nan,
        "r2": float(r2),
        "n": int(n),
    }


# ---------------------------------------------------------------------------
# Lag sweep — the Granger multiple-comparisons trap
# ---------------------------------------------------------------------------
def lag_sweep(frame: pd.DataFrame, lags=(1, 2, 3, 4, 5)) -> pd.DataFrame:
    """HAC t on the valence->return slope for each lag; the max |t| is the data-mined best."""
    rows = []
    for k in lags:
        reg = predictive_regression(frame, lag=k)
        rows.append((k, reg["slope"], reg["hac_t"], reg["n"]))
    return pd.DataFrame(rows, columns=["lag", "slope", "hac_t", "n"]).set_index("lag")


# ---------------------------------------------------------------------------
# Time-shuffle placebo null
# ---------------------------------------------------------------------------
def placebo_pvalue(frame: pd.DataFrame, lag: int = 1, n_perm: int = 2000, seed: int = 549) -> float:
    """Circular-shift placebo p-value for the lag-``lag`` predictive slope.

    Circularly shift the valence series against returns ``n_perm`` times (preserving valence's own
    autocorrelation) and read the fraction of shifts whose |slope| >= the observed |slope|.
    """
    if frame.empty or len(frame) <= lag + 6:
        return np.nan
    obs = abs(predictive_regression(frame, lag=lag)["slope"])
    if not np.isfinite(obs):
        return np.nan
    v = frame["valence"].to_numpy(dtype=float)
    r = frame["mkt_ret"].to_numpy(dtype=float)
    n = len(r)
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_perm):
        shift = int(rng.integers(1, n - 1))
        vs = np.roll(v, shift)
        x = _z(vs[:-lag])
        y = r[lag:]
        X = np.column_stack([np.ones(len(y)), x])
        b, *_ = np.linalg.lstsq(X, y, rcond=None)
        if abs(b[1]) >= obs - 1e-15:
            count += 1
    return (count + 1) / (n_perm + 1)


# ---------------------------------------------------------------------------
# The mood-timing trading rule — gross AND net
# ---------------------------------------------------------------------------
def mood_timing(
    frame: pd.DataFrame,
    long_short: bool = False,
    cost_bps: float = 5.0,
    borrow_ann_bps: float = 100.0,
) -> dict:
    """Trade the market on last month's valence, with a one-month execution lag.

    Rule: at month t, position = +1 if valence[t-1] > its trailing median (bullish mood), else 0
    (long-only) or -1 (long-short). Costs are charged one-way × NAV on each position *change*; the
    short leg (long-short only) pays an annual borrow while short. Returns gross/net mean monthly
    excess-vs-market, annualised, plus the strategy's own t-stat and the buy-and-hold benchmark.
    """
    if frame.empty or len(frame) < 24:
        return {}
    v = frame["valence"].to_numpy(dtype=float)
    r = frame["mkt_ret"].to_numpy(dtype=float)
    n = len(r)
    # trailing (expanding) median of valence, known at t-1
    pos = np.zeros(n)
    for t in range(13, n):
        med = np.median(v[:t])          # uses v[0..t-1], public before month t's return
        bullish = v[t - 1] > med
        pos[t] = 1.0 if bullish else (-1.0 if long_short else 0.0)
    strat_gross = pos * r
    # costs on position changes
    dpos = np.abs(np.diff(np.concatenate([[0.0], pos])))
    cost = dpos * (cost_bps * 1e-4)
    borrow = np.where(pos < 0, borrow_ann_bps * 1e-4 / MONTHS_PER_YEAR, 0.0)
    strat_net = strat_gross - cost - borrow

    live = pos != 0
    active_g = strat_gross[13:]
    active_n = strat_net[13:]
    mkt_active = r[13:]

    def _ann(x):
        return float((1.0 + np.mean(x)) ** MONTHS_PER_YEAR - 1.0)

    def _tstat(x):
        x = np.asarray(x, dtype=float)
        sd = x.std(ddof=1)
        return float(np.mean(x) / (sd / np.sqrt(len(x)))) if sd > 0 and len(x) > 1 else np.nan

    return {
        "gross_ann": _ann(active_g),
        "net_ann": _ann(active_n),
        "mkt_ann": _ann(mkt_active),
        "gross_t": _tstat(active_g),
        "net_t": _tstat(active_n),
        "months_live": int(live.sum()),
        "n": int(n),
    }


# ---------------------------------------------------------------------------
# Directional hit-rate vs the honest base rate
# ---------------------------------------------------------------------------
def hit_rate(frame: pd.DataFrame, lag: int = 1) -> dict:
    """Directional hit-rate of 'valence up-month predicts up-market' vs the unconditional up-rate."""
    if frame.empty or len(frame) <= lag + 4:
        return {"hit": np.nan, "base": np.nan, "n": 0}
    v = frame["valence"].to_numpy(dtype=float)
    r = frame["mkt_ret"].to_numpy(dtype=float)
    vz = _z(v[:-lag])
    y = r[lag:]
    pred_up = vz > 0
    actual_up = y > 0
    hit = float(np.mean(pred_up == actual_up))
    base = float(max(np.mean(actual_up), 1.0 - np.mean(actual_up)))
    return {"hit": hit, "base": base, "n": int(len(y))}


# ---------------------------------------------------------------------------
# Robustness — seed-robust synthetic positive control (house rule)
# ---------------------------------------------------------------------------
def synthetic_mean_t(data_mod, predictive_beta: float, n_seeds: int = 25,
                     base_seed: int = 549, n_months: int = 180) -> float:
    """Average the lag-1 predictive HAC t over ``n_seeds`` synthetic worlds.

    House rule: any synthetic-dependent claim averages a HAC-t over >= 20 seeds so no single lucky
    RNG seed can manufacture significance. Returns the mean HAC t for a planted ``predictive_beta``.
    """
    ts = []
    for s in range(base_seed, base_seed + n_seeds):
        frame, _ = data_mod.synthetic_series(
            n_months=n_months, predictive_beta=predictive_beta, seed=s
        )
        ts.append(predictive_regression(frame, lag=1)["hac_t"])
    return float(np.nanmean(ts))
