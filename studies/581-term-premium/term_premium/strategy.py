"""The engine and its honest controls — Study 581 (Term-Premium).

The claim, at full strength (Adrian, Crump & Moench 2013; Cochrane & Piazzesi 2005; Fama & Bliss
1987): the **term premium** — the part of the long yield not explained by expected future short
rates — is the *compensation* for bearing duration risk, and it *varies over time*. When the
premium is fat, long bonds are richly paid and should out-earn cash going forward; when it is
thin or negative, duration is poorly compensated and you should step aside. So a model
term-premium estimate ought to **time long-duration (TLT) returns**.

This module tests that, honestly, on the ACM-style proxy a retail stack can build:

1. **The signal.** The term-premium proxy ``tp = y10 - EWMA(short)`` (built in ``data.py``),
   ranked out-of-sample over a trailing 252-day window into ``tp_rank`` in [0,1]. High rank = fat
   premium = "own duration".

2. **The forward-return sort.** Bucket days by ``tp_rank`` (lagged one day) into quintiles and
   compare the mean forward TLT return of the fattest-premium quintile (Q5) vs the thinnest (Q1).
   The claim predicts Q5 > Q1.

3. **The headline statistic.** A HAC (Newey-West) *t* on the Q5-Q1 daily forward-return spread —
   autocorrelation-robust, the bar the desk requires for a `REAL` stamp.

4. **A placebo null.** A block-shuffle of the signal against forward returns: if the Q5-Q1 spread
   is noise, shuffled spreads sit around it; a real edge sits in the tail.

5. **A timing overlay + costs.** Own TLT on high-premium days, cash otherwise; charge a one-way
   cost per switch on the traded NAV and compare net Sharpe to buy-and-hold TLT.

6. **A robustness sweep.** Re-run the Q5-Q1 spread across forward horizons and sub-periods; read
   the sign and the *t*.

7. **A synthetic positive control.** A deterministic world where the timing edge is planted
   (``tp_signal > 0``); the engine must recover a positive Q5-Q1 spread that clears the bar as the
   planted signal grows, and stay flat at the null — averaged over >= 20 seeds (house rule).

Execution convention: the term premium at the close of day *t* forms the signal; the position is
entered at the close of day *t+1* and the forward return is measured over the following horizon.
That one-bar lag is the single documented execution convention, applied identically on both tapes
and across the robustness sweep. No future data enters the signal (the EWMA and the rolling rank
are causal).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# Signal construction
# ---------------------------------------------------------------------------
def tp_rank(df: pd.DataFrame, lookback: int = 252, min_periods: int = 63) -> pd.Series:
    """Out-of-sample rolling percentile rank of the term premium in [0,1] (higher = fatter)."""
    return df["tp"].rolling(lookback, min_periods=min_periods).rank(pct=True).rename("tp_rank")


def forward_return(df: pd.DataFrame, horizon: int = 21) -> pd.Series:
    """Forward simple TLT return over ``horizon`` trading days, entered at the NEXT close.

    The signal at close *t* trades at close *t+1*; the forward return runs from *t+1* to
    *t+1+horizon*. Implemented as the sum of the next ``horizon`` log returns shifted so it aligns
    with the day-*t* signal, then converted to simple. Uses only future prices relative to the
    signal — the single one-bar execution lag.
    """
    r = np.log(df["TLT_close"]).diff()
    # forward cumulative log return over (t+1 .. t+horizon], aligned to signal day t
    fwd_log = r.shift(-1).rolling(horizon).sum().shift(-(horizon - 1))
    return (np.expm1(fwd_log)).rename(f"fwd_{horizon}")


# ---------------------------------------------------------------------------
# The quintile sort + HAC t
# ---------------------------------------------------------------------------
def _hac_mean_t(x: np.ndarray, lags: int) -> float:
    """Newey-West HAC t-stat for the mean of an autocorrelated series."""
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 5:
        return float("nan")
    mu = x.mean()
    e = x - mu
    gamma0 = float(e @ e) / n
    var = gamma0
    for k in range(1, min(lags, n - 1) + 1):
        w = 1.0 - k / (lags + 1.0)
        cov = float(e[k:] @ e[:-k]) / n
        var += 2.0 * w * cov
    se = np.sqrt(max(var, 1e-18) / n)
    return float(mu / se) if se > 0 else float("nan")


def quintile_spread(df: pd.DataFrame, horizon: int = 21, lookback: int = 252) -> dict:
    """Q5-Q1 forward-TLT-return spread sorted on the (lagged) term-premium rank, with a HAC t.

    The claim predicts the fattest-premium quintile (Q5) out-earns the thinnest (Q1). Returns the
    two bucket means, the spread, a HAC (Newey-West) *t* on the day-level Q5-Q1 difference series,
    and the bucket sizes.
    """
    rank = tp_rank(df, lookback).shift(1)  # signal known at t, trades t+1
    fwd = forward_return(df, horizon)
    d = pd.DataFrame({"rank": rank, "fwd": fwd}).dropna()
    if len(d) < 100:
        return {"q1": float("nan"), "q5": float("nan"), "spread": float("nan"),
                "t": float("nan"), "n": len(d)}
    q = pd.qcut(d["rank"], 5, labels=False, duplicates="drop")
    d = d.assign(q=q)
    q1 = d.loc[d["q"] == d["q"].min(), "fwd"]
    q5 = d.loc[d["q"] == d["q"].max(), "fwd"]
    # HAC t on the difference of daily forward returns between the two extreme buckets:
    # build a paired series over the union of trading days (mean-diff test via pooled HAC).
    diff_series = pd.concat([q5.rename("v"), (-q1).rename("v")]).sort_index()
    lags = max(horizon, 21)
    t = _hac_mean_t(diff_series.to_numpy(dtype=float), lags)
    return {
        "q1": float(q1.mean()),
        "q5": float(q5.mean()),
        "spread": float(q5.mean() - q1.mean()),
        "t": t,
        "n": int(len(d)),
        "n_q": int(len(q5)),
    }


# ---------------------------------------------------------------------------
# Placebo / block-shuffle null
# ---------------------------------------------------------------------------
def placebo_pvalue(df: pd.DataFrame, horizon: int = 21, lookback: int = 252,
                   n_perm: int = 1000, block: int = 21, seed: int = 581) -> float:
    """Block-shuffle placebo p-value for the Q5-Q1 spread.

    Circularly rotate the forward-return series in blocks against the signal ``n_perm`` times; the
    p-value is the fraction of shuffles whose |Q5-Q1 spread| >= the observed |spread|. Block
    rotation preserves the autocorrelation of overlapping forward returns, so the null is honest.
    """
    rank = tp_rank(df, lookback).shift(1)
    fwd = forward_return(df, horizon)
    d = pd.DataFrame({"rank": rank, "fwd": fwd}).dropna().reset_index(drop=True)
    if len(d) < 200:
        return float("nan")
    obs = abs(_spread_from(d["rank"].to_numpy(), d["fwd"].to_numpy()))
    rng = np.random.default_rng(seed)
    r = d["rank"].to_numpy()
    f = d["fwd"].to_numpy()
    n = len(f)
    count = 0
    for _ in range(n_perm):
        shift = int(rng.integers(block, n - block))
        fs = np.roll(f, shift)
        if abs(_spread_from(r, fs)) >= obs - 1e-15:
            count += 1
    return (count + 1) / (n_perm + 1)


def _spread_from(rank: np.ndarray, fwd: np.ndarray) -> float:
    """Q5-Q1 forward-return spread for aligned rank/fwd arrays (helper for the placebo)."""
    try:
        q = pd.qcut(pd.Series(rank), 5, labels=False, duplicates="drop").to_numpy()
    except ValueError:
        return float("nan")
    lo, hi = np.nanmin(q), np.nanmax(q)
    return float(fwd[q == hi].mean() - fwd[q == lo].mean())


# ---------------------------------------------------------------------------
# Timing overlay + costs vs buy-and-hold
# ---------------------------------------------------------------------------
def timing_overlay(df: pd.DataFrame, lookback: int = 252, thresh: float = 0.5,
                   cost_bps: float = 2.0) -> dict:
    """Own TLT when the term-premium rank (lagged) is above ``thresh``, else cash; net of costs.

    Costs are charged one-way on the traded NAV at each switch (``cost_bps``). Returns the timer's
    net annualised Sharpe, the buy-and-hold TLT Sharpe (both excess of a flat cash proxy = 0), the
    number of switches per year, and the mean-return spread with a HAC *t*.
    """
    rank = tp_rank(df, lookback).shift(1)
    ret = np.log(df["TLT_close"]).diff()  # daily log return of TLT
    d = pd.DataFrame({"rank": rank, "ret": ret}).dropna()
    if len(d) < 200:
        return {}
    pos = (d["rank"] > thresh).astype(float)
    switches = pos.diff().abs().fillna(0.0)
    cost = switches * cost_bps * 1e-4
    timer_ret = pos * d["ret"] - cost
    bh_ret = d["ret"]

    def sharpe(x):
        s = x.std(ddof=1)
        return float(x.mean() / s * np.sqrt(TRADING_DAYS)) if s > 0 else float("nan")

    diff = (timer_ret - bh_ret).to_numpy(dtype=float)
    return {
        "timer_sharpe": sharpe(timer_ret),
        "bh_sharpe": sharpe(bh_ret),
        "switches_per_yr": float(switches.sum() / len(d) * TRADING_DAYS),
        "spread_bps_day": float(diff.mean() * 1e4),
        "spread_t": _hac_mean_t(diff, 21),
        "n": int(len(d)),
        "days_invested_frac": float(pos.mean()),
    }


# ---------------------------------------------------------------------------
# Robustness sweep
# ---------------------------------------------------------------------------
def horizon_sweep(df: pd.DataFrame, horizons=(5, 21, 63, 126), lookback: int = 252) -> pd.DataFrame:
    """Q5-Q1 spread + HAC t across forward horizons (in trading days)."""
    rows = []
    for h in horizons:
        s = quintile_spread(df, horizon=h, lookback=lookback)
        rows.append((f"{h}d", s["spread"], s["t"], s["n"]))
    return pd.DataFrame(rows, columns=["horizon", "spread", "t", "n"]).set_index("horizon")


def subperiod_sweep(df: pd.DataFrame, edges, horizon: int = 21, lookback: int = 252) -> pd.DataFrame:
    """Q5-Q1 spread + HAC t within date sub-periods (``edges`` = list of (label, start, end))."""
    rows = []
    for lab, start, end in edges:
        sub = df.loc[start:end]
        s = quintile_spread(sub, horizon=horizon, lookback=lookback)
        rows.append((lab, s["spread"], s["t"], s["n"]))
    return pd.DataFrame(rows, columns=["period", "spread", "t", "n"]).set_index("period")


# ---------------------------------------------------------------------------
# Synthetic positive control — seed-robust (house rule for synthetic claims)
# ---------------------------------------------------------------------------
def synthetic_mean_t(data_mod, tp_signal: float, n_seeds: int = 25, base_seed: int = 581,
                     horizon: int = 21) -> dict:
    """Average the Q5-Q1 spread and its HAC t over ``n_seeds`` synthetic worlds.

    House rule: any synthetic-dependent claim averages the statistic over >= 20 seeds so no single
    lucky RNG seed can manufacture significance. Returns the mean spread and mean HAC t across
    seeds for a planted ``tp_signal``.
    """
    spreads, ts = [], []
    for s in range(base_seed, base_seed + n_seeds):
        dfd, _ = data_mod.synthetic_daily(tp_signal=tp_signal, seed=s)
        out = quintile_spread(dfd, horizon=horizon)
        spreads.append(out["spread"])
        ts.append(out["t"])
    return {"mean_spread": float(np.nanmean(spreads)), "mean_t": float(np.nanmean(ts))}
