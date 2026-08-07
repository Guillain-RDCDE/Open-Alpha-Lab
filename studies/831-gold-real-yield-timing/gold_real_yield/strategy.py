"""The engine and its honest controls — Study 831 (Gold Real-Yield Timing).

The claim, at full strength (Gorton & Rouwenhorst 2006; Erb & Harvey 2013; Baur &
McDermott 2010; and the desk-standard practitioner story): gold has no yield, so its
appeal is inversely tied to the **real** yield on safe assets — when real rates fall,
the opportunity cost of holding gold drops and gold rises; when real rates climb, gold
should fall. The *tradeable* version: a "real yields are falling → own gold" **timing
rule** should beat buying and holding gold.

This module tests that, honestly, on the TIP-gauge real-yield proxy a retail stack can
build (see ``data.py``):

1. **The signal.** ``ryfall = log(TIP_t) − log(TIP_{t−L})`` — the trailing ``L``-day
   real-yield *fall* (TIP up ⇔ real yields down), ranked out-of-sample over a trailing
   252-day window into ``ryfall_rank`` in [0,1]. High rank = real yields have been
   falling hardest = the claim's "own gold" state.

2. **The forward-return sort.** Bucket days by the (lagged) ``ryfall_rank`` into
   quintiles and compare the mean forward GLD return of the fastest-falling-real-yield
   quintile (Q5) vs the fastest-rising (Q1). The claim predicts Q5 > Q1.

3. **The headline statistic.** A HAC (Newey-West) *t* on the Q5−Q1 daily forward-return
   spread — autocorrelation-robust, the bar the desk requires for a `REAL` stamp.

4. **A placebo null.** A block-shuffle of the signal against forward returns.

5. **A timing overlay + costs.** Own GLD when real yields are falling, cash otherwise;
   charge one-way cost per switch on the traded NAV and compare net Sharpe *and mean
   return* to buy-and-hold GLD.

6. **Robustness sweeps.** Q5−Q1 across forward horizons and sub-periods; read sign & *t*.

7. **The inverse-link cross-check (descriptive).** The *contemporaneous* correlation of
   the daily gold return with the same-day real-yield change — the famous inverse fact,
   which is real but *not* a timing signal (it needs the yield you are trading against).

8. **A synthetic positive control.** A deterministic world where the timing edge is
   planted (``edge > 0``); the engine must recover a positive Q5−Q1 spread clearing the
   bar as the planted signal grows, and stay flat at the null — averaged over ≥ 20 seeds.

Execution convention: the real-yield trend at the close of day *t* forms the signal; the
position enters at the close of *t+1* and the forward return is measured from *t+1*. That
one-bar lag is the single documented execution convention, applied identically on both
tapes and across the sweeps. No future data enters the signal (the rolling window and
rank are causal).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# Signal construction
# ---------------------------------------------------------------------------
def ryfall(df: pd.DataFrame, lookback: int = 63) -> pd.Series:
    """Trailing ``lookback``-day real-yield *fall* from the TIP gauge (>0 ⇔ yields fell).

    ``log(TIP_t) − log(TIP_{t−lookback})`` — the TIP total return, which rises exactly
    when real yields fall. Causal (only trailing prices).
    """
    lt = np.log(df["TIP_close"])
    return (lt - lt.shift(lookback)).rename("ryfall")


def ryfall_rank(df: pd.DataFrame, lookback: int = 63, rank_win: int = 252,
                min_periods: int = 63) -> pd.Series:
    """Out-of-sample rolling percentile rank of ``ryfall`` in [0,1] (higher = yields falling harder)."""
    s = ryfall(df, lookback)
    return s.rolling(rank_win, min_periods=min_periods).rank(pct=True).rename("ryfall_rank")


def forward_return(df: pd.DataFrame, horizon: int = 21) -> pd.Series:
    """Forward simple GLD return over ``horizon`` trading days, entered at the NEXT close.

    Signal at close *t* trades at close *t+1*; the forward return runs from *t+1* to
    *t+1+horizon*. Sum of the next ``horizon`` log returns, shifted to align with the
    day-*t* signal, converted to simple. Uses only future prices relative to the signal.
    """
    r = np.log(df["GLD_close"]).diff()
    fwd_log = r.shift(-1).rolling(horizon).sum().shift(-(horizon - 1))
    return np.expm1(fwd_log).rename(f"fwd_{horizon}")


# ---------------------------------------------------------------------------
# Inference primitives (shared house set)
# ---------------------------------------------------------------------------
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


def newey_west_t(x: np.ndarray, lags: int = 21) -> float:
    """HAC (Newey-West, Bartlett kernel) t of mean(x) vs 0."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 5:
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


# ---------------------------------------------------------------------------
# The quintile sort + HAC t
# ---------------------------------------------------------------------------
def quintile_spread(df: pd.DataFrame, horizon: int = 21, lookback: int = 63,
                    rank_win: int = 252) -> dict:
    """Q5−Q1 forward-GLD-return spread sorted on the (lagged) real-yield-fall rank, HAC t.

    The claim predicts the fastest-falling-real-yield quintile (Q5) out-earns the
    fastest-rising (Q1). Returns the two bucket means, the spread, a HAC (Newey-West) *t*
    on the day-level Q5−Q1 difference series, and bucket sizes.
    """
    rank = ryfall_rank(df, lookback, rank_win).shift(1)  # known at t, trades t+1
    fwd = forward_return(df, horizon)
    d = pd.DataFrame({"rank": rank, "fwd": fwd}).dropna()
    if len(d) < 100:
        return {"q1": float("nan"), "q5": float("nan"), "spread": float("nan"),
                "t": float("nan"), "n": len(d), "n_q": 0}
    q = pd.qcut(d["rank"], 5, labels=False, duplicates="drop")
    d = d.assign(q=q)
    q1 = d.loc[d["q"] == d["q"].min(), "fwd"]
    q5 = d.loc[d["q"] == d["q"].max(), "fwd"]
    diff_series = pd.concat([q5.rename("v"), (-q1).rename("v")]).sort_index()
    lags = max(horizon, 21)
    t = newey_west_t(diff_series.to_numpy(dtype=float), lags)
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
def _spread_from(rank: np.ndarray, fwd: np.ndarray) -> float:
    """Q5−Q1 forward-return spread for aligned rank/fwd arrays (helper for the placebo)."""
    try:
        q = pd.qcut(pd.Series(rank), 5, labels=False, duplicates="drop").to_numpy()
    except ValueError:
        return float("nan")
    lo, hi = np.nanmin(q), np.nanmax(q)
    return float(fwd[q == hi].mean() - fwd[q == lo].mean())


def placebo_pvalue(df: pd.DataFrame, horizon: int = 21, lookback: int = 63,
                   rank_win: int = 252, n_perm: int = 1000, block: int = 21,
                   seed: int = 831) -> float:
    """Block-shuffle placebo p-value for the Q5−Q1 spread.

    Circularly rotate the forward-return series in blocks against the signal ``n_perm``
    times; the p-value is the fraction of shuffles whose |Q5−Q1 spread| ≥ the observed
    |spread|. Block rotation preserves the overlapping-return autocorrelation, so the
    null is honest.
    """
    rank = ryfall_rank(df, lookback, rank_win).shift(1)
    fwd = forward_return(df, horizon)
    d = pd.DataFrame({"rank": rank, "fwd": fwd}).dropna().reset_index(drop=True)
    if len(d) < 200:
        return float("nan")
    r = d["rank"].to_numpy()
    f = d["fwd"].to_numpy()
    obs = abs(_spread_from(r, f))
    rng = np.random.default_rng(seed)
    n = len(f)
    count = 0
    for _ in range(n_perm):
        shift = int(rng.integers(block, n - block))
        if abs(_spread_from(r, np.roll(f, shift))) >= obs - 1e-15:
            count += 1
    return (count + 1) / (n_perm + 1)


# ---------------------------------------------------------------------------
# Timing overlay + costs vs buy-and-hold gold
# ---------------------------------------------------------------------------
def timing_overlay(df: pd.DataFrame, lookback: int = 63, rank_win: int = 252,
                   thresh: float = 0.5, cost_bps: float = 2.0) -> dict:
    """Own GLD when the real-yield-fall rank (lagged) is above ``thresh``, else cash; net of costs.

    Costs are charged one-way on the traded NAV at each switch (``cost_bps``). Returns
    the timer's net annualised Sharpe, the buy-and-hold GLD Sharpe, switches per year,
    the days-invested fraction, and the mean-return spread with a HAC *t*.
    """
    rank = ryfall_rank(df, lookback, rank_win).shift(1)
    ret = np.log(df["GLD_close"]).diff()
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
        "spread_t": newey_west_t(diff, 21),
        "n": int(len(d)),
        "days_invested_frac": float(pos.mean()),
    }


# ---------------------------------------------------------------------------
# Robustness sweeps
# ---------------------------------------------------------------------------
def horizon_sweep(df: pd.DataFrame, horizons=(5, 21, 63, 126), lookback: int = 63) -> pd.DataFrame:
    """Q5−Q1 spread + HAC t across forward horizons (trading days)."""
    rows = []
    for h in horizons:
        s = quintile_spread(df, horizon=h, lookback=lookback)
        rows.append((f"{h}d", s["spread"], s["t"], s["n"]))
    return pd.DataFrame(rows, columns=["horizon", "spread", "t", "n"]).set_index("horizon")


def subperiod_sweep(df: pd.DataFrame, edges, horizon: int = 21, lookback: int = 63) -> pd.DataFrame:
    """Q5−Q1 spread + HAC t within date sub-periods (``edges`` = list of (label, start, end))."""
    rows = []
    for lab, start, end in edges:
        sub = df.loc[start:end]
        s = quintile_spread(sub, horizon=horizon, lookback=lookback)
        rows.append((lab, s["spread"], s["t"], s["n"]))
    return pd.DataFrame(rows, columns=["period", "spread", "t", "n"]).set_index("period")


def lookback_sweep(df: pd.DataFrame, lookbacks=(21, 63, 126, 252), horizon: int = 21) -> pd.DataFrame:
    """Q5−Q1 spread + HAC t across trailing real-yield-fall lookbacks (trading days)."""
    rows = []
    for lb in lookbacks:
        s = quintile_spread(df, horizon=horizon, lookback=lb)
        rows.append((f"{lb}d", s["spread"], s["t"], s["n"]))
    return pd.DataFrame(rows, columns=["lookback", "spread", "t", "n"]).set_index("lookback")


# ---------------------------------------------------------------------------
# The inverse-link cross-check (descriptive) — contemporaneous, NOT tradable
# ---------------------------------------------------------------------------
def inverse_link(df: pd.DataFrame) -> dict:
    """Contemporaneous daily correlation of gold return with the same-day real-yield change.

    Real-yield change is proxied by the sign-flipped TIP log return (TIP up ⇔ real yield
    down), so ``d_real_yield ≈ −Δlog(TIP)``. A strong *negative* corr(GLD return, real-
    yield change) is the famous inverse fact — real, but same-day (needs the yield you
    trade against), hence descriptive only, never a timing edge. Also reports the OLS
    beta of gold on the real-yield change and its Newey-West t.
    """
    gret = np.log(df["GLD_close"]).diff()
    d_ry = -np.log(df["TIP_close"]).diff()  # +Δ real yield ≈ −Δlog TIP
    d = pd.DataFrame({"g": gret, "dry": d_ry}).dropna()
    if len(d) < 50:
        return {"corr": float("nan"), "beta": float("nan"), "t": float("nan"), "n": len(d)}
    g = d["g"].to_numpy(); x = d["dry"].to_numpy()
    corr = float(np.corrcoef(g, x)[0, 1])
    vx = x.var(ddof=0)
    beta = float(np.cov(g, x, ddof=0)[0, 1] / vx) if vx > 0 else float("nan")
    # HAC t on the covariance (⇔ beta ⇔ corr) via the cross-product score series:
    # mean(score) = cov(x,g); a Newey-West t of that mean tests H0: cov=0 ⇔ beta=0.
    score = (x - x.mean()) * (g - g.mean())
    t_beta = newey_west_t(score, 21)
    return {"corr": corr, "beta": beta, "t": t_beta, "n": int(len(d))}


# ---------------------------------------------------------------------------
# Synthetic positive control — seed-robust (house rule for synthetic claims)
# ---------------------------------------------------------------------------
def synthetic_mean_t(data_mod, edge: float, n_seeds: int = 25, base_seed: int = 831,
                     horizon: int = 21, n_days: int = 3000, link_beta: float = 8.0) -> dict:
    """Average the Q5−Q1 spread and its HAC t over ``n_seeds`` synthetic worlds.

    House rule: any synthetic-dependent claim averages the statistic over ≥ 20 seeds so
    no single lucky RNG seed can manufacture significance. Returns the mean spread and
    mean HAC t across seeds for a planted timing ``edge``.
    """
    spreads, ts = [], []
    for s in range(base_seed, base_seed + n_seeds):
        dfd, _ = data_mod.synthetic_daily(n_days=n_days, edge=edge, link_beta=link_beta, seed=s)
        out = quintile_spread(dfd, horizon=horizon)
        spreads.append(out["spread"])
        ts.append(out["t"])
    return {"mean_spread": float(np.nanmean(spreads)), "mean_t": float(np.nanmean(ts))}
