"""Strategy and analysis layer for Study 268 (Sahm-Rule).

The Sahm Rule (Sahm, 2019, in *Recession Ready*): a recession has begun when the
3-month moving average of the seasonally-adjusted U-3 unemployment rate rises
**0.50 percentage points or more** above its minimum over the trailing 12 months.
It was designed as a *recession identifier* (for triggering automatic fiscal
stabilizers), NOT as a market-timing signal. The table hook asks the obvious
follow-up: *is the Sahm trigger a usable sell button?*

We implement the trigger mechanically and then ask two honest questions:

  - **Does the trigger lead a worse-than-average S&P drawdown?** We measure the
    forward H-month S&P return after each trigger (with an execution lag) and
    compare it to the *unconditional* forward H-month return. The right null is
    the unconditional distribution of forward returns, because equities have an
    upward drift -- a "sell" signal must beat sitting in the index, not a coin.

  - **Can you trade it?** We build a binary timing overlay (out of the market
    while the Sahm trigger is on, long otherwise), net of one-way costs, and pin
    it against passive buy-and-hold. With only a handful of triggers in 65 years,
    the standard errors are enormous.

The tiny-n reckoning: there are ~7-9 distinct Sahm episodes since 1959. Any claim
about average post-trigger returns rests on that handful of events; we report a
HAC (Newey-West) t-stat on the overlapping forward-return series and a block
permutation test, and we are explicit that n_episodes < 10 cannot clear a t >= 2
bar except by luck.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


# ---------------------------------------------------------------------------
# The Sahm indicator
# ---------------------------------------------------------------------------
def sahm_indicator(unrate: pd.Series, threshold: float = 0.50) -> pd.DataFrame:
    """Compute the Sahm recession indicator from a monthly unemployment series.

    Parameters
    ----------
    unrate : monthly SA unemployment rate (percent), date-indexed.
    threshold : the trigger gap in percentage points (Sahm's value is 0.50).

    Returns a DataFrame indexed by month with columns:
      - ``unrate``    : the input rate
      - ``ma3``       : 3-month moving average of the rate
      - ``min12``     : trailing-12-month minimum of ``ma3`` (inclusive)
      - ``sahm_gap``  : ``ma3 - min12`` (the Sahm value, in pp)
      - ``trigger``   : bool, True where ``sahm_gap >= threshold``
      - ``onset``     : bool, True only on the *first* month a new trigger turns on
                        (a rising edge after at least one month of no-trigger)
    """
    s = unrate.astype(float).sort_index()
    ma3 = s.rolling(3, min_periods=3).mean()
    min12 = ma3.rolling(12, min_periods=1).min()
    gap = ma3 - min12
    trigger = gap >= threshold

    df = pd.DataFrame(
        {
            "unrate": s,
            "ma3": ma3,
            "min12": min12,
            "sahm_gap": gap,
            "trigger": trigger.fillna(False),
        }
    )
    prev = df["trigger"].shift(1, fill_value=False)
    df["onset"] = df["trigger"] & (~prev)
    return df


def trigger_onsets(unrate: pd.Series, threshold: float = 0.50) -> pd.DatetimeIndex:
    """Return the month-start dates on which a new Sahm trigger turns on (onsets)."""
    df = sahm_indicator(unrate, threshold=threshold)
    return pd.DatetimeIndex(df.index[df["onset"].to_numpy()])


# ---------------------------------------------------------------------------
# Forward returns around triggers
# ---------------------------------------------------------------------------
def monthly_close(gspc_daily: pd.Series) -> pd.Series:
    """Resample a daily ^GSPC close series to month-start-stamped month-end closes.

    Returns a monthly series indexed by month-START (to align with the unrate
    table), whose value is the *last* close of that calendar month.
    """
    s = gspc_daily.astype(float).sort_index()
    me = s.resample("ME").last().dropna()
    me.index = me.index.to_period("M").to_timestamp(how="start")
    me.name = "gspc_m"
    return me


def forward_return(monthly_px: pd.Series, horizon: int) -> pd.Series:
    """H-month forward simple return of a monthly price series (price-only)."""
    fwd = monthly_px.shift(-horizon) / monthly_px - 1.0
    return fwd.rename(f"fwd_{horizon}m")


def event_study(
    unrate: pd.Series,
    gspc_daily: pd.Series,
    horizon: int = 12,
    exec_lag: int = 1,
    threshold: float = 0.50,
) -> dict:
    """Compare forward S&P returns after Sahm onsets vs the unconditional baseline.

    For each Sahm onset at month m, the entry month is m + ``exec_lag`` (the print
    for month m is released in m+1; we act at that month's close). We then measure
    the ``horizon``-month forward price return from the entry month.

    The honest baseline is the *unconditional* distribution of the same
    horizon-month forward return over all aligned months -- equities drift up, so
    a sell signal must beat the index, not zero.

    Returns a dict with the per-event forward returns, the unconditional mean,
    a Welch t-test (event vs all months), and the difference. (HAC inference on
    the overlapping monthly series is in ``timing_overlay`` / ``hac_tstat``.)
    """
    px = monthly_close(gspc_daily)
    fwd = forward_return(px, horizon)

    onsets = trigger_onsets(unrate, threshold=threshold)
    # Entry month = onset shifted forward by exec_lag months.
    entries = [d + pd.DateOffset(months=exec_lag) for d in onsets]
    # snap each entry to the nearest available monthly index at or after it
    idx = px.index
    event_fwd = []
    event_dates = []
    for e in entries:
        pos = idx.searchsorted(pd.Timestamp(e))
        if pos >= len(idx):
            continue
        d = idx[pos]
        v = fwd.get(d, np.nan)
        if pd.notna(v):
            event_fwd.append(float(v))
            event_dates.append(d)

    event_fwd = np.array(event_fwd, dtype=float)
    uncond = fwd.dropna().to_numpy(dtype=float)
    uncond_mean = float(uncond.mean()) if uncond.size else float("nan")
    event_mean = float(event_fwd.mean()) if event_fwd.size else float("nan")

    if event_fwd.size >= 2 and uncond.size >= 2:
        welch_t, welch_p = scipy_stats.ttest_ind(event_fwd, uncond, equal_var=False)
    else:
        welch_t, welch_p = float("nan"), float("nan")

    return {
        "horizon": horizon,
        "exec_lag": exec_lag,
        "threshold": threshold,
        "n_events": int(event_fwd.size),
        "event_dates": [pd.Timestamp(d) for d in event_dates],
        "event_fwd": event_fwd,
        "event_mean": round(event_mean, 4),
        "uncond_mean": round(uncond_mean, 4),
        "diff": round(event_mean - uncond_mean, 4),
        "welch_t": round(float(welch_t), 3),
        "welch_p": round(float(welch_p), 4),
    }


# ---------------------------------------------------------------------------
# The tradable timing overlay
# ---------------------------------------------------------------------------
def timing_overlay(
    unrate: pd.Series,
    gspc_daily: pd.Series,
    exec_lag: int = 1,
    threshold: float = 0.50,
    cost_bps: float = 10.0,
    stay_out_months: int = 12,
) -> dict:
    """Binary Sahm timing overlay vs buy-and-hold (price-only, net of costs).

    Rule: be OUT of the S&P for ``stay_out_months`` months once a Sahm trigger
    onset fires (acted on with ``exec_lag`` months delay), else be long. Each
    switch (in or out) pays ``cost_bps`` one-way on NAV.

    Returns annualized CAGR, vol, Sharpe-like ratio, max drawdown for both the
    overlay and buy-and-hold, plus the monthly return series and the position
    series, and the HAC t-stat on the monthly active return (overlay - B&H).
    """
    px = monthly_close(gspc_daily)
    ret = px.pct_change().dropna()

    onsets = trigger_onsets(unrate, threshold=threshold)
    entries = [d + pd.DateOffset(months=exec_lag) for d in onsets]

    pos = pd.Series(1.0, index=ret.index)  # 1 = long, 0 = out
    for e in entries:
        e = pd.Timestamp(e)
        out_end = e + pd.DateOffset(months=stay_out_months)
        mask = (ret.index >= e) & (ret.index < out_end)
        pos.loc[mask] = 0.0

    # Trades: position changes -> one-way cost on NAV.
    switches = pos.diff().abs().fillna(pos.iloc[0]).gt(0)
    cost = switches.astype(float) * (cost_bps / 1e4)

    overlay_ret = pos * ret - cost
    bah_ret = ret

    def _stats(r: pd.Series) -> dict:
        r = r.dropna()
        n = len(r)
        if n == 0:
            return {"cagr": float("nan"), "vol": float("nan"), "sharpe": float("nan"), "mdd": float("nan")}
        cum = (1 + r).prod()
        yrs = n / 12.0
        cagr = cum ** (1 / yrs) - 1 if cum > 0 else -1.0
        vol = r.std(ddof=1) * np.sqrt(12)
        sharpe = (r.mean() * 12) / vol if vol > 0 else float("nan")
        eq = (1 + r).cumprod()
        mdd = float((eq / eq.cummax() - 1).min())
        return {"cagr": float(cagr), "vol": float(vol), "sharpe": float(sharpe), "mdd": mdd}

    ov = _stats(overlay_ret)
    bh = _stats(bah_ret)

    active = (overlay_ret - bah_ret).dropna()
    hac_t = hac_tstat(active.to_numpy())

    return {
        "exec_lag": exec_lag,
        "threshold": threshold,
        "cost_bps": cost_bps,
        "stay_out_months": stay_out_months,
        "n_switches": int(switches.sum()),
        "overlay": {k: round(v, 4) for k, v in ov.items()},
        "bah": {k: round(v, 4) for k, v in bh.items()},
        "active_mean_ann": round(float(active.mean() * 12), 4),
        "active_hac_t": round(float(hac_t), 3),
        "overlay_ret": overlay_ret,
        "bah_ret": bah_ret,
        "position": pos,
    }


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------
def hac_tstat(x: np.ndarray, lags: int | None = None) -> float:
    """Newey-West (HAC) t-stat for the mean of a series against zero.

    Uses Bartlett kernel weights with ``lags`` lags (default floor(4*(n/100)^(2/9))).
    Returns ``mean / HAC_se``.
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = x.size
    if n < 3:
        return float("nan")
    if lags is None:
        lags = int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0)))
        lags = max(lags, 1)
    mu = x.mean()
    e = x - mu
    gamma0 = np.dot(e, e) / n
    var = gamma0
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        cov = np.dot(e[k:], e[:-k]) / n
        var += 2.0 * w * cov
    se = np.sqrt(var / n)
    if se == 0:
        return float("nan")
    return float(mu / se)


def block_permutation_p(
    unrate: pd.Series,
    gspc_daily: pd.Series,
    horizon: int = 12,
    exec_lag: int = 1,
    threshold: float = 0.50,
    n_perm: int = 5_000,
    seed: int = 268,
) -> dict:
    """Permutation test: is the post-trigger mean forward return unusually low?

    We keep the number of events fixed and draw that many random entry months
    (uniformly from the valid forward-return window), recomputing the mean
    forward return each time. The p-value is the fraction of random draws whose
    mean is <= the observed event mean (a one-sided 'sell' test: triggers should
    precede *low* forward returns).
    """
    es = event_study(unrate, gspc_daily, horizon=horizon, exec_lag=exec_lag, threshold=threshold)
    px = monthly_close(gspc_daily)
    fwd = forward_return(px, horizon).dropna()
    pool = fwd.to_numpy(dtype=float)
    k = es["n_events"]
    obs = es["event_mean"]
    if k == 0 or pool.size < k:
        return {"perm_p": float("nan"), "obs_mean": obs, "n_events": k}

    rng = np.random.default_rng(seed)
    means = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        idx = rng.choice(pool.size, size=k, replace=False)
        means[i] = pool[idx].mean()
    perm_p = float((means <= obs).mean())
    return {
        "perm_p": round(perm_p, 4),
        "obs_mean": obs,
        "n_events": k,
        "perm_means": means,
    }


# ---------------------------------------------------------------------------
# Summarize -- compact R-dict for notebooks / results.md
# ---------------------------------------------------------------------------
def summarize(
    unrate: pd.Series,
    gspc_daily: pd.Series,
    horizon: int = 12,
    exec_lag: int = 1,
    threshold: float = 0.50,
    n_perm: int = 5_000,
    seed: int = 268,
) -> dict:
    """One-stop summary dict for the real (or synthetic-as-real) series.

    Mirrors the R = dict(...) in build_notebooks.py and docs/results.md.
    """
    es = event_study(unrate, gspc_daily, horizon=horizon, exec_lag=exec_lag, threshold=threshold)
    perm = block_permutation_p(
        unrate, gspc_daily, horizon=horizon, exec_lag=exec_lag,
        threshold=threshold, n_perm=n_perm, seed=seed,
    )
    ov = timing_overlay(unrate, gspc_daily, exec_lag=exec_lag, threshold=threshold)

    return {
        "n_events": es["n_events"],
        "horizon": horizon,
        "event_mean_pct": round(es["event_mean"] * 100, 1),
        "uncond_mean_pct": round(es["uncond_mean"] * 100, 1),
        "diff_pct": round(es["diff"] * 100, 1),
        "welch_t": es["welch_t"],
        "welch_p": es["welch_p"],
        "perm_p": perm["perm_p"],
        "overlay_cagr_pct": round(ov["overlay"]["cagr"] * 100, 1),
        "bah_cagr_pct": round(ov["bah"]["cagr"] * 100, 1),
        "overlay_mdd_pct": round(ov["overlay"]["mdd"] * 100, 1),
        "bah_mdd_pct": round(ov["bah"]["mdd"] * 100, 1),
        "active_hac_t": ov["active_hac_t"],
        "n_switches": ov["n_switches"],
    }
