"""Robustness: separating a real edge from luck and data-mining.

Three traps sink most "buy the dip" backtests, and this module addresses each:

1. Regime dependence — dip-buying thrives under a "Fed put" (post-2009, post-2020)
   and dies in a grind-down bear (2000-2002). :func:`regime_split` reports the
   strategy era by era so a single bull-market run can't masquerade as a law.

2. Tiny effective sample — -3% days cluster, so 60 raw events might be ~10
   independent episodes. :func:`block_bootstrap_excess` resamples *blocks* of
   days (preserving the clustering) to get an honest confidence interval on the
   conditional-vs-unconditional excess return.

3. Multiple testing — scan 40 (trigger x exit) cells and the best Sharpe is
   inflated by selection. :func:`deflated_sharpe` discounts the top Sharpe for
   the number of trials, and :func:`split_sample` checks the winner out-of-sample.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .benchmark import forward_returns


# --- Standard-normal helpers (kept dependency-free; no scipy needed) -------- #

def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    """Inverse normal CDF via Acklam's rational approximation (|err| < 1.2e-9)."""
    if not 0.0 < p < 1.0:
        return float("-inf") if p <= 0.0 else float("inf")
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


# Hand-labelled NDX macro regimes. Boundaries are approximate and deliberately
# coarse — the point is "does it work outside the obvious bull runs?".
REGIMES = [
    ("Dot-com bust", "2000-01-01", "2002-12-31"),
    ("Mid-2000s bull", "2003-01-01", "2007-10-31"),
    ("GFC", "2007-11-01", "2009-03-31"),
    ("QE bull", "2009-04-01", "2020-02-19"),
    ("COVID crash", "2020-02-20", "2020-04-30"),
    ("Post-COVID bull", "2020-05-01", "2021-12-31"),
    ("2022 bear", "2022-01-01", "2022-12-31"),
    ("Recent", "2023-01-01", "2099-12-31"),
]


def regime_split(daily: pd.Series) -> pd.DataFrame:
    """Per-regime return and Sharpe of a strategy's daily-return series.

    Pass ``BacktestResult.daily``. Returns one row per regime that overlaps the
    sample, so you can see whether the equity curve is one lucky era or a stable
    effect.
    """
    rows = []
    for name, start, end in REGIMES:
        seg = daily[(daily.index >= start) & (daily.index <= end)]
        active = seg[seg != 0.0]
        if seg.empty:
            continue
        total = float((1.0 + seg).prod() - 1.0)
        sharpe = (float(active.mean() / active.std(ddof=1) * np.sqrt(252))
                  if len(active) > 1 and active.std(ddof=1) > 0 else np.nan)
        rows.append({
            "regime": name,
            "start": start,
            "total_return": total,
            "sharpe": sharpe,
            "active_days": int((seg != 0.0).sum()),
        })
    return pd.DataFrame(rows).set_index("regime")


def block_bootstrap_excess(
    ohlc: pd.DataFrame,
    signal: pd.Series,
    horizon: int = 5,
    block: int = 20,
    n_iter: int = 2000,
    seed: int = 0,
) -> dict:
    """Block-bootstrap confidence interval for the conditional excess return.

    Resamples contiguous blocks of (forward-return, is-event) pairs so the
    clustering of -3% days is preserved, then recomputes the excess
    (mean_cond - mean_uncond) on each synthetic history. Wide / zero-straddling
    intervals mean the apparent edge is within sampling noise.

    Returns ``mean, ci_low, ci_high, p_excess_le_0``.
    """
    rng = np.random.default_rng(seed)
    fwd = forward_returns(ohlc, horizon)
    sig = signal.reindex(ohlc.index).fillna(False).to_numpy()
    valid = ~np.isnan(fwd)
    fwd, sig = fwd[valid], sig[valid]
    n = len(fwd)
    if n == 0 or sig.sum() == 0:
        return {"mean": np.nan, "ci_low": np.nan, "ci_high": np.nan, "p_excess_le_0": np.nan}

    n_blocks = int(np.ceil(n / block))
    excesses = np.empty(n_iter)
    for i in range(n_iter):
        starts = rng.integers(0, n, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + block) % n for s in starts])[:n]
        f, s = fwd[idx], sig[idx]
        if s.sum() == 0:
            excesses[i] = np.nan
            continue
        excesses[i] = f[s].mean() - f.mean()

    excesses = excesses[~np.isnan(excesses)]
    return {
        "mean": float(np.mean(excesses)),
        "ci_low": float(np.percentile(excesses, 2.5)),
        "ci_high": float(np.percentile(excesses, 97.5)),
        "p_excess_le_0": float((excesses <= 0).mean()),
    }


def deflated_sharpe(best_sharpe: float, n_trials: int, n_obs: int) -> float:
    """Probabilistic Sharpe ratio deflated for the number of strategies tried.

    A simplified Bailey-López de Prado deflated Sharpe: the probability that the
    best observed (annualised) Sharpe over ``n_trials`` independent configs would
    NOT have arisen by chance under a true Sharpe of zero. ``n_obs`` is the number
    of return observations behind the Sharpe (e.g. trading days).

    Values near 1 = the winner is unlikely to be a fluke; near 0.5 or below = the
    "edge" is consistent with picking the luckiest of many coin-flips.
    """
    if n_trials < 1 or n_obs < 2 or not np.isfinite(best_sharpe):
        return np.nan
    # Expected max of n_trials standard normals (the data-mining bar to clear).
    euler = 0.5772156649
    z = _norm_ppf(1 - 1.0 / n_trials) * (1 - euler) + _norm_ppf(1 - 1.0 / (n_trials * np.e)) * euler
    sr_daily = best_sharpe / np.sqrt(252)              # de-annualise
    expected_max_daily = z / np.sqrt(n_obs)            # SE of a zero-true Sharpe
    deflated = (sr_daily - expected_max_daily) * np.sqrt(n_obs)
    return float(_norm_cdf(deflated))


def split_sample(
    ohlc: pd.DataFrame,
    frac: float = 0.6,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological in-sample / out-of-sample split (no shuffling).

    Use it to pick the best (trigger, exit) on the first ``frac`` of history and
    then confirm it on the held-out tail. An edge that evaporates out-of-sample
    was data-mining.
    """
    cut = int(len(ohlc) * frac)
    return ohlc.iloc[:cut].copy(), ohlc.iloc[cut:].copy()


def oos_best_cell(
    ohlc: pd.DataFrame,
    trigger_signals: dict,
    exit_grid: list,
    costs=None,
    frac: float = 0.6,
    cooldown: int = 20,
) -> dict:
    """Pick the best (trigger x exit) in-sample, then judge it out-of-sample.

    This is the honest counterpart to :func:`falling_knife.backtest.family_scan`:
    the scan *finds* the prettiest cell, this *tests* whether that choice holds up
    on data it was never selected on. A Sharpe that collapses (or flips sign) from
    IS to OOS is the fingerprint of data-mining.

    ``trigger_signals`` maps name -> raw boolean Series over the FULL sample (we
    re-slice and re-debounce per split internally). Returns the winning cell plus
    its in-sample and out-of-sample stats.
    """
    from .backtest import run, CostModel
    from .triggers import first_crossings

    costs = costs or CostModel()
    is_ohlc, oos_ohlc = split_sample(ohlc, frac)

    # 1) Select the best cell IN-SAMPLE only.
    best = None
    for tname, raw in trigger_signals.items():
        raw_is = raw.reindex(is_ohlc.index)
        for rule in exit_grid:
            cd = max(cooldown, rule.max_hold)
            sig = first_crossings(raw_is, cooldown=cd)
            res = run(is_ohlc, sig, rule, costs)
            sh = res.stats["sharpe"]
            if sh == sh and (best is None or sh > best["is_sharpe"]):
                best = {"trigger": tname, "rule": rule, "is_sharpe": sh,
                        "is_stats": res.stats}

    if best is None:
        return {"selected": None}

    # 2) Apply that EXACT cell OUT-OF-SAMPLE.
    raw_oos = trigger_signals[best["trigger"]].reindex(oos_ohlc.index)
    cd = max(cooldown, best["rule"].max_hold)
    sig_oos = first_crossings(raw_oos, cooldown=cd)
    res_oos = run(oos_ohlc, sig_oos, best["rule"], costs)

    return {
        "trigger": best["trigger"],
        "exit": best["rule"].label(),
        "is_sharpe": best["is_sharpe"],
        "is_cagr": best["is_stats"]["cagr"],
        "is_trades": best["is_stats"]["n_trades"],
        "oos_sharpe": res_oos.stats["sharpe"],
        "oos_cagr": res_oos.stats["cagr"],
        "oos_trades": res_oos.stats["n_trades"],
        "verdict": ("holds up" if res_oos.stats["sharpe"] == res_oos.stats["sharpe"]
                    and res_oos.stats["sharpe"] > 0.5 * best["is_sharpe"]
                    else "decays out-of-sample (likely data-mining)"),
    }
