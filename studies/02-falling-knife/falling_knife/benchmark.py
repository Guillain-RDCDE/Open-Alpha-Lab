"""The yardstick: does buying the dip beat buying a *random* day?

This is the single most important module in the project, and the one most "buy
the dip" backtests skip. The Nasdaq-100 rises structurally over decades, so
*any* "buy and hold N days" rule makes money on average. A positive backtest
therefore proves nothing on its own.

The honest question is conditional vs unconditional:

    conditional   = average forward return *after a -3% event*
    unconditional = average forward return *on any random day*  (the null)
    excess        = conditional - unconditional

If ``excess`` is ~0, the "strategy" is just harvesting the market's drift and you
might as well have bought any day (and saved yourself the drama). We quantify
significance with a **permutation test**: resample random baskets of the same
size as the event set, build the null distribution of mean forward returns, and
read off where the observed conditional mean falls.

Caveat we surface rather than hide: -3% events *cluster* (2008, 2020, 2022), so
the effective number of independent draws is far smaller than the raw count. The
iid permutation here slightly *overstates* significance; the block bootstrap in
:mod:`falling_knife.robustness` stress-tests that.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def forward_returns(ohlc: pd.DataFrame, horizon: int) -> np.ndarray:
    """Close-to-close forward return at ``horizon`` days for every bar.

    ``NaN`` for the last ``horizon`` bars (no full forward window).
    """
    close = ohlc["Close"].to_numpy()
    fwd = np.full(len(close), np.nan)
    if horizon < len(close):
        fwd[:-horizon] = close[horizon:] / close[:-horizon] - 1.0
    return fwd


def _permutation_pvalue(
    fwd: np.ndarray,
    event_mask: np.ndarray,
    n_iter: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """One- and two-sided p-values for the event mean vs random baskets.

    Returns ``(p_greater, p_two_sided)`` where ``p_greater`` is the fraction of
    random baskets whose mean forward return is >= the observed event mean.
    """
    valid = ~np.isnan(fwd)
    pool = fwd[valid]
    k = int(event_mask[valid].sum())
    if k == 0 or k > len(pool):
        return float("nan"), float("nan")

    observed = pool[event_mask[valid]].mean() if event_mask[valid].any() else np.nan
    uncond = pool.mean()

    draws = np.empty(n_iter)
    for i in range(n_iter):
        draws[i] = rng.choice(pool, size=k, replace=False).mean()

    p_greater = float((draws >= observed).mean())
    p_two = float((np.abs(draws - uncond) >= abs(observed - uncond)).mean())
    return p_greater, p_two


def conditional_vs_unconditional(
    ohlc: pd.DataFrame,
    signal: pd.Series,
    horizons=(1, 3, 5, 10, 20),
    n_iter: int = 2000,
    seed: int = 0,
) -> pd.DataFrame:
    """Compare event-conditional forward returns to the random-day baseline.

    For each horizon returns: ``n_events, mean_cond, mean_uncond, excess,
    pct_pos_cond, pct_pos_uncond, p_greater, p_two_sided``.

    ``p_greater`` is the headline number: the probability a random basket of the
    same size beats the dip-buyer. Small (< 0.05) = the dip genuinely adds
    something beyond market drift; near 0.5 = it does not.
    """
    rng = np.random.default_rng(seed)
    sig = signal.reindex(ohlc.index).fillna(False).to_numpy()

    cols = ["horizon", "n_events", "mean_cond", "mean_uncond", "excess",
            "pct_pos_cond", "pct_pos_uncond", "p_greater", "p_two_sided"]
    rows = []
    for h in horizons:
        fwd = forward_returns(ohlc, h)
        valid = ~np.isnan(fwd)
        ev = sig & valid
        pool = fwd[valid]
        cond = fwd[ev]
        if cond.size == 0:
            continue
        p_greater, p_two = _permutation_pvalue(fwd, sig, n_iter, rng)
        rows.append(
            {
                "horizon": h,
                "n_events": int(ev.sum()),
                "mean_cond": float(cond.mean()),
                "mean_uncond": float(pool.mean()),
                "excess": float(cond.mean() - pool.mean()),
                "pct_pos_cond": float((cond > 0).mean()),
                "pct_pos_uncond": float((pool > 0).mean()),
                "p_greater": p_greater,
                "p_two_sided": p_two,
            }
        )
    if not rows:
        return pd.DataFrame(columns=cols).set_index("horizon")
    return pd.DataFrame(rows).set_index("horizon")
