"""The curve-fitting engine — Study 348.

A moving-average crossover goes long when a fast MA is above a slow MA, flat otherwise.
There is nothing special about MA crossovers here; they are just a clean two-parameter
family with a big, plausible grid — the perfect vehicle for the *real* subject of the
study: **parameter optimisation**.

The protocol (announced before we run it):

1. Split the series into an in-sample (IS) slice and an out-of-sample (OOS) slice.
2. Sweep every (fast, slow) pair on a grid, score each by its IS Sharpe.
3. Crown the IS winner.
4. Run *that* winner, untouched, on the OOS slice.
5. Compare the IS-winner's IS Sharpe to its OOS Sharpe. On a tape with no real edge the
   IS Sharpe is large by selection and the OOS Sharpe reverts to ~0 — the **mirage**. On a
   tape with a planted, harvestable trend the OOS Sharpe stays meaningfully positive.

The honesty discipline (the same one the whole desk uses, so the *demo* is itself clean):

- **One execution lag, documented exactly.** The crossover sign known at the close of day
  ``t`` earns the return of day ``t+1`` — a single ``shift(1)``, applied once in
  ``crossover_returns``. Docstrings state the *effective* lag.
- **Costs one-way × NAV.** A round-trip on a position flip is two one-way trades; we charge
  ``cost_bps`` per one-way leg on the change in position, i.e. one-way turnover × NAV.
- **Excess-vs-excess.** The rule is long-or-flat (in cash when flat), so its return stream
  is compared to *buy-and-hold's excess over cash* on the same footing in the headline; the
  HAC test runs on the strategy's own net return (excess-of-cash, ``rf`` defaults to 0 for
  the offline synthetic core and is supplied for the real tape).
- **Block bootstrap** for the OOS-Sharpe CI; **HAC (Newey-West)** *t* on the OOS mean.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# The crossover rule (one execution lag, one-way costs)
# ---------------------------------------------------------------------------
def crossover_position(prices: pd.Series, fast: int, slow: int) -> pd.Series:
    """Long-or-flat position from a fast/slow simple-MA crossover.

    Position is 1.0 when the ``fast``-day MA is strictly above the ``slow``-day MA at the
    close, else 0.0. The position is the signal *as known at the close of day t*; the lag
    that turns it into a tradable book lives in ``crossover_returns``.
    """
    if fast >= slow:
        raise ValueError(f"fast ({fast}) must be < slow ({slow})")
    ma_fast = prices.rolling(fast).mean()
    ma_slow = prices.rolling(slow).mean()
    pos = (ma_fast > ma_slow).astype(float)
    pos[ma_slow.isna()] = np.nan  # undefined until the slow MA exists
    return pos


def crossover_returns(
    prices: pd.Series,
    fast: int,
    slow: int,
    cost_bps: float = 0.0,
    rf_daily: float = 0.0,
) -> pd.Series:
    """Net daily return of the long/flat crossover, in *excess of cash*.

    Execution lag (stated exactly): the position known at the close of day ``t`` (from
    ``crossover_position``) earns the asset's return of day ``t+1`` — a single ``shift(1)``.
    When flat, the book earns 0 (cash); reporting in excess-of-cash subtracts ``rf_daily``
    from the held leg, so a flat day is exactly 0 excess and a held day is the asset's
    excess return. Costs: ``cost_bps`` per one-way leg on ``|Δposition|`` (one-way turnover
    × NAV), charged on the day the position changes.
    """
    asset_ret = prices.pct_change()
    pos = crossover_position(prices, fast, slow)
    held = pos.shift(1)  # the ONE execution lag

    gross = held * (asset_ret - rf_daily)             # excess-of-cash on the held leg
    turnover = held.fillna(0.0).diff().abs()          # one-way change in position
    cost = turnover * (cost_bps * 1e-4)
    net = (gross - cost).dropna()
    return net


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def sharpe(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    """Annualised Sharpe of a (already excess-of-cash) return series."""
    r = returns.dropna()
    if len(r) < 2:
        return float("nan")
    sd = r.std(ddof=1)
    if sd <= 0:
        return float("nan")
    return float(r.mean() / sd * np.sqrt(periods_per_year))


def cagr(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    r = returns.dropna()
    if r.empty:
        return float("nan")
    total = float((1.0 + r).prod())
    yrs = len(r) / periods_per_year
    return total ** (1.0 / yrs) - 1.0 if (yrs > 0 and total > 0) else float("nan")


def hac_tstat(returns: pd.Series) -> dict:
    """Newey-West HAC t-stat that the mean daily (excess) return differs from zero."""
    x = returns.dropna().to_numpy(dtype=float)
    n = x.size
    if n < 3:
        return {"mean": float("nan"), "tstat": float("nan"), "n": n}
    mu = x.mean()
    e = x - mu
    lags = max(1, int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))))
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        wk = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * wk * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    return {"mean": float(mu), "tstat": float(mu / se) if se > 0 else float("nan"), "n": n}


def block_bootstrap_sharpe_ci(
    returns: pd.Series, block: int = 21, n_boot: int = 2000, seed: int = 348,
    alpha: float = 0.05, periods_per_year: int = TRADING_DAYS,
) -> tuple[float, float]:
    """Circular block-bootstrap CI for the annualised Sharpe of a return series.

    Block resampling preserves the autocorrelation i.i.d. resampling would destroy.
    """
    x = returns.dropna().to_numpy(dtype=float)
    n = x.size
    if n < block + 1:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    out = np.empty(n_boot)
    sq = np.sqrt(periods_per_year)
    for b in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel() % n
        samp = x[idx][:n]
        sd = samp.std(ddof=1)
        out[b] = samp.mean() / sd * sq if sd > 0 else 0.0
    return (float(np.percentile(out, 100 * alpha / 2)),
            float(np.percentile(out, 100 * (1 - alpha / 2))))


# ---------------------------------------------------------------------------
# The curve-fitting protocol
# ---------------------------------------------------------------------------
def default_grid(
    fasts: tuple[int, ...] = (5, 10, 15, 20, 30, 40, 50, 75, 100, 125, 150, 200),
    slows: tuple[int, ...] = (20, 30, 50, 75, 100, 125, 150, 175, 200, 225, 250),
) -> list[tuple[int, int]]:
    """All valid (fast, slow) pairs with fast < slow — the parameter grid we sweep."""
    return [(f, s) for f, s in itertools.product(fasts, slows) if f < s]


def sweep(
    prices: pd.Series, grid: list[tuple[int, int]],
    cost_bps: float = 0.0, rf_daily: float = 0.0,
) -> pd.DataFrame:
    """Score every grid pair by its Sharpe on ``prices``. Returns a frame sorted by Sharpe.

    Columns: ``fast``, ``slow``, ``sharpe``, ``cagr``, ``n``. The top row is the IS winner.
    """
    rows = []
    for f, s in grid:
        r = crossover_returns(prices, f, s, cost_bps=cost_bps, rf_daily=rf_daily)
        rows.append({"fast": f, "slow": s, "sharpe": sharpe(r), "cagr": cagr(r), "n": len(r)})
    df = pd.DataFrame(rows).sort_values("sharpe", ascending=False).reset_index(drop=True)
    return df


def split(prices: pd.Series, frac_is: float = 0.5) -> tuple[pd.Series, pd.Series]:
    """Chronological split into (in-sample, out-of-sample) by fraction of observations."""
    k = int(len(prices) * frac_is)
    return prices.iloc[:k], prices.iloc[k:]


def curve_fit_experiment(
    prices: pd.Series,
    grid: list[tuple[int, int]] | None = None,
    frac_is: float = 0.5,
    cost_bps: float = 0.0,
    rf_daily: float = 0.0,
    seed: int = 348,
) -> dict:
    """Run the full optimise-then-validate experiment on one price series.

    Sweeps the grid on the IS slice, crowns the IS-best (fast, slow), then runs that pair
    untouched on the OOS slice. Returns a bundle with the winner, its IS and OOS Sharpe (the
    headline gap), the OOS HAC test and block-bootstrap Sharpe CI, and the *median* OOS
    Sharpe across the whole grid (the honest "average parameterisation" yardstick).

    Keys: ``winner`` (fast, slow), ``is_sharpe``, ``oos_sharpe``, ``shrinkage`` (IS−OOS),
    ``oos_test`` (HAC), ``oos_ci`` (Sharpe 95% block-bootstrap), ``grid_oos_median_sharpe``,
    ``is_table`` (full IS sweep), ``n_grid``.
    """
    if grid is None:
        grid = default_grid()
    is_px, oos_px = split(prices, frac_is)

    is_table = sweep(is_px, grid, cost_bps=cost_bps, rf_daily=rf_daily)
    winner = (int(is_table.iloc[0]["fast"]), int(is_table.iloc[0]["slow"]))
    is_sharpe = float(is_table.iloc[0]["sharpe"])

    oos_ret = crossover_returns(oos_px, winner[0], winner[1],
                                cost_bps=cost_bps, rf_daily=rf_daily)
    oos_sharpe = sharpe(oos_ret)

    # The honest yardstick: how the *whole grid* fares OOS (no cherry-picking).
    oos_sweep = sweep(oos_px, grid, cost_bps=cost_bps, rf_daily=rf_daily)
    grid_oos_median = float(oos_sweep["sharpe"].median())

    return {
        "winner": winner,
        "is_sharpe": is_sharpe,
        "oos_sharpe": float(oos_sharpe),
        "shrinkage": float(is_sharpe - oos_sharpe),
        "oos_test": hac_tstat(oos_ret),
        "oos_ci": block_bootstrap_sharpe_ci(oos_ret, seed=seed),
        "grid_oos_median_sharpe": grid_oos_median,
        "is_table": is_table,
        "oos_ret": oos_ret,
        "n_grid": len(grid),
    }


def alpha_vs_buyhold(
    prices: pd.Series, fast: int, slow: int,
    cost_bps: float = 0.0, rf_daily: float = 0.0,
) -> dict:
    """Is the crossover's return anything but the market it was long in?

    Races the long/flat crossover against buy-and-hold of the same asset on the *same*
    days, both in excess of cash, and HAC-tests the daily difference. A long-or-flat timing
    rule spends most days fully invested, so a positive OOS Sharpe can be almost entirely
    the asset's beta — this strips it out. Returns the two Sharpes, the average exposure,
    and the HAC test of (strategy − buy-and-hold).
    """
    strat = crossover_returns(prices, fast, slow, cost_bps=cost_bps, rf_daily=rf_daily)
    bh = (prices.pct_change() - rf_daily).reindex(strat.index)
    diff = (strat - bh).dropna()
    pos = crossover_position(prices, fast, slow).shift(1)
    return {
        "strat_sharpe": sharpe(strat),
        "buyhold_sharpe": sharpe(bh),
        "avg_exposure": float(pos.reindex(strat.index).mean()),
        "alpha_test": hac_tstat(diff),
        "alpha_ann": float(hac_tstat(diff)["mean"] * TRADING_DAYS),
    }


def grid_size_sweep(
    prices: pd.Series, grid_sizes: tuple[int, ...] = (4, 9, 16, 36, 64),
    frac_is: float = 0.5, cost_bps: float = 0.0, rf_daily: float = 0.0,
    seed: int = 348,
) -> pd.DataFrame:
    """How the IS−OOS Sharpe gap grows with the number of parameters searched.

    For each target grid size we build a nested grid of roughly that many pairs and run the
    experiment. On a no-edge tape the IS Sharpe inflates with grid size while the OOS Sharpe
    stays at ~0 — the mirage gets louder the more you tune. Returns one row per grid size.
    """
    fasts_all = [5, 10, 15, 20, 25, 30, 40, 50, 60, 75, 100, 125, 150, 175, 200]
    slows_all = [20, 30, 40, 50, 60, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300]
    rng = np.random.default_rng(seed)
    rows = []
    for target in grid_sizes:
        # Build a grid of ~target valid pairs by widening the candidate windows.
        side = max(2, int(np.ceil(np.sqrt(target * 2))))
        fasts = tuple(fasts_all[:side])
        slows = tuple(slows_all[:side])
        grid = [(f, s) for f in fasts for s in slows if f < s]
        if len(grid) > target:
            keep = rng.choice(len(grid), size=target, replace=False)
            grid = [grid[i] for i in sorted(keep)]
        res = curve_fit_experiment(prices, grid=grid, frac_is=frac_is,
                                   cost_bps=cost_bps, rf_daily=rf_daily, seed=seed)
        rows.append({
            "n_params": len(grid),
            "is_sharpe": res["is_sharpe"],
            "oos_sharpe": res["oos_sharpe"],
            "shrinkage": res["shrinkage"],
        })
    return pd.DataFrame(rows)
