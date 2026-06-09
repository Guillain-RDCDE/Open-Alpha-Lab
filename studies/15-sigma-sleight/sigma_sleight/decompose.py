"""The teardown engine — turn a price tape into the numbers that earn the verdict.

Four questions, four tools, in the order an honest investigation runs them:

1. **Does the σ-transform move any signal, or is it a relabel?**
   :func:`monotone_check` confirms ``σ(RSI)`` is strictly increasing at a fixed length (so a
   σ-zone is an order-preserving rename of an RSI level), and :func:`crossing_identity` runs the
   same mean-reversion strategy through both the σ band and its implied constant RSI band and
   measures the difference in trades — which, for the framework's own arithmetic, is **exactly
   zero**. Within a length, "adaptive" adds no information; it picks a constant.

2. **Are the famous cheat-sheet levels calibration or arithmetic?**
   :func:`zone_arithmetic` reproduces the published zone table from the formula and pins the
   marketing landmark (``RSI(14)=70`` ⟺ ``+1.53σ``), showing the levels are a deterministic
   function of ``n`` that compresses toward 50 — no market data touched.

3. **Does σ-calibration actually beat a sensibly-chosen constant?**
   :func:`strategy_compare` runs a cost-charged horse race per length: fixed 70/30, the adaptive-σ
   band, and a per-length *re-optimised* constant. The adaptive band ties some constant by
   construction (result 1); the live question is whether σ-calibration beats simply re-optimising
   two numbers — and whether either beats fixed 70/30 out of the box.

4. **Does Rescaled long-RSI add anything over native short RSI?**
   :func:`rescale_increment` regresses forward returns on native ``RSI(target)`` and then on
   native + rescaled long RSI, reporting the incremental information coefficient. On a
   single-horizon tape the increment is ~0 (the baked-in null); the real run asks if markets,
   with their multiple horizons, differ.

Pure NumPy/pandas; deterministic. The robust-inference and as-of machinery for the *real* run
lives in :mod:`quantlab`; this module is the offline core the tests pin.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import signals
from .rsi import (ZONE_SIGMA, adaptive_zone_levels, rescaled_rsi, rsi_to_sigma,
                  sigma_to_rsi, wilder_rsi, zone_levels_table)

_TRADING_DAYS = 252.0


# --------------------------------------------------------------------------- #
# 1 · The σ-transform is a monotone relabel
# --------------------------------------------------------------------------- #

def monotone_check(length: int, n_grid: int = 4001) -> dict:
    """Is ``σ(RSI)`` strictly increasing at this ``length`` (so a σ-zone is an order-preserving
    rename of an RSI level)? Sweep RSI across (0, 100), confirm σ rises monotonically and that the
    round-trip ``σ → RSI`` recovers the input. Returns the monotonicity flag and the worst
    round-trip error — the formal basis for "the transform cannot move a crossing"."""
    grid = np.linspace(0.5, 99.5, n_grid)
    sig = rsi_to_sigma(grid, length)
    back = sigma_to_rsi(sig, length)
    diffs = np.diff(sig)
    return {
        "length": int(length),
        "strictly_increasing": bool(np.all(diffs > 0)),
        "min_step": float(diffs.min()),
        "max_roundtrip_err": float(np.max(np.abs(back - grid))),
    }


def crossing_identity(close: pd.Series, length: int, lower_sigma: float,
                      exit_sigma: float = 0.0) -> dict:
    """Run the mean-reversion strategy through the σ band and its implied constant RSI band, and
    measure the gap in trades.

    The σ band (``enter σ < lower_sigma``, ``exit σ > exit_sigma``) and the constant band it maps
    to (``sigma_to_rsi`` of each) are, by monotonicity, the *same events*. This computes both
    position series and returns the max absolute position difference (the headline: **0.0**), the
    implied constant ``(lower, exit)`` RSI levels, and the trade count — so "adaptive == constant"
    is a measured fact, not an assertion.
    """
    rsi = wilder_rsi(close, length)
    pos_sigma = signals.sigma_band_positions(rsi, length, lower_sigma, exit_sigma)
    lower_rsi, exit_rsi = signals.implied_rsi_band(length, lower_sigma, exit_sigma)
    pos_const = signals.rsi_band_positions(rsi, lower_rsi, exit_rsi)
    diff = float((pos_sigma - pos_const).abs().max())
    return {
        "length": int(length),
        "lower_sigma": float(lower_sigma),
        "implied_lower_rsi": float(lower_rsi),
        "implied_exit_rsi": float(exit_rsi),
        "max_position_diff": diff,
        "identical": bool(diff == 0.0),
        "n_entries": int((pos_const.diff() > 0).sum()),
    }


# --------------------------------------------------------------------------- #
# 2 · The cheat-sheet is arithmetic, not calibration
# --------------------------------------------------------------------------- #

def zone_arithmetic(lengths=(2, 5, 14, 50, 200)) -> dict:
    """Reproduce the published adaptive-zone table from the formula and pin the marketing landmark.

    Returns the per-length zone table (:func:`zone_levels_table`), the σ value that ``RSI(14)=70``
    maps to (should be ~+1.53), and a monotone-compression flag: as length grows, every σ
    landmark's RSI level falls toward 50. No market data — these numbers are fixed by ``n`` alone.
    """
    table = zone_levels_table(lengths)
    sigma_at_70_14 = float(rsi_to_sigma(70.0, 14))
    # Each σ-zone's RSI level should be (weakly) decreasing as length increases — compression.
    compresses = bool((table.diff().dropna().to_numpy() <= 1e-9).all())
    return {
        "table": table,
        "sigma_at_rsi70_len14": sigma_at_70_14,
        "rsi70_is_1p53_sigma": bool(abs(sigma_at_70_14 - 1.53) < 0.01),
        "compresses_toward_50": compresses,
    }


# --------------------------------------------------------------------------- #
# 3 · The cost-charged horse race
# --------------------------------------------------------------------------- #

def _strategy_stats(close: pd.Series, pos: pd.Series, cost_bps: float) -> dict:
    """Net annualised return, Sharpe and trade count of a long/flat position, charged ``cost_bps``
    per unit of turnover (entry + exit each pay once)."""
    ret = close.pct_change().fillna(0.0)
    turn = pos.diff().abs().fillna(pos.abs())
    net = pos * ret - turn * (cost_bps / 1e4)
    mean, sd = net.mean(), net.std()
    sharpe = float(mean / sd * np.sqrt(_TRADING_DAYS)) if sd > 0 else 0.0
    return {
        "net_ann_return": float(mean * _TRADING_DAYS),
        "sharpe": sharpe,
        "n_entries": int((pos.diff() > 0).sum()),
        "exposure": float(pos.mean()),
    }


def _best_constant(close: pd.Series, rsi: pd.Series, grid, cost_bps: float) -> dict:
    """The in-sample best fixed lower-RSI band over ``grid`` (exit at 50), by net Sharpe. Honest
    about what it is: a *re-optimised, in-sample* constant — the strawman-beating control, not a
    deployable rule."""
    best = None
    for lower in grid:
        pos = signals.rsi_band_positions(rsi, float(lower), 50.0)
        s = _strategy_stats(close, pos, cost_bps)
        if best is None or s["sharpe"] > best["sharpe"]:
            best = {**s, "lower": float(lower)}
    return best


def strategy_compare(close: pd.Series, length: int = 2, lower_sigma: float | None = None,
                     cost_bps: float = 1.0, grid=None) -> dict:
    """Horse race for one ``length``: fixed 70/30 vs adaptive-σ vs a re-optimised constant.

    * **fixed**: the textbook ``lower=30, exit=50`` band, applied regardless of length.
    * **adaptive**: the σ oversold landmark (``-√3`` by default, the framework's OB/OS edge) on the
      AdaptiveRSI scale — which, per :func:`crossing_identity`, *is* a particular constant band.
    * **reopt**: the in-sample best fixed lower band over ``grid`` — what "just pick a sensible
      length-matched number" achieves.

    Returns the three stat blocks plus the two gaps that decide the Signal stamp: ``adaptive_vs_fixed``
    (does length-awareness beat naive 70/30?) and ``adaptive_vs_reopt`` (does σ-calibration beat
    simply re-optimising a constant? — expected ≤ 0, since adaptive is just one constant the grid
    also contains). All Sharpes are gross of the multiple-testing the ``reopt`` search incurs — a
    caveat the real run prices with a Reality Check.
    """
    if lower_sigma is None:
        lower_sigma = -ZONE_SIGMA["trend"]            # the framework's oversold edge, -√3σ
    if grid is None:
        grid = np.arange(5.0, 45.0, 1.0)

    rsi = wilder_rsi(close, length)
    fixed = _strategy_stats(close, signals.rsi_band_positions(rsi, 30.0, 50.0), cost_bps)
    adaptive_pos = signals.sigma_band_positions(rsi, length, lower_sigma, 0.0)
    adaptive = _strategy_stats(close, adaptive_pos, cost_bps)
    lower_rsi, _ = signals.implied_rsi_band(length, lower_sigma, 0.0)
    reopt = _best_constant(close, rsi, grid, cost_bps)

    return {
        "length": int(length),
        "lower_sigma": float(lower_sigma),
        "adaptive_implied_lower_rsi": float(lower_rsi),
        "fixed": fixed,
        "adaptive": adaptive,
        "reopt": reopt,
        "adaptive_vs_fixed_sharpe": adaptive["sharpe"] - fixed["sharpe"],
        "adaptive_vs_reopt_sharpe": adaptive["sharpe"] - reopt["sharpe"],
    }


# --------------------------------------------------------------------------- #
# 4 · Does Rescaled long-RSI add anything over native short RSI?
# --------------------------------------------------------------------------- #

def _ic(x: pd.Series, y: pd.Series) -> float:
    """Spearman rank IC between a signal and a forward return, over their common valid rows."""
    valid = x.notna() & y.notna()
    if valid.sum() < 30:
        return float("nan")
    return float(x[valid].rank().corr(y[valid].rank()))


def rescale_increment(close: pd.Series, target_length: int = 14, long_length: int = 70,
                      horizon: int = 5) -> dict:
    """Does *rescaling* ``RSI(long)`` onto the short scale add any information, or just relabel it?

    Rescaled RSI is ``RSI(long) → σ → RSI(target)`` — a composition of two strictly monotone maps,
    hence a **monotone relabel of ``RSI(long)`` itself**. So for any rank- or threshold-based use
    its information is *identical* to the raw long RSI: this function shows that as a measured
    equality of rank ICs (``ic_rescaled_long`` == ``ic_native_long``), the cross-length twin of the
    within-length crossing identity.

    It also reports the genuinely interesting number that rescaling is *sold* as delivering — the
    incremental IC of the long signal over native ``RSI(target)`` (partial rank correlation,
    residualising on the short RSI). On a mean-reverting tape this is non-zero — but it is the
    contribution of the **longer window**, already present in *raw* ``RSI(long)``; rescaling adds
    none of it. The translation changes the number you read, not the signal you act on.
    """
    fwd = signals.forward_return(close, horizon)
    native = wilder_rsi(close, target_length)
    long_native = wilder_rsi(close, long_length)
    rescaled = rescaled_rsi(close, long_length, target_length)

    ic_native = _ic(native, fwd)
    ic_long = _ic(long_native, fwd)
    ic_rescaled = _ic(rescaled, fwd)

    # Incremental IC of the long signal over native short RSI (partial rank correlation): residualise
    # forward-return ranks and long-RSI ranks on the short-RSI ranks, then correlate the residuals.
    # Rescaled and raw-long share ranks, so this increment is a property of the *window*, not the σ.
    valid = fwd.notna() & native.notna() & long_native.notna()
    inc = float("nan")
    if valid.sum() >= 30:
        ry = fwd[valid].rank()
        rn = native[valid].rank()
        rl = long_native[valid].rank()
        b_y = np.polyfit(rn, ry, 1)
        b_l = np.polyfit(rn, rl, 1)
        res_y = ry - (b_y[0] * rn + b_y[1])
        res_l = rl - (b_l[0] * rn + b_l[1])
        if res_l.std() > 0 and res_y.std() > 0:
            inc = float(np.corrcoef(res_y, res_l)[0, 1])

    return {
        "target_length": int(target_length),
        "long_length": int(long_length),
        "horizon": int(horizon),
        "ic_native_target": ic_native,
        "ic_native_long": ic_long,
        "ic_rescaled_long": ic_rescaled,
        "rescale_ic_gap": float(ic_rescaled - ic_long),   # ~0: rescaling is rank-invariant
        "incremental_ic_window_over_native": inc,          # real, but it's the window, not the σ
        "n": int(valid.sum()),
    }
