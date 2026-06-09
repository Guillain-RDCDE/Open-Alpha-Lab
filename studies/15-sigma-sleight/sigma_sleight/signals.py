"""The strategy — RSI mean reversion, written two equivalent ways so the relabel is visible.

The AdaptiveRSI pitch is interpretive ("describe the RSI environment"), but the only falsifiable
claim it implies is that **length-aware σ-zones read oversold/overbought better than a fixed
70/30**. The cleanest way to grade "better" is the textbook RSI mean-reversion rule the zones are
meant to improve:

    enter long the bar after RSI dips below a *lower* band, hold until RSI closes back above the
    50 equilibrium, then flat.

This module builds the position series for that rule two ways that *must* coincide:

    * :func:`rsi_band_positions` — threshold the RSI directly against a constant ``lower`` level
      (the fixed-70/30 way, and also where a re-optimised constant plugs in).
    * :func:`sigma_band_positions` — threshold the **σ-transformed** RSI against a fixed σ
      landmark (the "adaptive" way). Exit at ``σ = 0``, which is *exactly* RSI 50.

Because ``σ = logit(RSI)·√(n−1)/2`` is strictly increasing at fixed length, entering when
``σ < lower_sigma`` is the same event as entering when ``RSI < sigma_to_rsi(lower_sigma)`` — so
the two builders return identical trades. Demonstrating that equality is the study's first result;
the functions live here, the proof is in :mod:`sigma_sleight.decompose`.

Everything is a pure function of a price/RSI series; no fitting, no look-ahead (positions are
shifted one bar, so a signal on the close is traded the next session).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .rsi import rsi_to_sigma, sigma_to_rsi, wilder_rsi


def _hold_positions(level: np.ndarray, enter_below: float, exit_above: float) -> np.ndarray:
    """Stateful long/flat hold: go long when ``level`` first dips below ``enter_below``, stay long
    until it closes back above ``exit_above``. ``NaN`` levels (warm-up) force flat."""
    pos = np.zeros(len(level))
    holding = False
    for i, v in enumerate(level):
        if np.isnan(v):
            holding = False
        elif not holding and v < enter_below:
            holding = True
        elif holding and v > exit_above:
            holding = False
        pos[i] = 1.0 if holding else 0.0
    return pos


def rsi_band_positions(rsi: pd.Series, lower: float, exit_level: float = 50.0) -> pd.Series:
    """Long/flat mean-reversion positions from a **constant RSI** lower band.

    Enter long after ``rsi`` dips below ``lower``; exit once it recovers above ``exit_level``
    (default the 50 equilibrium). The position is shifted one bar so a close-based signal is traded
    the next session. This is the fixed-70/30 strategy (``lower=30``) and the container a
    re-optimised per-length constant slots into.
    """
    raw = _hold_positions(rsi.to_numpy(), enter_below=lower, exit_above=exit_level)
    return pd.Series(raw, index=rsi.index, name="pos").shift(1).fillna(0.0)


def sigma_band_positions(rsi: pd.Series, length: int, lower_sigma: float,
                         exit_sigma: float = 0.0) -> pd.Series:
    """Long/flat mean-reversion positions from a **σ landmark** on the AdaptiveRSI scale.

    Identical rule to :func:`rsi_band_positions`, but the band is a fixed σ level: enter when the
    σ-transformed RSI dips below ``lower_sigma``, exit above ``exit_sigma`` (default ``0`` = RSI
    50). Because the σ↔RSI map is strictly monotone at fixed ``length``, this returns the *same*
    trades as :func:`rsi_band_positions` called with ``lower = sigma_to_rsi(lower_sigma, length)``
    and ``exit_level = sigma_to_rsi(exit_sigma, length)`` — which is the whole point.
    """
    sig = rsi_to_sigma(rsi, length)
    raw = _hold_positions(sig.to_numpy(), enter_below=lower_sigma, exit_above=exit_sigma)
    return pd.Series(raw, index=rsi.index, name="pos").shift(1).fillna(0.0)


def implied_rsi_band(length: int, lower_sigma: float, exit_sigma: float = 0.0) -> tuple[float, float]:
    """The constant RSI ``(lower, exit)`` band a σ landmark pair collapses to for this ``length``.

    The bridge that makes "adaptive" concrete: an adaptive σ-zone is *nothing but* this pair of
    fixed RSI numbers. Used both to prove the crossing identity and to print what each adaptive
    band "really is" on the 0–100 scale.
    """
    return sigma_to_rsi(lower_sigma, length), sigma_to_rsi(exit_sigma, length)


def forward_return(close: pd.Series, horizon: int = 5) -> pd.Series:
    """The ``horizon``-bar ahead simple return of ``close`` — the target the IC test predicts."""
    return close.shift(-horizon) / close - 1.0
