"""The indicator and the AdaptiveRSI transform — RSI, the logit bridge, and the σ↔RSI map.

This is where the framework under test actually lives. Four layers, smallest to largest:

1. :func:`wilder_rsi` — the classic Wilder RSI(``length``). Nothing adaptive yet; the familiar
   bounded 0–100 oscillator everything else is derived from.

2. :func:`rsi_to_sigma` / :func:`sigma_to_rsi` — the **load-bearing transform**. AdaptiveRSI
   takes RSI out of the bounded scale with a logit and standardises it by a length factor::

       σ = logit(RSI/100) · √(n−1)/2          where logit(p) = ln(p / (1−p))

   The ``√(n−1)/2`` factor is the framework's own ``2/√(n−1)`` scaling, inverted. This is the
   exact arithmetic that makes the marketing line true: ``RSI(14) = 70`` ⟺ ``σ = +1.53``
   (``ln(70/30)·√13/2 = 1.527``). Both maps are strictly monotone in their argument at fixed
   ``n`` — the single fact the whole "is this real?" verdict turns on.

3. :func:`adaptive_zone_levels` / :func:`zone_levels_table` — the cheat-sheet. The framework fixes
   a handful of σ landmarks (``±0.66, ±1, ±√3, ±2.14``) and translates them back into RSI values
   *per length*. Those values are a pure function of ``n``; they compress toward 50 as ``n`` grows.
   This module reproduces the published table from the formula — no fitting, no data.

4. :func:`rescaled_rsi` — Rescaled RSI: read a long-length RSI on a short length's scale by going
   ``RSI(long) → σ → RSI(target)``. The framework's pitch for comparing horizons; here, the thing
   the decomposition checks for *incremental* information over native short RSI.

Everything is a pure, deterministic function of a price series. SciPy is not needed; NumPy only.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

# The framework's fixed σ landmarks (one side; the map is symmetric about RSI 50 / σ 0).
# Consolidation band edge, support/resistance edge, trend edge, overbought/oversold edge.
ZONE_SIGMA = {
    "consolidation": 0.66,
    "support_resistance": 1.0,
    "trend": math.sqrt(3.0),          # ≈ 1.732
    "overbought_oversold": 2.14,
}

_EPS = 1e-9


# --------------------------------------------------------------------------- #
# Classic Wilder RSI
# --------------------------------------------------------------------------- #

def wilder_rsi(close: pd.Series, length: int = 14) -> pd.Series:
    """Wilder's RSI(``length``) of a close series, on the 0–100 scale.

    Average gain / average loss are Wilder-smoothed (an EMA with ``alpha = 1/length``), seeded
    after ``length`` observations. A run with no losses returns 100; with no gains, 0. The first
    ``length`` values are ``NaN`` (insufficient history), exactly as a charting package shows them.
    """
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = (-delta).clip(lower=0.0)
    roll_up = up.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()
    roll_down = down.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()
    rs = roll_up / roll_down.replace(0.0, np.nan)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    rsi = rsi.where(roll_down != 0.0, 100.0)          # all-gains window -> RSI 100
    rsi = rsi.where(roll_up != 0.0, 0.0)              # all-losses window -> RSI 0
    rsi[roll_up.isna()] = np.nan                       # respect the warm-up NaNs
    return rsi.rename(f"rsi_{length}")


# --------------------------------------------------------------------------- #
# The AdaptiveRSI σ↔RSI transform
# --------------------------------------------------------------------------- #

def _scale(length: int) -> float:
    """The length factor ``√(n−1)/2`` — the inverse of the framework's ``2/√(n−1)`` scaling."""
    return math.sqrt(length - 1) / 2.0


def rsi_to_sigma(rsi, length: int):
    """Map RSI (0–100) into AdaptiveRSI σ units: ``σ = logit(RSI/100) · √(n−1)/2``.

    Strictly increasing in ``rsi`` at fixed ``length``. Accepts a scalar, array, or ``pd.Series``;
    preserves a Series' index/name. RSI is clipped just inside (0, 100) so the logit stays finite.
    """
    arr = rsi.to_numpy() if isinstance(rsi, pd.Series) else np.asarray(rsi, dtype="float64")
    p = np.clip(arr / 100.0, _EPS, 1.0 - _EPS)
    sigma = np.log(p / (1.0 - p)) * _scale(length)
    if isinstance(rsi, pd.Series):
        return pd.Series(sigma, index=rsi.index, name=f"sigma_{length}")
    return sigma if sigma.ndim else float(sigma)


def sigma_to_rsi(sigma, length: int):
    """Inverse of :func:`rsi_to_sigma`: ``RSI = 100 · expit(2σ / √(n−1))``.

    The translation that turns a fixed σ landmark into a length-specific RSI threshold. Strictly
    increasing in ``sigma``; scalar/array/Series in, same shape out.
    """
    arr = sigma.to_numpy() if isinstance(sigma, pd.Series) else np.asarray(sigma, dtype="float64")
    logit = arr / _scale(length)
    rsi = 100.0 / (1.0 + np.exp(-logit))
    if isinstance(sigma, pd.Series):
        return pd.Series(rsi, index=sigma.index, name=f"rsi_{length}")
    return rsi if rsi.ndim else float(rsi)


# --------------------------------------------------------------------------- #
# Adaptive zones — the cheat-sheet, as pure arithmetic in n
# --------------------------------------------------------------------------- #

def adaptive_zone_levels(length: int) -> dict[str, float]:
    """The AdaptiveRSI zone boundaries for one ``length``, in RSI units (upper and lower side).

    Each fixed σ landmark in :data:`ZONE_SIGMA` is translated back to an RSI value for this length
    via :func:`sigma_to_rsi`. Returns ``{name_upper: rsi, name_lower: rsi, ...}``. These are the
    numbers the framework's cheat-sheet prints — here derived, not calibrated.
    """
    out: dict[str, float] = {}
    for name, s in ZONE_SIGMA.items():
        out[f"{name}_upper"] = sigma_to_rsi(+s, length)
        out[f"{name}_lower"] = sigma_to_rsi(-s, length)
    return out


def zone_levels_table(lengths) -> pd.DataFrame:
    """The cheat-sheet as a frame: one row per ``length``, the *upper* RSI value of each σ zone.

    Columns are the σ landmarks (``consolidation … overbought_oversold``); each cell is the RSI
    level that landmark maps to for that length. Reading down a column shows the framework's
    headline behaviour — **the same σ boundary compresses toward 50 as the length grows**.
    """
    rows = {}
    for n in lengths:
        rows[n] = {name: sigma_to_rsi(+s, n) for name, s in ZONE_SIGMA.items()}
    df = pd.DataFrame(rows).T
    df.index.name = "length"
    return df[list(ZONE_SIGMA.keys())]


# --------------------------------------------------------------------------- #
# Rescaled RSI — read one length on another's scale
# --------------------------------------------------------------------------- #

def rescaled_rsi(close: pd.Series, length: int, target_length: int = 14) -> pd.Series:
    """Rescaled RSI: express ``RSI(length)`` on the ``RSI(target_length)`` scale.

    Goes ``RSI(length) → σ`` (standardise out of the source length) ``→ RSI(target_length)``
    (re-express on the familiar scale). When ``length == target_length`` this is the identity. The
    framework's tool for putting a long-horizon RSI back on a scale a trader already reads; the
    decomposition asks whether it carries information native ``RSI(target_length)`` does not.
    """
    src = wilder_rsi(close, length)
    sig = rsi_to_sigma(src, length)
    return sigma_to_rsi(sig, target_length).rename(f"rescaled_{length}_to_{target_length}")
