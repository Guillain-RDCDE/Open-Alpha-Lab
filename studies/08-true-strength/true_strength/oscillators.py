"""The three oscillators, written so they can be compared *like with like*.

The whole study hinges on one comparison, so the three indicators are computed here with
matched conventions and a single normaliser, then everything downstream reads the same
table. No indicator gets a home-field advantage.

**True Strength Index (Blau).** Take the one-bar price change ``m = close.diff()``. Smooth it
with two chained EMAs (the classic spans are ``r = 25`` then ``s = 13``); do the same to
``|m|``; the TSI is ``100 × EMA_s(EMA_r(m)) / EMA_s(EMA_r(|m|))``. It lives in [-100, +100],
crosses zero with momentum, and many platforms add a signal line (an EMA of the TSI). The
"double smoothing" is the indicator's whole selling point — it is what is supposed to make
it a *truer*, cleaner strength reading than a raw momentum oscillator.

**MACD (Appel).** ``EMA_12(close) − EMA_26(close)``, with a 9-EMA signal line. We read its
**histogram** (``macd − signal``) as the momentum object, because that is the zero-crossing,
mean-reverting quantity directly comparable to the TSI.

**RSI (Wilder).** Wilder's 14-period smoothing of up- vs down-moves, in [0, 100]. We centre
it at 50 so its zero-cross lines up with the others'.

To compare them we **z-score each oscillator within its own name** (full-sample, mean/σ): a
unit-free score that strips the arbitrary [-100,100] vs [0,100] vs bp scales and leaves only
*shape*. If the double-smoothed TSI is genuinely a different read on the market than MACD or
RSI, the z-scored series will diverge. If it is the same trade repainted, they won't.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Parameters
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class TSIParams:
    """TSI smoothing spans — Blau's classic 25/13, with a 13-EMA signal line."""
    r: int = 25          # the slow (first) EMA span
    s: int = 13          # the fast (second) EMA span
    signal: int = 13     # EMA of the TSI itself (the signal line)


@dataclass(frozen=True)
class MACDParams:
    fast: int = 12
    slow: int = 26
    signal: int = 9


def _ema(x: pd.Series, span: int) -> pd.Series:
    return x.astype(float).ewm(span=span, adjust=False).mean()


# --------------------------------------------------------------------------- #
# The three oscillators
# --------------------------------------------------------------------------- #

def tsi(close: pd.Series, p: TSIParams = TSIParams()) -> pd.DataFrame:
    """True Strength Index and its signal line. Columns: ``tsi``, ``signal``.

    ``tsi = 100 · EMA_s(EMA_r(Δclose)) / EMA_s(EMA_r(|Δclose|))``. The denominator is
    floored away from zero so a dead-flat patch returns 0, not a divide-by-zero.
    """
    m = close.astype(float).diff()
    ds = _ema(_ema(m, p.r), p.s)
    da = _ema(_ema(m.abs(), p.r), p.s)
    val = 100.0 * ds / da.where(da.abs() > 1e-12)
    val = val.fillna(0.0)
    return pd.DataFrame({"tsi": val, "signal": _ema(val, p.signal)})


def macd(close: pd.Series, p: MACDParams = MACDParams()) -> pd.DataFrame:
    """MACD line, signal line and histogram. Columns: ``macd``, ``signal``, ``hist``."""
    line = _ema(close, p.fast) - _ema(close, p.slow)
    sig = _ema(line, p.signal)
    return pd.DataFrame({"macd": line, "signal": sig, "hist": line - sig})


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI in [0, 100]. Wilder smoothing = EMA with ``alpha = 1/period``.

    With no down-moves RSI pins at 100, with no up-moves at 0, and a dead-flat patch (no
    moves either way) returns the neutral 50 — the textbook limits, handled explicitly so a
    zero denominator never silently collapses to the mid-line.
    """
    delta = close.astype(float).diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    roll_up = up.ewm(alpha=1.0 / period, adjust=False).mean()
    roll_down = down.ewm(alpha=1.0 / period, adjust=False).mean()
    rs = roll_up / roll_down.replace(0.0, np.nan)
    out = 100.0 - 100.0 / (1.0 + rs)
    out = out.where(roll_down > 0, 100.0)                       # no losses -> 100
    out = out.where(~((roll_up == 0) & (roll_down == 0)), 50.0)  # truly flat -> 50
    return out.fillna(50.0).rename("rsi")


# --------------------------------------------------------------------------- #
# The comparable momentum object for each, on one common z-scored scale
# --------------------------------------------------------------------------- #

def oscillator_panel(
    close: pd.Series,
    tsi_p: TSIParams = TSIParams(),
    macd_p: MACDParams = MACDParams(),
    rsi_period: int = 14,
) -> pd.DataFrame:
    """One frame with each oscillator's zero-centred momentum *level*, raw and z-scored.

    To compare like with like we read each indicator's **momentum-level** object — the thing
    that crosses zero when momentum turns — not a faster derivative of it:
      ``tsi`` (already ±100), ``macd_norm`` (MACD **line** / close, i.e. percent momentum,
      *not* the histogram, which is a faster line-minus-signal acceleration), ``rsi_c``
      (RSI − 50). Z-scored columns (full-sample, per name): ``z_tsi``, ``z_macd``, ``z_rsi``.
    """
    t = tsi(close, tsi_p)["tsi"]
    mnorm = macd(close, macd_p)["macd"] / close.astype(float)
    rc = rsi(close, rsi_period) - 50.0
    out = pd.DataFrame({"tsi": t, "macd_norm": mnorm, "rsi_c": rc})
    for src, dst in [("tsi", "z_tsi"), ("macd_norm", "z_macd"), ("rsi_c", "z_rsi")]:
        s = out[src]
        sd = s.std(ddof=0)
        out[dst] = (s - s.mean()) / sd if sd > 0 else 0.0
    return out


# --------------------------------------------------------------------------- #
# Positions — the tradable reading of each oscillator
# --------------------------------------------------------------------------- #

def position_zero_cross(osc: pd.Series, long_short: bool = False) -> pd.Series:
    """Long when the oscillator is above zero, else flat (or short, if ``long_short``).

    The signal is **lagged one bar** before it ever reaches the backtest (you trade on
    tomorrow's open off today's close), so this returns the *raw* contemporaneous stance;
    the backtest applies the shift. ``long_short`` flips flat -> short for the equity-curve
    comparison where a symmetric position is cleaner.
    """
    up = osc > 0
    if long_short:
        return up.astype(float) * 2.0 - 1.0       # +1 / -1
    return up.astype(float)                        # +1 / 0


def position_signal_cross(osc: pd.Series, signal: pd.Series, long_short: bool = False) -> pd.Series:
    """Long when the oscillator is above its signal line (MACD-style crossover)."""
    up = osc > signal
    if long_short:
        return up.astype(float) * 2.0 - 1.0
    return up.astype(float)


# --------------------------------------------------------------------------- #
# The TSI parameter grid — the search space the Reality Check has to price
# --------------------------------------------------------------------------- #

def tsi_grid(
    r_set=(13, 20, 25, 40),
    s_set=(7, 13, 21),
    signal_set=(7, 13),
) -> list[TSIParams]:
    """Every (r, s, signal) the way a tuner would sweep them — the implicit search.

    Default = 4 × 3 × 2 = 24 variants around Blau's 25/13/13. The point of the grid is
    not to *find* the best TSI; it's to count how many were tried, so White's Reality
    Check can ask whether the best one's Sharpe survives the search.
    """
    return [TSIParams(r=r, s=s, signal=sig) for r in r_set for s in s_set for sig in signal_set]
