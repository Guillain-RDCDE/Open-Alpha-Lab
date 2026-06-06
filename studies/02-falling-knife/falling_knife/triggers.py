"""Entry triggers: four honest ways to define "the knife is falling".

Each trigger takes a returns frame (from :func:`falling_knife.data.daily_returns`)
and returns a boolean ``Series`` aligned to the index: ``True`` on days that fire
an entry signal. We keep them as *signals* (not trades) so the same definition can
feed the event study, the benchmark and the backtest unchanged.

The four definitions answer genuinely different questions:

    T1  close-to-close <= -3%   "the classic down-3%-day"      (reactive, end of day)
    T2  intraday low   <= -3%   "down 3% at some point today"  (most reactive, same-day fill)
    T3  drawdown       <= -3%   "we are 3% below a recent high" (the slide already started)
    T4  N-day return   <= -3%   "3% lost over a few days"       (slow bleed)

Why this matters: a single threshold ("-3%") is not a single strategy. T1 and T3
can disagree wildly — a market grinding down 0.5%/day for a week never triggers
T1 but is deep into T3. Testing all four is how we avoid mistaking one lucky
definition for an "edge".
"""

from __future__ import annotations

import pandas as pd

DEFAULT_THRESHOLD = -0.03  # the "-3%" of the project name


def close_to_close(returns: pd.DataFrame, threshold: float = DEFAULT_THRESHOLD) -> pd.Series:
    """T1 — a session that *closes* at or below ``threshold`` (close-to-close)."""
    sig = returns["r_cc"] <= threshold
    return sig.rename("T1_close_to_close").fillna(False)


def intraday(returns: pd.DataFrame, threshold: float = DEFAULT_THRESHOLD) -> pd.Series:
    """T2 — price trades ``threshold`` below the open at some point in the day.

    Uses the open-to-low return, so it fires even if the market recovers by the
    close. This is the most reactive definition (you could be filled the same day
    when the -3% level is touched).
    """
    sig = returns["r_intraday_low"] <= threshold
    return sig.rename("T2_intraday").fillna(False)


def drawdown(
    returns: pd.DataFrame,
    threshold: float = DEFAULT_THRESHOLD,
    window: int | None = 20,
) -> pd.Series:
    """T3 — close is ``threshold`` below a rolling high.

    ``window`` is the look-back in trading days (default 20 ≈ one month). Pass
    ``window=None`` for drawdown from the all-time (expanding) high.
    """
    close = returns["Close"]
    if window is None:
        peak = close.cummax()
        name = "T3_drawdown_ATH"
    else:
        peak = close.rolling(window, min_periods=1).max()
        name = f"T3_drawdown_{window}d"
    dd = close / peak - 1.0
    return (dd <= threshold).rename(name).fillna(False)


def cumulative(
    returns: pd.DataFrame,
    threshold: float = DEFAULT_THRESHOLD,
    n_days: int = 5,
) -> pd.Series:
    """T4 — cumulative return over the trailing ``n_days`` is <= ``threshold``."""
    close = returns["Close"]
    cum = close / close.shift(n_days) - 1.0
    return (cum <= threshold).rename(f"T4_cum_{n_days}d").fillna(False)


# Registry so callers can iterate the whole family uniformly.
TRIGGERS = {
    "T1_close_to_close": close_to_close,
    "T2_intraday": intraday,
    "T3_drawdown_20d": lambda r, threshold=DEFAULT_THRESHOLD: drawdown(r, threshold, 20),
    "T3_drawdown_ATH": lambda r, threshold=DEFAULT_THRESHOLD: drawdown(r, threshold, None),
    "T4_cum_5d": lambda r, threshold=DEFAULT_THRESHOLD: cumulative(r, threshold, 5),
}


def first_crossings(signal: pd.Series, cooldown: int = 0) -> pd.Series:
    """Debounce a raw signal into *fresh* events.

    A raw threshold signal stays ``True`` for the whole time the condition holds
    (e.g. T3 is ``True`` every day the market sits below its high). For an event
    study or a one-position backtest you usually want only the *first* day of each
    episode, optionally with a ``cooldown`` (trading days) before a new event can
    fire — typically set to your maximum holding period so you never re-enter a
    position you are still holding.

    Returns a boolean ``Series`` of the same shape, ``True`` only on event days.
    """
    sig = signal.fillna(False).to_numpy()
    out = sig.copy()
    out[1:] &= ~sig[:-1]  # keep only rising edges (False -> True)

    if cooldown > 0:
        idx = out.nonzero()[0]
        last = -10**9
        for i in idx:
            if i - last < cooldown:
                out[i] = False
            else:
                last = i
    return pd.Series(out, index=signal.index, name=signal.name)
