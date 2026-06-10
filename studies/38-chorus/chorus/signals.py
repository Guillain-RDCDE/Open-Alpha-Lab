"""The component signals — a handful of simple, self-contained, CAUSAL cross-sectional alphas.

Each function takes a ``dates × ticker`` daily-returns panel and returns a same-shaped frame of daily
**weights** with three properties enforced identically across all of them, so they can be combined
apples-to-apples:

  * **dollar-neutral** — long and short legs net to zero each day (``Σ_i w_{i,t} = 0``);
  * **gross-1** — total absolute exposure is one each day (``Σ_i |w_{i,t}| = 1``);
  * **lagged one day** — the weight applied to day ``t``'s return is built only from information up to
    ``t-1``, so the book is tradable (no look-ahead).

The three are deliberately the desk's *own* prior anomalies, each individually thin:

  * :func:`momentum`  — trailing 12-1 month winners-minus-losers (Study 24 Stampede);
  * :func:`reversal`  — short-horizon contrarian fade of the relative move (Study 33 Slingshot);
  * :func:`low_vol`   — long the low-realised-vol names, short the high (the low-volatility anomaly).

:func:`all_signals` returns the three as a dict, the input the combiner expects.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _dollar_neutral_gross1(raw: pd.DataFrame) -> pd.DataFrame:
    """Demean each row to dollar-neutral, normalise to gross 1, lag one day. Shared by every signal."""
    x = raw.sub(raw.mean(axis=1), axis=0)                  # cross-sectionally demeaned ⇒ Σ w = 0
    gross = x.abs().sum(axis=1).replace(0.0, np.nan)
    w = x.div(gross, axis=0).fillna(0.0)                   # Σ |w| = 1
    return w.shift(1).fillna(0.0)                          # lagged ⇒ causal


def momentum(returns: pd.DataFrame, lookback: int = 252, skip: int = 21) -> pd.DataFrame:
    """Cross-sectional momentum: rank by trailing ``lookback``-day return, *skipping* the most recent
    ``skip`` days (the classic 12-1 month construction that omits the short-term reversal month). Long the
    relative winners, short the relative laggards."""
    prices = (1.0 + returns.fillna(0.0)).cumprod()
    trail = prices.shift(skip) / prices.shift(lookback) - 1.0   # return from t-lookback to t-skip
    return _dollar_neutral_gross1(trail)


def reversal(returns: pd.DataFrame, lookback: int = 5) -> pd.DataFrame:
    """Short-horizon reversal: the trailing ``lookback``-day return, *negated*. Long the relative
    laggards, short the relative leaders — fade the recent move (Study 33 Slingshot)."""
    prices = (1.0 + returns.fillna(0.0)).cumprod()
    trail = prices / prices.shift(lookback) - 1.0
    return _dollar_neutral_gross1(-trail)


def low_vol(returns: pd.DataFrame, lookback: int = 63) -> pd.DataFrame:
    """Low-volatility tilt: rank by trailing ``lookback``-day realised vol and go *long the low-vol*
    names, short the high-vol. Built from the NEGATED, demeaned vol so low vol ⇒ positive weight."""
    vol = returns.rolling(lookback, min_periods=lookback // 2).std()
    return _dollar_neutral_gross1(-vol)


def all_signals(returns: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """The three component weight-streams, keyed by name — the input :func:`chorus.strategy.combine`
    expects. Add or drop entries here to widen or narrow the chorus."""
    return {
        "momentum": momentum(returns),
        "reversal": reversal(returns),
        "low_vol": low_vol(returns),
    }
