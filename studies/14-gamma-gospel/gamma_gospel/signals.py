"""The signals — the regime labels the GEX pitch trades on, as named conditions and outcomes.

The pitch's whole framework is one binary read, knowable before the open:

    neg_gamma   net dealer GEX < 0  -> dealers are SHORT gamma. They hedge *with* the move
                (buy as it rises, sell as it falls): the "amplifier". The pitch's TREND day.
    pos_gamma   net dealer GEX > 0  -> dealers are LONG gamma. They hedge *against* the move
                (sell rips, buy dips): the "shock absorber". The pitch's RANGE day.

and two realised outcomes a study of the claim must keep apart (both pure functions of the day's
OHLC, already on the panel):

    realized_vol            range-based (Parkinson) vol — tests "negative gamma => more vol".
    directional_efficiency  |close-open| / (high-low) — the trend-vs-chop character that *is* the
                            range-vs-trend claim: high on a trend day, low on a chop/range day.

Everything here is a pure function of the panel; no parameters, no fitting.
"""

from __future__ import annotations

import pandas as pd


def neg_gamma(panel: pd.DataFrame) -> pd.Series:
    """Dealers short gamma at the prior close (GEX < 0) — the pitch's 'amplifier' / trend regime."""
    return panel["neg_gamma"].astype("boolean")


def pos_gamma(panel: pd.DataFrame) -> pd.Series:
    """Dealers long gamma at the prior close (GEX > 0) — the pitch's 'shock absorber' / range regime."""
    return ~panel["neg_gamma"].astype("boolean")


def realized_vol(panel: pd.DataFrame) -> pd.Series:
    """The day's range-based (Parkinson) volatility — the outcome behind 'negative gamma => more vol'."""
    return panel["rv"].astype(float)


def directional_efficiency(panel: pd.DataFrame) -> pd.Series:
    """The day's |close-open| / (high-low) — high on a trend day, low on a range day."""
    return panel["de"].astype(float)


def regime_masks(panel: pd.DataFrame) -> dict[str, pd.Series]:
    """The bank of regime conditions, as named boolean masks: baseline, neg-gamma, pos-gamma."""
    neg = neg_gamma(panel)
    return {
        "baseline": pd.Series(True, index=panel.index, dtype="boolean"),
        "neg_gamma": neg,
        "pos_gamma": ~neg,
    }
