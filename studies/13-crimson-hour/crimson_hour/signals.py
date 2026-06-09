"""The signals — the two morning tells edgeful combines, as named boolean conditions.

The pitch stacks two early-bearish reads observable by 10:30 ET and asks how often the day
then closes red:

    OC-red            the 09:30->10:30 "opening candle" closes below its open.
    IB-high-rejected  the first hour's HIGH prints before its LOW (so the low breaks second).
    confluence        OC-red AND IB-high-rejected — the "88% on ES" setup.

This module turns a session panel (:mod:`crimson_hour.data`) into those masks, plus the two
*outcomes* a study of the claim must keep distinct:

    session_red   the full session closes below the 09:30 open — what the pitch predicts, but
                  one that is **mechanically** already half-decided the moment OC-red is true
                  (the day is "behind" at 10:30 and need only fail to fully recover).
    rest_red      the *rest of the day* (10:30->16:00) closes red — the genuinely *forecastable*
                  question at 10:30, with no head-start baked in. The gap between the two is the
                  whole game (see :mod:`crimson_hour.decompose`).

Everything here is a pure function of the panel; no parameters, no fitting.
"""

from __future__ import annotations

import pandas as pd


def oc_red(feat: pd.DataFrame) -> pd.Series:
    """The opening candle closed red (first-hour return < 0)."""
    return feat["oc_red"].astype("boolean")


def ib_high_rejected(feat: pd.DataFrame) -> pd.Series:
    """The first hour's high printed before its low (``NA`` when bars are too coarse to tell)."""
    return feat["ib_high_rejected"].astype("boolean")


def confluence(feat: pd.DataFrame) -> pd.Series:
    """The edgeful setup: OC-red **and** IB-high-rejected (``NA`` where the flag is unknown)."""
    return oc_red(feat) & ib_high_rejected(feat)


def session_red(feat: pd.DataFrame) -> pd.Series:
    """Outcome the pitch sells: the full RTH session closed red."""
    return feat["session_red"].astype("boolean")


def rest_red(feat: pd.DataFrame) -> pd.Series:
    """Outcome that is actually forecast at 10:30: the rest of the day (10:30->16:00) closed red."""
    return feat["rest_red"].astype("boolean")


def condition_masks(feat: pd.DataFrame) -> dict[str, pd.Series]:
    """The bank of morning conditions, as named boolean masks, for the conditional tables.

    Returns every cell the decomposition compares: the unconditional baseline, each single
    signal and its complement, and the two-way confluence and its OC-red-but-*not*-rejected
    sibling (the control that isolates what IB-rejection adds on top of OC-red).
    """
    oc = oc_red(feat)
    ib = ib_high_rejected(feat)
    return {
        "baseline": pd.Series(True, index=feat.index, dtype="boolean"),
        "oc_red": oc,
        "oc_green": ~oc,
        "ib_rejected": ib,
        "ib_not_rejected": ~ib,
        "confluence (oc_red & ib_rej)": oc & ib,
        "oc_red & ib_NOT_rej": oc & ~ib,
    }
