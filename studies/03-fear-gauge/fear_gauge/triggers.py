"""Entry triggers: the family of "the fear gauge says buy" definitions.

Each trigger takes a VIX frame (from :func:`fear_gauge.data.vix_frame`, which must
expose at least ``Close`` and the 1-day change ``vix_chg``) and returns a boolean
``Series`` aligned to the index: ``True`` on days that fire an entry signal. We keep
them as *signals* (not trades) so the same definition can feed the event study, the
benchmark and the backtest unchanged — exactly as in Study 02.

The screenshot that started this study conflates two genuinely different triggers,
so we keep them separate and add the regime split the famous chart ignores:

    V1  VIX close >= K            "the Prof's level rule"        (a cran of panic)
    V2  VIX 1-day change >= +30%  "Altucher's spike chart"       (a jump, any base)
    V3  V2 split by base level    "+30% from 13" vs "+30% from 60" (regime matters)
    V4  level >= 30 then add @ 50  "the martingale"              (a sizing rule, see backtest)

Why this matters: a +46% spike landing VIX at 15.59 (calm) and a +43% spike landing
it at 82.69 (1987-grade panic) are the *same* V2 event but opposite worlds. Testing
the family is how we avoid mistaking one regime — or one lucky decade — for an edge.
"""

from __future__ import annotations

import pandas as pd

LEVEL_K = 30.0          # the Prof's first rung
LEVEL_DOUBLE = 50.0     # the "double down" rung
SPIKE_THRESHOLD = 0.30  # Altucher's "+30% single-day spike"


def level(vix: pd.DataFrame, k: float = LEVEL_K) -> pd.Series:
    """V1 — VIX *closes* at or above ``k`` (the level rule, K in {30, 40, 50})."""
    sig = vix["Close"] >= k
    return sig.rename(f"V1_level_{int(k)}").fillna(False)


def spike(vix: pd.DataFrame, threshold: float = SPIKE_THRESHOLD) -> pd.Series:
    """V2 — VIX 1-day change at or above ``threshold`` (Altucher's spike)."""
    sig = vix["vix_chg"] >= threshold
    return sig.rename("V2_spike").fillna(False)


def spike_from_base(
    vix: pd.DataFrame,
    threshold: float = SPIKE_THRESHOLD,
    base_max: float | None = None,
    base_min: float | None = None,
) -> pd.Series:
    """V3 — a spike, conditioned on the VIX level *before* the jump.

    ``base_*`` bound the prior-day close: ``base_max`` isolates spikes from a *low*
    base (calm blowing up), ``base_min`` isolates spikes from an already-*high* base
    (panic getting worse). Pass one or the other to split V2 into its two regimes.
    """
    fired = vix["vix_chg"] >= threshold
    prior = vix["Close"].shift(1)
    name = "V3_spike"
    if base_max is not None:
        fired &= prior <= base_max
        name += f"_base_le{int(base_max)}"
    if base_min is not None:
        fired &= prior >= base_min
        name += f"_base_ge{int(base_min)}"
    return fired.rename(name).fillna(False)


# Registry so callers can iterate the whole family uniformly.
# V4 (the martingale) is a *sizing* rule, not a point signal — it lives in the
# backtest, built from V1 at the 30 and 50 rungs; it is intentionally absent here.
TRIGGERS = {
    "V1_level_30": lambda v: level(v, 30.0),
    "V1_level_40": lambda v: level(v, 40.0),
    "V1_level_50": lambda v: level(v, 50.0),
    "V2_spike": spike,
    "V3_spike_low_base": lambda v: spike_from_base(v, base_max=20.0),
    "V3_spike_high_base": lambda v: spike_from_base(v, base_min=30.0),
}


def first_crossings(signal: pd.Series, cooldown: int = 0) -> pd.Series:
    """Debounce a raw signal into *fresh* events (rising edges only).

    A level signal stays ``True`` for the whole time the VIX sits above the rung;
    for an event study or a one-position backtest you usually want only the *first*
    day of each episode, optionally with a ``cooldown`` (trading days) before a new
    event can fire — typically your maximum holding period, so you never re-enter a
    position you are still holding.

    Shared verbatim with Study 02's ``triggers.first_crossings`` so episodes are
    counted the same way across the desk.
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
