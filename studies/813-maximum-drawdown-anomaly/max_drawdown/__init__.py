"""Study 813 — the Maximum-Drawdown Anomaly.

The claim: sort a liquid US cross-section on each name's **trailing 12-month maximum
drawdown** (the largest peak-to-trough decline of its cumulative total return); ask
whether the recently distressed names — deepest past drawdown — subsequently
**under-earn** (a distress premium) or **rebound** (a reversal). The desk takes no prior
on the sign and stamps whatever the tape shows.

* ``data``     — the real cross-section (yfinance daily OHLC, cached under the study's
                 own ``_cache/`` through the ``quantlab.universe`` survivorship guard)
                 plus a deterministic seeded synthetic positive control (a planted
                 distress->underperformance relation, null at ``edge=0``).
* ``strategy`` — the trailing maximum-drawdown signal, the point-in-time
                 cross-sectional sort (long calm / short distressed), the inference
                 primitives (Welch / one-sample / Newey-West HAC / Wilson / placebo),
                 and the costed timer.
"""

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
