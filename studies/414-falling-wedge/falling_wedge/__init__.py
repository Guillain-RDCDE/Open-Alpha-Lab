"""Study 414 — Falling Wedge (the classic bullish-reversal figure).

Package layout mirrors the desk house style:

* :mod:`falling_wedge.data` — the real (yfinance, cache-first) and synthetic (planted-edge)
  daily-OHLC tapes plus a content fingerprint.
* :mod:`falling_wedge.strategy` — an objective mechanical detector for the falling wedge
  (downward-converging trendlines + a confirmed upside break), a one-day-lag forward-return
  rule, and the honest arbiters (one-sample/HAC t, a same-tape label-shuffle placebo, costs,
  a down-break symmetry myth-check, and a synthetic positive control orchestrated by
  ``run_experiment``).
"""

from __future__ import annotations
