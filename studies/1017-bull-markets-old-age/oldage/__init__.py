"""Study 1017 — Do Bull Markets Die of Old Age?

"This bull market is long in the tooth." The phrase carries a precise statistical claim:
**positive duration dependence** — the older a bull market gets, the more likely it is to end
in the next month, so a prudent investor de-risks late in a long bull. This study takes the
claim at its word and asks the question that usually goes missing: is the ageing a property of
markets, or a property of the ruler we date them with?

A ±20% peak-to-trough rule cannot end a bull in its first few months (prices have to climb 20%
before anyone calls it a bull at all), so *any* series dated that way — including a pure random
walk — shows a hazard that starts at zero and rises. The honest test compares real bulls with
random-walk bulls dated by the very same rule.

- :mod:`oldage.data` — the real tapes (Fama-French monthly total return 1926→2018 and the daily
  S&P 500 price index 1990→2022, both SHA-pinned via :mod:`quantlab.bundled`) and a
  deterministic synthetic generator with a ``signal_strength`` knob that plants genuine ageing.
- :mod:`oldage.strategy` — the dating rule, the Weibull duration model, the random-walk and GARCH
  nulls, the predictive regression, the de-risking rule and the pre-registered verdict.
"""

from __future__ import annotations

__all__ = ["data", "strategy"]
