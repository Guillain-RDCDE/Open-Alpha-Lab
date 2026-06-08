"""Twin-Spread (Study 05) — the GGR (1999) pairs-trading rule, tested for decay.

Modules:
  * :mod:`pairs_trading.data`       — cached real universe + a synthetic one with true twins.
  * :mod:`pairs_trading.pairs`      — minimum-SSD pair formation (the parameter-free GGR rule).
  * :mod:`pairs_trading.backtest`   — the open-at-2σ / close-on-crossing trade, costs & exec lag.
  * :mod:`pairs_trading.robustness` — decay-by-year, the bid-ask-bounce wait rule, neutrality, capacity.
"""

from . import backtest, data, pairs, robustness

__all__ = ["data", "pairs", "backtest", "robustness"]
