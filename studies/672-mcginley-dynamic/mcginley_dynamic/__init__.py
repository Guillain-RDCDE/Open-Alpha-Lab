"""Study 672 — McGinley Dynamic.

Does John McGinley's self-adjusting "Dynamic" line hug price and time crossovers better
than a plain SMA/EMA — or is it just another moving average wearing a smarter costume?

See ``data.py`` for the tapes (real + synthetic positive control) and ``strategy.py`` for
the indicator, the timing rules and the inference engine.
"""

from . import data, strategy

__all__ = ["data", "strategy"]
