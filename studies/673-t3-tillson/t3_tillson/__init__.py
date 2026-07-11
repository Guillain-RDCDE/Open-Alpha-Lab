"""Study 673 — T3 (Tillson).

Does Tim Tillson's "T3" — a six-times-smoothed, generalized-DEMA moving average with a
tunable "volume factor" v — give a genuinely lower-lag, cleaner-crossover trend filter
than a plain SMA/EMA, or is it just another shape wearing a smarter costume?

See ``data.py`` for the tapes (real basket + synthetic positive control) and
``strategy.py`` for the indicator, the timing rules and the inference engine.
"""

from . import data, strategy

__all__ = ["data", "strategy"]
