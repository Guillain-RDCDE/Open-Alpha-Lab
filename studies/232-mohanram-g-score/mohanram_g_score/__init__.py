"""Study 232 -- Mohanram G-score: can 8 fundamental signals find the growth stocks that deliver?

Mohanram (2005) "Separating Winners from Losers among Low Book-to-Market Stocks using
Financial Statement Analysis": the G-score packages eight accounting signals into a composite
score to identify which growth/glamour stocks have genuine fundamental support.
"""

from . import data, strategy

__all__ = ["data", "strategy"]
