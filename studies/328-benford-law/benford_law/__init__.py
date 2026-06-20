"""Study 328 — Benford-Law: do stock prices obey Benford's first-digit law, and can
the deviation flag anything tradable? (Spoiler: prices conform, the deviation is a
property of the price *range*, and it predicts nothing.)"""

from . import data, strategy

__all__ = ["data", "strategy"]
