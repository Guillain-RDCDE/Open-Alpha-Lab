"""Study 230 — Ohlson O-score: the 9-variable logit distress model as an equity signal.

O-score = -1.32 - 0.407*SIZE + 6.03*TLTA - 1.43*WCTA + 0.076*CLCA
          - 1.72*OENEG - 2.37*NITA - 1.83*FUTL + 0.285*INTWO - 0.521*CHIN

Built from the desk's shared EDGAR caches; maps high O-score (distress) vs low O-score
(safe) to next-year equity returns. See companion study 123-altman-z for Altman's Z.
"""

from . import data, strategy

__all__ = ["data", "strategy"]
