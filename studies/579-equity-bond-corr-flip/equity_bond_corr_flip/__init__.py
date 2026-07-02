"""Study 579 — Equity-Bond-Corr-Flip.

The sign of the rolling stock-bond return correlation as a *regime* timing signal for the
60/40 (equity/bond) portfolio. When the correlation is negative, bonds hedge equity drawdowns
and the 60/40 diversification works; when it flips positive (as it did in 2022), the hedge
stops working. This study asks whether the *sign of the trailing correlation* predicts anything
tradable about the forward 60/40 (or equity) experience.
"""

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
