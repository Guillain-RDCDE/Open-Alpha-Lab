"""Study 505 -- Left-Tail-Momentum (Atilgan-Bali-Demirtas-Gunaydin 2020).

Two public surfaces:

- ``data``     -- a deterministic synthetic panel (offline, tunable planted effect)
                  plus a yfinance real panel cached to this study's own ``_cache/``.
- ``strategy`` -- the left-tail signal (trailing VaR / worst-day), the long-short
                  book, one-sample t, a placebo label-shuffle null, costs x turnover
                  (+ borrow on the short leg), seed-robustness, and a deterministic
                  synthetic positive control.
"""

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
