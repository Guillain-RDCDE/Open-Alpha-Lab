"""Study 502 — Betting-Against-Correlation (Asness-Frazzini-Gormsen-Pedersen 2020).

The BAB premium decomposes as beta = correlation x (vol_i / vol_mkt). AFGP argue the
*correlation* slice — not the volatility slice — is what carries the premium. This package
sorts a large-cap survivor cross-section on each name's trailing correlation-to-market,
builds a beta-neutralised long-low-correlation / short-high-correlation book (BAC), and
reports it honestly: one-sample / HAC t, a label-shuffle placebo null, costs x turnover
plus short borrow, a seed-robust synthetic positive control, and the beta-vs-correlation
decomposition that distinguishes this from betting-against-beta (Study 238).
"""

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
