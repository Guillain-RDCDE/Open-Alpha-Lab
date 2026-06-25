"""Study 484 — Vertical-Horizontal-Filter (VHF) trend/range gate.

The VHF (Adam White, 1991) tries to tell a *trend* from a *range*:

    VHF_t = |max(close, N) - min(close, N)| / sum_{i=1..N-1} |close_i - close_{i-1}|

The numerator is the net vertical travel over the window; the denominator is the
total horizontal path length. High VHF ⇒ price went somewhere (trending); low VHF
⇒ price churned in place (ranging). The folklore says: only take momentum/breakout
signals when the VHF says "trending". We test whether **gating a momentum entry on a
high VHF adds any edge** over the same momentum entry ungated, and over a
drift-matched random baseline — with a placebo that scrambles the VHF gate while
preserving its marginal.
"""

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
