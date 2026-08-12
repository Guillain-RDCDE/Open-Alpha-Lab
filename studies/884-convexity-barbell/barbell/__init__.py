"""Study 884 — Convexity Barbell.

A duration-matched **barbell** (SHY + TLT weighted to the same empirical duration as the
IEF bullet) carries more **convexity** than the bullet, so at equal duration it should
out-earn the bullet when yields move a lot. This package builds the barbell, the bullet,
the duration-matched spread, the convexity-capture regression, the excess-vs-excess
Sharpe race, an era cut, a costed timer, and a seeded synthetic positive control.
"""

from . import data, strategy

__all__ = ["data", "strategy"]
