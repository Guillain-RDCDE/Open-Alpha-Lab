"""Study 142 — Split-Drift: do stocks drift up after a stock split?

Ikenberry, Rankine & Stice (1996) document a post-announcement drift after splits.
We test the *post-effective-date* version — which is strictly weaker than the
announcement-date effect because the market has already had time to react.
"""

from . import data, strategy

__all__ = ["data", "strategy"]
