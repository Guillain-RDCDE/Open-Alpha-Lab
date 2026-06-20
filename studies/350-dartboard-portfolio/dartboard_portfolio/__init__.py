"""Study 350 — Dartboard-Portfolio: random stock *selection* (the Malkiel dartboard)
raced against the cap-weighted index and a concentrated expert.

Distinct from Study 171 (Naive-1-Over-N), which tests 1/N *allocation* across a fixed
universe versus Markowitz optimisers. Here the question is the other half of Malkiel's
quip: does *which names* you pick — chosen blindfolded — matter, once they are held
equal-weight?"""

from . import data, strategy

__all__ = ["data", "strategy"]
