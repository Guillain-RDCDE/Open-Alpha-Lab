"""Study 537 — Factor-Momentum: do the factors themselves trend (Ehsani-Linnainmaa 2022)?

We build a panel of monthly long-short factor returns (momentum, low-vol, low-beta,
short-reversal, size-proxy) from a survivor large-cap basket, then run time-series momentum
on the *factors* and test whether the timed meta-premium beats the static factors and clears
the desk's |t| >= 2 + placebo bar.
"""

from . import data, strategy

__all__ = ["data", "strategy"]
