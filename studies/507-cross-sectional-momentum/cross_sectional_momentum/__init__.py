"""Study 507 -- Cross-Sectional-Momentum (Jegadeesh-Titman 12-1 relative strength).

The canonical equity momentum factor, replicated honestly on a fixed ~40-name large-cap
survivor basket: rank by trailing 12-month return skipping the most recent month, go long
the top fraction (decile/quintile), short the bottom, rebalance monthly, hold one month.

Public surface:
    data.synthetic_panel / data.fetch_panel / data.fingerprint
    strategy.momentum_signal / strategy.long_short / strategy.summary /
    strategy.placebo_pvalue / strategy.synthetic_control
"""

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
