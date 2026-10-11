"""overnight — verify and (carefully) implement the overnight anomaly.

A small, reproducible toolkit that turns Knuteson's "overnight returns are
suspicious" claim into code you can run yourself: decompose night vs day,
stress-test the headline magnitudes (compounding, data artefacts, weighting),
and backtest the buy-at-close/sell-at-open strategy *with real costs* to see
why the gross edge rarely survives.

Not investment advice. See LICENSE.
"""

from . import (
    analytics,
    backtest,
    bayes,
    data,
    decompose,
    diagnostics,
    plots,
    simulate,
    stats,
    universe,
)

__all__ = [
    "decompose", "data", "plots", "diagnostics", "backtest",
    "stats", "analytics", "simulate", "bayes", "universe",
]
__version__ = "0.3.0"  # keep in step with pyproject.toml and CITATION.cff (tests/test_version.py)
