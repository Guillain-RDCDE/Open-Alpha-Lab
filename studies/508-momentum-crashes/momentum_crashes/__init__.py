"""Study 508 -- Momentum-Crashes.

Daniel & Moskowitz (2016): the cross-sectional momentum (winners-minus-losers) factor
carries rare but severe *crashes* -- they cluster in panicky, high-volatility rebounds out
of bear markets, when the short (past-loser) leg snaps back violently. We build the canonical
12-1 WML book, dissect its drawdowns and crash months, condition its conditional performance
on bear/panic regimes, and test whether *vol-scaling* (constant-volatility "dynamic momentum")
repairs the left tail.

Public surface:
    data.fetch_panel / data.fetch_market / data.synthetic_panel / data.fingerprint
    strategy.long_short / strategy.summary / strategy.placebo_pvalue
    strategy.vol_scaled / strategy.crash_table / strategy.regime_split / strategy.synthetic_control
"""

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
