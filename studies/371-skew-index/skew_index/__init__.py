"""Study 371 — CBOE SKEW index (the "black-swan" gauge) vs forward SPY.

The pitch: SKEW reads the price of out-of-the-money S&P 500 puts relative to at-the-money
options, so a high SKEW supposedly means the option market is *paying up for crash insurance* —
a tail-risk warning the symmetric, at-the-money VIX can't give. We test whether SKEW carries
forward-return or tail information **beyond** what VIX already prices.

The decisive object is incremental information: a forward-return regression on *both* SKEW and
VIX with HAC (Newey-West) standard errors. If VIX already prices the forward distribution, the
SKEW coefficient's own t is indistinguishable from zero. We also bucket forward returns by SKEW
level (directional test), check whether high SKEW raises the forward tail-event probability, and
cost a "fade high SKEW" sleeve for the Tradability axis.

See :mod:`skew_index.data` (real loader for ^SKEW/^VIX/SPY + a deterministic synthetic control
with a planted-edge knob) and :mod:`skew_index.strategy` (bucketed forward returns, the
VIX-controlled regression, tail-warning test, placebo null, and the costed fade sleeve)."""

from . import data, strategy

__all__ = ["data", "strategy"]
