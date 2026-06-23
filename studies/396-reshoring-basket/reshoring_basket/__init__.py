"""Study 396 — Reshoring-Basket (the "bring it home" trade: US industrials + Mexico).

The pitch: deglobalisation, tariffs and the post-COVID supply-chain scare are dragging
manufacturing back to North America — so a basket of US industrials / factory-automation
names plus Mexican equities (the nearshoring beneficiary) should *structurally* out-earn a
plain global index. We build a transparent reshoring basket — equal-weight ``XLI`` (US
industrials), ``EWW`` (Mexico) and an automation/robotics sleeve — and test whether its
return *in excess of* the benchmark (SPY, and the global ``ACWI``) is a real, positive,
autocorrelation-robust edge, or just sector beta you were always paid for.

The decisive question is not "did the basket go up?" (almost everything did) but "did it
beat its benchmark by *more* than its beta explains, reliably enough to clear t ≥ 2 on the
real tape, net of costs?" — the desk's two-axis verdict.

See :mod:`reshoring_basket.data` (real ETF loader + deterministic synthetic control with a
planted-alpha knob) and :mod:`reshoring_basket.strategy` (excess return, plain + HAC t,
CAPM alpha, excess-of-cash Sharpe race, placebo null, costs)."""

from . import data, strategy

__all__ = ["data", "strategy"]
