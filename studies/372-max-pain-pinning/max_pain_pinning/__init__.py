"""Study 372 — Max-Pain Pinning (does the stock get pinned to the max-pain strike?).

The folklore: on option-expiry day the underlying is "pinned" to the **strike of maximum
pain** — the strike at which the largest dollar amount of open interest expires worthless
(i.e. option *holders* feel the most pain / option *writers* pay out the least). Dealers,
the story goes, hedge their books in a way that drags the price toward that strike into the
close.

What yfinance gives us is the **current** option chain (future expiries only), so we can
compute today's max-pain strike per underlying but we *cannot* see how those contracts
expire — there is no retained history of expiry-day closes matched to that expiry's
max-pain. So the real tape here is an explicit **snapshot**: 20 liquid US underlyings on a
single as-of date, each with its max-pain strike and its spot-vs-max-pain gap. It documents
*where the price sits relative to max-pain right now* — it cannot, by construction, test the
*landing-at-expiry* claim.

The decisive evidence is therefore a deterministic **synthetic pinning model** (the control):
many simulated (underlying, expiry) episodes, each with a real max-pain strike, and a tunable
``pin`` knob that drags the final days toward max-pain. With ``pin = 0`` there is no pinning
and the test must NOT manufacture significance; with a large ``pin`` it must light up. We test
whether the expiry close lands closer to max-pain than to a *random* strike (paired Welch t +
a strike-shuffle placebo), with a win-rate vs the geometric base rate.

See :mod:`max_pain_pinning.data` (real snapshot loader + deterministic synthetic episodes)
and :mod:`max_pain_pinning.strategy` (max-pain computation, closeness inference, placebo null,
a 1-day-lag / one-way-cost expiry-fade trade)."""

from . import data, strategy

__all__ = ["data", "strategy"]
