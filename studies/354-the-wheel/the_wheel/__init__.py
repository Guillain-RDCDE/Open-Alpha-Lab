"""Study 354 — The Wheel (selling puts then calls "forever" for income).

The retail "Wheel": sell a cash-secured at-the-money **put** on an index ETF each month;
if assigned (the ETF closed below the strike), switch to selling at-the-money **covered
calls** until called away; repeat for "income". The pitch frames it as a money machine that
spins out premium regardless of market direction.

This study shows the Wheel is not free income — it is a **packaged short-volatility /
covered-call exposure** with the classic short-vol payoff: it keeps the option premium and a
sliver of upside, caps the rest of the upside, and **takes the full downside**. We model the
monthly ATM option premium transparently with Black-Scholes (SPY price, ^VIX as the implied
vol — a clearly-labelled model, not a live option chain), then simulate the Wheel against
buy-and-hold SPY and a cash baseline.

See :mod:`the_wheel.data` (real SPY+VIX loader + a deterministic synthetic short-vol world)
and :mod:`the_wheel.strategy` (the Black-Scholes premium, the Wheel engine, the payoff
decomposition, and the inference: a paired *t* of monthly excess returns vs buy-and-hold and
the upside/downside capture that names the short-vol shape)."""

from . import data, strategy

__all__ = ["data", "strategy"]
