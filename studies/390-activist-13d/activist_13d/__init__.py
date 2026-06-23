"""Study 390 — Activist-13D (does a stock pop and keep drifting after a 13D?).

When an activist investor — Icahn, Elliott, Third Point, Pershing Square, Starboard,
ValueAct — crosses 5% of a target's shares "with the purpose of influencing control,"
US securities law (Section 13(d) of the 1934 Act) forces a public **Schedule 13D**
filing within a short window. The folklore: the stock *pops* on the announcement and
then keeps *drifting up* for months as the activist's plan unfolds — a free ride for
anyone who buys the day the 13D hits EDGAR.

True EDGAR coverage of every 13D is uneven on a free feed, so we work from a
**transparent, hardcoded table of ~25 famous activist 13D campaigns** (target, activist,
announcement date), pull each target's price + SPY from yfinance, and measure the
**announcement pop** (day-0/+1) and the **post-announcement drift** (excess of SPY over
1/3/6 months), with a one-day execution lag for the drift leg. The decisive question is
not "did some campaigns work" (Icahn/Apple did) but whether, across a basket selected
*because it is famous*, the average drift after you can actually buy beats the market by
more than luck.

See :mod:`activist_13d.data` (hardcoded 13D table + real-tape loader + a deterministic
synthetic event control with a planted-edge knob) and :mod:`activist_13d.strategy`
(announcement pop, post-event drift vs SPY, Welch t, placebo null, win-rate vs base rate,
1-day lag, one-way costs)."""

from . import data, strategy

__all__ = ["data", "strategy"]
