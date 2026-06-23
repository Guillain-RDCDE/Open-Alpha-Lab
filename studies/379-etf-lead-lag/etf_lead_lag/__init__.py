"""Study 379 — ETF Lead-Lag (does the big liquid ETF lead its slow members?).

The folklore of microstructure: the big, liquid ETF (SPY) digests information first,
so a move in SPY today should *predict* tomorrow's move in its smaller, slower-to-update
constituents. If true, you could buy the laggards the instant the leader pops — a clean,
mechanical edge.

We measure the lead-lag relationship two honest ways: (1) the cross-correlation of the
leader's return with the *next-day* member returns (a lead-lag profile), and (2) a tradable
next-bar rule — long the basket of laggards the day after a big SPY up-move, net of one-way
costs — with a 1-day execution lag and a placebo/bootstrap null sized to the trade count.

True intraday tick lead-lag (where genuine sub-second effects live) is not on yfinance, which
serves daily OHLCV only; we say so on the Signal axis. The decisive question on *daily* bars
is whether any next-day predictability survives costs — and a deterministic synthetic control
with a planted lead-lag knob proves the engine is faithful (edge=0 must not manufacture
significance; a large planted lead must light up).

See :mod:`etf_lead_lag.data` (real basket loader + synthetic lead-lag control) and
:mod:`etf_lead_lag.strategy` (cross-correlation profile, next-bar rule, Welch t / placebo
null, win-rate vs base rate, one-way costs)."""

from . import data, strategy

__all__ = ["data", "strategy"]
