"""Study 373 — Dispersion-Trade (is index vol reliably "cheap" vs its constituents?).

The dispersion trade is the volatility desk's favourite "free lunch": short index
volatility, long the volatility of the index's single names. The pitch rests on a hard
identity — index variance is a *correlation-weighted* average of single-name variances,
so index vol is always **below** the (weighted) average single-name vol whenever pairwise
correlation is < 1. Sellers of index vol vs. buyers of single-name vol harvest that gap,
which they frame as "index vol is reliably cheap."

True dispersion is an **implied-vol / implied-correlation** trade (sell index variance
swaps, buy single-name variance swaps). Implied vols are not on yfinance, so we build a
transparent **realized-vol proxy**: rolling realized vol of SPY vs. the basket of its
large constituents, and the realized dispersion gap = mean single-name vol − index vol.
We label it a proxy throughout. The decisive questions are (1) is the gap *reliably
positive* — yes, but that is a near-mechanical consequence of subadditivity, not an edge;
and (2) does the *carry* of holding the long-single / short-index leg survive a one-day
lag and real bid/ask + vega costs — where the realized-proxy P&L is a thin, fat-tailed,
correlation-spike-exposed stream, not a NAV-scale strategy.

See :mod:`dispersion_trade.data` (real basket loader + deterministic synthetic control with
a planted dispersion-carry knob) and :mod:`dispersion_trade.strategy` (rolling realized vol,
the dispersion gap, the carry P&L, Welch *t* / placebo null / win-rate / costs)."""

from . import data, strategy

__all__ = ["data", "strategy"]
