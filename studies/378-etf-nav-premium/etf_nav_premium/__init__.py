"""Study 378 — ETF NAV Premium/Discount (does buying below NAV earn the discount back?).

The folk claim: an ETF that trades **below its NAV** is "on sale" — the discount is a
free gap that closes, so buying the discount and waiting harvests it. We test it on three
liquid bond/EM/muni ETFs (HYG, EMB, MUB) where premium/discount is large and famous
(March-2020 bond-ETF dislocations are the canonical example).

True intraday iNAV/NAV is not on yfinance, so we CONSTRUCT a transparent **NAV proxy**: a
hedge-basket fair value for each ETF (a sister credit ETF + a rate leg), and define the
transient **premium/discount basis** as the ETF's cumulative return *residual* to that
fair value, detrended by a rolling mean. A "discount" fires when the basis z-score drops
below -1. We then ask the only question that pays: when you enter the discount (1-day lag)
and hold, do you **harvest the residual back** — net of beta to the hedge basket, and net
of the very real bond-ETF bid-ask?

The decisive finding is structural: the discount **does** partially mean-revert (the basis
is stationary), but the *hedged* harvest is a handful of basis points with a marginal,
horizon- and ETF-fragile t, and it is **smaller than the round-trip cost** of trading the
ETF plus its hedge. Worse, the deepest discounts cluster in **stress** (March 2020), exactly
when the arbitrage that "closes the gap" breaks and you cannot transact at NAV at all. Real
as a statistical wobble, busted as a free lunch.

See :mod:`etf_nav_premium.data` (real ETF loader + NAV-proxy basis + deterministic synthetic
control) and :mod:`etf_nav_premium.strategy` (discount detection, hedged-harvest inference,
placebo null, costs)."""

from . import data, strategy

__all__ = ["data", "strategy"]
