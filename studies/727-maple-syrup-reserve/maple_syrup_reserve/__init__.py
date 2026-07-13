"""Study 727 — "the Quebec strategic maple-syrup reserve as a trade" (tested).

Quebec runs a *strategic maple-syrup reserve* — a barrel stockpile the producers' cartel
(PPAQ) uses to defend an administered bulk price, the one raided in the 2011–12 "Great
Canadian Maple Syrup Heist". Is any of this tradable? We test the strongest version:
the (hardcoded, cited, **approximate**) PPAQ bulk price vs the S&P/TSX, the only listed
name with real maple exposure (Rogers Sugar ``RSI.TO``, plus a sugar-futures ``SB=F``
placebo), and a sugaring-season (Feb–Apr) seasonal — with a placebo and costs net.

See :mod:`maple_syrup_reserve.data` (hardcoded maple price + yfinance proxies + a
deterministic synthetic-season control) and :mod:`maple_syrup_reserve.strategy`
(CAGR/vol/MDD, annual-excess *t*, Newey-West proxy alpha, per-month HAC *t*, the
season Welch test + block bootstrap, and a costed sugaring-season timer)."""

from . import data, strategy

__all__ = ["data", "strategy"]
