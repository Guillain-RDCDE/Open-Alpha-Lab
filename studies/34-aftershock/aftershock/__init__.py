"""Study 34 — Aftershock: post-earnings-announcement drift (PEAD), earnings momentum.

A stock's price keeps drifting in the direction of its earnings *surprise* for weeks after the
announcement (Ball-Brown 1968; Bernard-Thomas 1989), so a book that goes long positive-surprise names and
short negative-surprise names, rolled as earnings land, should harvest that drift. PEAD is one of the most
robustly documented anomalies in finance — but its tradability is limited by costs and by the fact that it
concentrates in the small, illiquid names that cost the most to trade.

**Data note.** A credible PEAD backtest needs years of reported-earnings dates + surprises, which no free
source supplies in this sandbox (yfinance exposes only ~6-8 reported quarters per name). So this study
follows the desk's *pending-fetch* pattern (cf. Study 27 Steamroller): a fully-offline synthetic control
that proves the machinery, a literature-grounded verdict, and a real run that is **pre-registered and
pending** an earnings-history fetch.
"""

from . import costs, data, extension, strategy  # noqa: F401
