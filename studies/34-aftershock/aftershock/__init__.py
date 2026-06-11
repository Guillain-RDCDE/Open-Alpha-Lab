"""Study 34 — Aftershock: post-earnings-announcement drift (PEAD), earnings momentum.

A stock's price keeps drifting in the direction of its earnings *surprise* for weeks after the
announcement (Ball-Brown 1968; Bernard-Thomas 1989), so a book that goes long positive-surprise names and
short negative-surprise names, rolled as earnings land, should harvest that drift. PEAD is one of the most
robustly documented anomalies in finance — but its tradability is limited by costs and by the fact that it
concentrates in the small, illiquid names that cost the most to trade.

**Data.** The real run uses cached EDGAR quarterly EPS (the announcement date is the *earliest filing*
per quarter) turned into a **SUE via a seasonal random walk** — ``surprise_q = eps_q − eps_{q−4}``,
standardised by the stock's own trailing surprise dispersion (the classic Bernard-Thomas measure, no
analyst estimate needed) — and the cached S&P 500 price panel (split/dividend-adjusted closes). A
fully-offline synthetic control (drift + null) proves the machinery and backs the test-suite. The real
panel is *current* index membership, so it carries survivorship bias — stated wherever a real number
appears.
"""

from . import costs, data, extension, strategy  # noqa: F401
