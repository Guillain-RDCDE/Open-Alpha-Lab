"""Study 152 — Inflation-Hedge.

Does the common wisdom that "stocks are a real asset, so they hedge inflation"
hold up in the data?  Using the Shiller monthly S&P 500 panel (1871-present) we
test two related propositions:

1. **The Fisher hypothesis for stocks** — nominal equity returns should *rise*
   one-for-one with inflation, leaving *real* returns unaffected.
2. **Real-return resilience** — real equity returns in high-inflation regimes
   should be statistically indistinguishable from (or better than) those in
   low-inflation regimes.

The famous finding (Fama & Schwert 1977; Modigliani & Cohn 1979) is that stocks
are a **poor short-run inflation hedge**: the correlation of nominal returns with
inflation is negative or flat, so real returns fall when inflation spikes.  Over
very long horizons (20+ years) the picture is more nuanced — but at investment
horizons of 1-5 years the hedge breaks down.

This package provides:

- ``data.py``     — synthetic generator + Shiller real-data loader.
- ``strategy.py`` — regime-split, rolling-window, and Fisher-hypothesis tests.
"""
