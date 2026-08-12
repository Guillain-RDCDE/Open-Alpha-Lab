"""Study 848 — Costco Hot-Dog Index 🌭.

The $1.50 hot-dog-and-soda combo has been nominally frozen since ~1985. This package
tests the folk "anti-inflation" narrative: (a) the combo's *real* price has collapsed
with CPI (a mechanical identity we quantify), and (b) whether Costco's (`COST`)
pricing-power / membership model gives its stock a distinctive edge that lets it outrun
CPI and SPY — in particular, whether COST's beta to inflation *surprises* differs from
consumer staples (`XLP`).

* :mod:`hotdog_index.data`     — real COST/SPY/XLP tape (yfinance, cached) + the real
  monthly CPIAUCSL series (BLS ``CUSR0000SA0``, fetched from the BLS API and cached, with
  the same real values embedded as an offline fallback) + a seeded synthetic control.
* :mod:`hotdog_index.strategy` — the real-price identity, the total-return race, the
  inflation-surprise regressions (Newey-West), placebo, sub-era cut, a costed timer and
  the machinery-proof synthetic detector.
"""

from __future__ import annotations

__all__ = ["data", "strategy"]
