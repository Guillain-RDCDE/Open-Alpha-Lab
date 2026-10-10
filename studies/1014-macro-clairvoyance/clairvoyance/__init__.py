"""Study 1014 — The Perfect Macro Forecaster.

The macro-forecasting industry sells one premise: *if you knew where GDP, unemployment and
inflation were heading, you would know where stocks are heading.* This study does not ask
whether anyone can forecast the economy. It grants the forecast — perfectly, from the
final-vintage history — and measures what that clairvoyance is worth for timing US equities.

The answer depends almost entirely on *how far ahead* the oracle sees. A perfect forecast
of the coming quarter's economy turns out to be worth little or nothing, because the stock
market spends that quarter pricing the economy that comes *after* it. The value only shows
up when the oracle sees beyond the holding period — that is, when it knows something the
market is still guessing at.

- :mod:`clairvoyance.data` — the real tapes (statsmodels ``macrodata``, final vintage;
  Fama-French market total return and T-bills via ``arch``), both SHA-256 pinned through
  :mod:`quantlab.bundled`, and the deterministic synthetic generator the tests run on.
- :mod:`clairvoyance.strategy` — lead-lag correlations, oracle timing rules at every
  information horizon, block-bootstrap and rotation inference, the perfect-market ceiling,
  and the pre-registered verdict.
"""

from __future__ import annotations

__all__ = ["data", "strategy"]
