"""Study 1024 — The Drawdown You Were Promised.

A risk model that knows an asset's mean and volatility is supposed to know its drawdown
risk too: for a Brownian motion with drift the whole distribution of the maximum drawdown
over a horizon is a solved problem (Magdon-Ismail, Atiya, Pratap & Abu-Mostafa, 2004), and
risk budgets, Calmar normalisations and "how bad can it get" slides are built on that
promise. This study writes the promise down *before* each window — from the preceding data
only — and then checks what the tape delivered.

- :mod:`promised.data` — the bundled real tapes (S&P 500 and Nasdaq price indices, the
  Fama-French market total return, a 20-stock survivor panel) and the deterministic
  synthetic generator whose ``signal_strength`` knob dials volatility clustering and fat
  tails in and out.
- :mod:`promised.strategy` — exact simulation of the discretely-sampled GBM drawdown, the
  stationary block bootstrap and GARCH-t challengers, the ex-ante coverage test, the
  multiplier a risk manager would need, and the pre-registered verdict.
"""

from __future__ import annotations

__all__ = ["data", "strategy"]
