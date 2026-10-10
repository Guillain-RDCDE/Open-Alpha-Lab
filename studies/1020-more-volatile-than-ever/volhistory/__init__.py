"""Study 1020 — More Volatile Than Ever?

Every few years the same sentence comes back: *markets are more volatile than ever.* The culprit
changes — program trading in 1987, day traders in 1999, high-frequency trading after the 2010
flash crash, ETFs, then social media and meme stocks — but the claim does not. Its sharper cousin
says the crashes themselves are getting more frequent.

This study puts a century of the US market on the bench and asks three separate questions:

1. **Is volatility trending up?** Trend tests on log realised volatility that are honest about
   how persistent volatility is (Newey-West, Kiefer-Vogelsang-Bunzel fixed-b, and a residual
   block bootstrap), on three tapes and four eras.
2. **Are extreme days getting more frequent?** Counts of 3σ and 4σ days per decade, against a
   fixed long-run σ *and* a trailing σ, with exact Poisson intervals and a robust count trend.
3. **Why does it feel that way?** The points-versus-percent mechanism (a constant-percent index
   produces ever-larger point moves as it grows) and the recency of what people remember.

The tradable translation is a risk manager's choice: forecast next period's volatility from the
long-run average, from recent history, or from an extrapolated trend.

- :mod:`volhistory.data` — the frozen bundled tapes (Fama-French monthly total return, skfolio
  daily S&P 500 price index, arch daily S&P 500 OHLC) and the synthetic stochastic-volatility
  generator with a ``signal_strength`` knob.
- :mod:`volhistory.strategy` — the measurement, the inference, the forecast race and the
  pre-registered verdict.
"""

from __future__ import annotations

__all__ = ["data", "strategy"]
