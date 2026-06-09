"""Study 25 — Clean-Slate: does stripping out the market make momentum behave?

The eighth study mined from Kakushadze & Serur's *151 Trading Strategies* (strategy 3.7, residual
momentum) -- and the natural sequel to [Study 24](../../24-stampede/), which found total-return momentum
real-in-principle but faint and crash-prone. The steelman (Blitz, Huij & Martens 2011): run 12-1
momentum on each stock's *residual* return -- the part not explained by the market/factors -- and you
keep the premium while shedding the violent loser-rebound crash, because the crash lives in the
systematic (beta) exposure that residualising removes. We run it through the desk's protocol. The
reusable pieces:

    * :mod:`data` — a synthetic panel where the persistent momentum lives in the *idiosyncratic
      residual* (so total momentum is contaminated by beta dispersion, residual momentum is clean), plus
      a cache-only S&P 500 reader.
    * :mod:`momentum` — the engine: a causal (rolling-beta) **residualiser** and the 12-1 score on the
      residuals, with :func:`momentum.momentum_spread` reading the residual-winners-minus-losers premium.
    * :mod:`strategy` — the residual-WML book *and* the total-WML book ([Study 24](../../24-stampede/)'s
      factor) for a like-for-like comparison, vs the equal-weight market.
    * :mod:`decompose` — the inference: the residual-WML **CAPM alpha** (HAC), and the load-bearing
      **crash comparison** (residual vs total skew/worst-month/drawdown), plus decay and a bootstrap.
    * :mod:`extension` — the beat-7 worked complement: **stacking the two crash defences** -- residualise
      *and* vol-manage -- and reading the progressive taming of the tail.
"""
