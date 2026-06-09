"""Study 26 — Sand-Castle: the 'optimal' stat-arb portfolio that washes away out of sample.

The ninth study mined from Kakushadze & Serur's *151 Trading Strategies* (strategy 3.18, statistical
arbitrage via optimization). The steelman is mean-variance optimization: given expected stock returns
``E`` and a covariance ``C``, the Sharpe-maximizing dollar-neutral weights are ``w ~ C^{-1} E``. Here
``E`` is a short-horizon residual mean-reversion signal. The reversion is real -- but inverting a noisy
*sample* covariance turns a stable signal into an unstable, self-defeating portfolio (Michaud's
"optimization is error-maximization"). We run it through the desk's protocol. The reusable pieces:

    * :mod:`data` — a synthetic panel whose idiosyncratic residual mean-reverts one day to the next (the
      signal), plus a no-reversion null and a cache-only S&P 500 reader (capped so the sample covariance
      is estimable).
    * :mod:`statarb` — the engine: a causal residualiser, the reversion signal ``E``, the sample
      covariance, the ``C^{-1}E`` **optimal weights** (with optional shrinkage), the **naive** weights,
      and the signal's cross-sectional information coefficient.
    * :mod:`strategy` — the daily, causal optimized vs naive books, net of (large) turnover cost.
    * :mod:`decompose` — the inference: **optimizer-vs-naive** (does ``C^{-1}`` help net of cost?), the
      **weight instability** (condition number, concentration), and the **in-sample-vs-causal** overfit
      gap (the sand castle). The verdict it lands: Signal `REAL` (the reversion is genuine), Tradability
      `MIRAGE` (daily turnover and an error-maximizing inverse), optimization-helps `BUSTED`.
    * :mod:`extension` — the beat-7 worked complement: **covariance shrinkage** -- the textbook fix -- only
      lets the optimizer climb back *toward* the naive book; the best it can do is stop optimizing.
"""
