"""Study 16 — Storm-Shy: does scaling exposure down when markets get loud actually pay?

The desk's first *green*. After fifteen teardowns, the residual that kept surfacing as the only
real thing inside a fake one — the **vol-targeting overlay** that Study 12 (Paper-Prophet) found
hiding inside ARIMA+GARCH — gets its own study, as the *hero* this time, run through the same
brutal protocol. The reusable pieces, in the desk's usual split:

    * :mod:`data` — the price tape the overlay runs on: a synthetic generator with **baked-in
      volatility clustering** (a two-state Markov vol regime around a *constant* drift, so high-vol
      stretches carry the same expected return but far more risk — the exact condition under which
      scaling exposure by inverse variance raises the Sharpe), plus a cache-only reader that pulls
      real daily SPY/factor closes and reduces them to the same single-column tape. A *flat-vol*
      tape (one regime) is the null: no clustering to read, so the overlay must add nothing.
    * :mod:`vol` — the engine of the whole thing: realized-vol estimators (trailing window and
      RiskMetrics EWMA) and, load-bearing, :func:`vol.forecastability` — is tomorrow's variance
      predictable from today's? Volatility clustering (Mandelbrot 1963; Engle 1982 ARCH) is the
      single most robust stylized fact in markets; if it weren't, the overlay could not work.
    * :mod:`strategy` — the overlay itself: ``w_t = σ_target / σ̂_{t−1}`` (Moreira–Muir 2017),
      using **only past information**, capped at a realistic max leverage, plus buy-&-hold vs
      vol-managed summaries, turnover and a cost sweep. Sharpe is scale-invariant, so the
      comparison is already fair to average leverage — the honest catch lives in :mod:`decompose`.
    * :mod:`decompose` — the inference that earns the stamps: (1) the **Moreira–Muir spanning
      alpha** (regress managed on buy-&-hold; a positive HAC-significant α means the overlay
      expands the mean–variance frontier); (2) a **bootstrap CI on the Sharpe gain**; (3) the
      honest counter — a CRRA **certainty-equivalent** test at matched unconditional risk
      (Cederburg–O'Doherty–Wang–Yan 2020), which shrinks the gain to its real, smaller size rather
      than the headline; (4) **decay & equal-risk** checks so the win is not just a lucky window or
      disguised de-risking. The verdict it lands: Signal `REAL`, Tradability `INVESTABLE`,
      "Free lunch?" `RISK-MANAGED` — a real edge, honestly bounded.
"""
