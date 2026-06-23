# References & literature map — Study 397 (Hurst-Regime)

## The claim under test

- **The Hurst exponent as a regime diagnostic.** H. E. Hurst (1951), *Long-term storage
  capacity of reservoirs*, Transactions of the American Society of Civil Engineers — the
  original rescaled-range (R/S) statistic, devised for Nile flood data. Mandelbrot & Wallis
  (1969), *Robustness of the rescaled range R/S in the measurement of noncyclic long-run
  statistical dependence*, popularised it for time series with long-range dependence. The
  market folklore reads it as a switch: **H > 0.5 = persistent / trending** (momentum should
  pay), **H < 0.5 = anti-persistent / mean-reverting** (fade should pay), **H ≈ 0.5 = random
  walk**. Edgar Peters' *Fractal Market Analysis* (1994) and *Chaos and Order in the Capital
  Markets* (1991) are the canonical popularisations that turned the exponent into a
  trade-the-regime recipe.
- **The trading folklore.** Countless blog posts, quant-forum threads and vendor indicators
  pitch a "Hurst regime filter": compute a rolling Hurst, trend-follow when it is above 0.5 and
  mean-revert when it is below, and you supposedly harvest the right premium in every market
  state. The pitch is seductive because it promises a *self-diagnosing* market — let the
  geometry of the series tell you which strategy to run.

## Why the premise is fragile before any P&L — the estimator

- **R/S has a documented small-sample / short-memory bias.** Andrew W. Lo (1991), *Long-term
  memory in stock market prices*, Econometrica 59(5) — shows classical R/S is badly biased by
  short-range dependence and conditional heteroskedasticity, spuriously reporting long memory
  (H > 0.5) where none exists, and proposes a modified, autocorrelation-robust R/S that mostly
  erases the apparent long memory in equity returns. Teverovsky, Taqqu & Willinger (1999), *A
  critical look at Lo's modified R/S statistic*, J. Statistical Planning & Inference, and Weron
  (2002), *Estimating long-range dependence: finite sample properties and confidence
  intervals*, Physica A, quantify how much the **finite-window** estimate is biased upward — the
  reason a trailing 1-year R/S Hurst on real markets sits pinned above 0.5 and almost never
  signals the mean-reverting regime the claim needs.
- **Detrended Fluctuation Analysis (DFA), the modern alternative.** Peng et al. (1994),
  *Mosaic organization of DNA nucleotides*, Phys. Rev. E — DFA is the estimator most quant
  write-ups now prefer; it shares the same finite-sample caveats. We use classical R/S
  (transparent, dependency-free) and note that DFA does not rescue the premise.

## Why a regime gate must be tested against a shuffled label, not just "did it make money"

- **Base rate / beta confound.** US equities (and most of the basket) drift up, so *any*
  mostly-long book makes money; the question is whether conditioning the **style** on H beats
  the same style mix with the regime label **shuffled**. This is Fisher's randomization logic
  (R. A. Fisher, 1935, *The Design of Experiments*) applied to a regime filter.
- **Data-snooping on a famous recipe.** Halbert White (2000), *A reality check for data
  snooping*, Econometrica, and Harvey, Liu & Zhu (2016), *…and the cross-section of expected
  returns*, Review of Financial Studies — a rule discovered and re-tuned ex-post needs a far
  higher bar than a naive in-sample Sharpe. We avoid tuning by fixing the gate and stressing it
  with a block bootstrap and a label placebo.
- **Block bootstrap for dependent P&L.** Künsch (1989) and Politis & Romano (1994), *The
  stationary bootstrap*, JASA — i.i.d. resampling destroys the volatility clustering that the
  Sharpe-difference inference must respect, so we resample contiguous blocks.

## Method lineage (the desk's shared engine)

- **R/S estimator + rolling Hurst.** [`data.hurst_rs`](../hurst_regime/data.py) and
  [`data.rolling_hurst`](../hurst_regime/data.py) — pure-trailing R/S over a 252-day window,
  no look-ahead.
- **Style signals + the gate.** [`strategy.trend_signal`](../hurst_regime/strategy.py),
  [`strategy.revert_signal`](../hurst_regime/strategy.py),
  [`strategy.regime_position`](../hurst_regime/strategy.py) — the trend / revert books and the
  H-gated switch, executed with a single 1-day lag and one-way costs in
  [`strategy.book_returns`](../hurst_regime/strategy.py).
- **Inference.** [`strategy.block_bootstrap_sharpe_diff`](../hurst_regime/strategy.py) (Sharpe
  difference with a block bootstrap CI) and
  [`strategy.placebo_shuffled_regime`](../hurst_regime/strategy.py) (the regime-label placebo
  null — the decisive test).
- **Deterministic synthetic control.**
  [`data.synthetic_prices`](../hurst_regime/data.py) (fractional-Gaussian path with a *known*
  Hurst, to validate the estimator) and
  [`data.synthetic_regimes`](../hurst_regime/data.py) (an alternating-regime path with a
  planted-edge knob — the gate's power test). The offline core runs with no network: it
  confirms the estimator recovers a planted H, the gate banks a planted regime edge, and the
  gate manufactures **nothing** when the planted edge is zero.

## Data sources used here

- **yfinance** daily adjusted closes for SPY (headline) + QQQ, GLD, TLT, EFA, 1995-01-03 →
  2026-06-18, cached under `_cache/prices.csv`. All headline numbers are pinned in
  [`docs/results.md`](results.md) and reproduced by [`examples/verify.py`](../examples/verify.py).

## Related desk studies

- **[Study 106 — Supertrend](../../106-supertrend/)** and **[Study 210 — Crypto-Trend](../../210-crypto-trend/)**:
  the trend-following styles the gate switches *into* — whether a trend filter earns its keep
  on its own.
- **[Study 119 — Real-Rate-Regime](../../119-real-rate-regime/)** and
  **[Study 384 — ISM-PMI-Regime](../../384-ism-pmi-regime/)**: other "regime filter" claims —
  does conditioning a strategy on a macro/state variable add anything beyond the unconditional
  premium? Same question, different gate.
- **[Study 184 — Williams-Fractals](../../184-williams-fractals/)**: another "the geometry of the
  series tells you what to do" indicator from the fractal-markets tradition.
