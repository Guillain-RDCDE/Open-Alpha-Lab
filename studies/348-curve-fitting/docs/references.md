# References & literature map — Study 348 (Curve-Fitting)

## The claim under test

This is a **research-method demo**, so the "claim" is the seductive *practice* the study
dismantles: *tune a strategy's parameters until the backtest shines, and the winning
configuration will keep working when you trade it.* The moving-average crossover is just the
vehicle — two integer windows, a big plausible grid, a single optimise-then-validate
protocol. The testable hypothesis: **the in-sample-best parameterisation carries a real,
out-of-sample edge.** The null we plant says it does not — the IS Sharpe is selection, and
it reverts to noise OOS.

## Why the in-sample best overstates the truth — the snooping mechanism

- **Data snooping / the multiple-testing inflation.** Lo & MacKinlay (1990), *Data-Snooping
  Biases in Tests of Financial Asset Pricing Models* (Review of Financial Studies). Searching
  a grid and reporting the maximum is *selecting on the dependent variable*; the max of many
  noisy statistics is biased upward even when every underlying effect is zero. The grid-size
  sweep in this study is that bias made visible.
- **The Reality Check / SPA test.** White (2000), *A Reality Check for Data Snooping*
  (Econometrica); Hansen (2005), *A Test for Superior Predictive Ability* (JBES). The
  formal correction for "I tried N rules and kept the best" — the bootstrap distribution of
  the *maximum* statistic across the grid, against which a single winner must be judged.
- **The backtest-overfitting literature.** Bailey, Borwein, López de Prado & Zhu (2014),
  *Pseudo-Mathematics and Financial Charlatanism: The Effects of Backtest Overfitting on
  Out-of-Sample Performance* (Notices of the AMS); Bailey & López de Prado (2014), *The
  Deflated Sharpe Ratio* (Journal of Portfolio Management). They quantify how the *number of
  trials* inflates the best in-sample Sharpe and how to deflate it — the precise quantity our
  IS−OOS shrinkage table measures empirically.
- **Harvey, Liu & Zhu (2016), *…and the Cross-Section of Expected Returns* (RFS)** — the
  factor-zoo reckoning: with hundreds of strategies tested, a *t* of 2 is far too low a bar.
  The same logic applies within a single grid search.

## Moving-average crossovers (the vehicle, not the subject)

- **Brock, Lakonishok & LeBaron (1992), *Simple Technical Trading Rules and the Stochastic
  Properties of Stock Returns* (Journal of Finance)** — the canonical MA-crossover study, and
  itself a cautionary tale: Sullivan, Timmermann & White (1999), *Data-Snooping, Technical
  Trading Rule Performance, and the Bootstrap* (Journal of Finance) re-ran it through White's
  Reality Check and showed the apparent profits shrink sharply once the universe of rules
  searched is accounted for. That is exactly this study's point, on exactly this rule family.

## The honest controls and method lineage (the desk's shared engine)

- **Out-of-sample / walk-forward validation.** The discipline of holding back data the
  optimiser never sees — the minimal antidote, and the protocol this study runs.
- **HAC / Newey-West t-stat.** Newey & West (1987), *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix* (Econometrica) —
  [`strategy.hac_tstat`](../curve_fitting/strategy.py).
- **Circular block bootstrap.** Politis & Romano (1992), *A circular block-resampling
  procedure for stationary data* — preserves autocorrelation when CI-ing the OOS Sharpe
  ([`strategy.block_bootstrap_sharpe_ci`](../curve_fitting/strategy.py)).
- **Alpha vs beta.** A long/flat timing rule's OOS Sharpe can be almost all market exposure;
  [`strategy.alpha_vs_buyhold`](../curve_fitting/strategy.py) HAC-tests the rule minus
  buy-and-hold to strip the beta out — decisive on the real SPY tape.

## Data sources used here

- **Yahoo! Finance** (via `yfinance` and the shared `quantlab.data` loader), SPY daily
  total-return closes, 1993–2026. The headline run is pinned with an as-of date
  (2026-05-29, the last full month) and a content fingerprint (see
  [`docs/results.md`](results.md)). The offline reproducible core and the test-suite run on
  the deterministic [`data.synthetic_series`](../curve_fitting/data.py) generator, never the
  network.

## Related desk studies

- The bench's single-rule MA / trend studies — **Faber timing**, the **golden/death cross**,
  **Supertrend** — ask "does *this one* rule work?". Study 348 asks the meta-question they
  all depend on: does *choosing the best* rule from a grid of past performance tell you
  anything about the future? It is why the desk refuses to grid-search a verdict.
- **[Study 350 — Dartboard-Portfolio](../../350-dartboard-portfolio/)** and
  **[Study 97 — Balancing-Act](../../97-balancing-act/)**: same desk machinery
  (excess-vs-excess races, block-bootstrap CIs, HAC inference, synthetic positive control).
