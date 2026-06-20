# References & literature map — Study 343 (Data-Mining-Roulette)

## The claim under test (and why it matters)

- **The core demonstration.** If you backtest enough rules, the *best* one will look great
  even when nothing is real. This is not folklore — it is a theorem about the distribution
  of the maximum of many noisy statistics. The study is a hands-on demonstration that the
  single best of N random rules, on a tape with **provably nothing to find**, routinely
  clears the naive |*t*| ≥ 2 bar — and that the only honest defence is to correct for the
  search.

## Data-snooping and backtest overfitting — the canonical literature

- **White, H. (2000), *A Reality Check for Data Snooping* (Econometrica 68(5)).** The
  foundational test: under the null that no model beats the benchmark, bootstrap the
  *maximum* performance statistic over the whole family and read off the snooping-adjusted
  *p*-value. Implemented here as
  [`strategy.reality_check`](../data_mining_roulette/strategy.py) on the stationary bootstrap.
- **Hansen, P. R. (2005), *A Test for Superior Predictive Ability* (JBES).** The SPA test —
  a more powerful refinement of White's Reality Check that down-weights poor and irrelevant
  alternatives. The natural next step beyond the Bonferroni / Reality-Check pair used here.
- **Sullivan, Timmermann & White (1999), *Data-Snooping, Technical Trading Rule
  Performance, and the Bootstrap* (Journal of Finance).** Applies the Reality Check to a
  universe of ~7,800 technical trading rules on the Dow — the direct ancestor of this
  study's "spin the roulette over a rule space" design.
- **Bailey, Borwein, López de Prado & Zhu (2014), *Pseudo-Mathematics and Financial
  Charlatanism* (Notices of the AMS) and *The Probability of Backtest Overfitting* (2017).**
  Show that with enough trials a backtest Sharpe of 2 is achievable on pure noise, and
  formalise the *deflated Sharpe ratio* and *PBO*. The intellectual core of "luck mimics
  skill."
- **Harvey, Liu & Zhu (2016), *…and the Cross-Section of Expected Returns* (RFS).** Argues
  that, given the hundreds of published "factors," a *t*-stat of 2 is far too low a bar; a
  multiple-testing-aware hurdle is closer to 3. Direct motivation for the Bonferroni axis.
- **Lo, A. & MacKinlay, A. C. (1990), *Data-Snooping Biases in Tests of Financial Asset
  Pricing Models* (RFS).** The early, definitive statement of how reusing the same data to
  both form and test hypotheses inflates significance.

## Method lineage (the desk's shared engine)

- **Newey, W. & West, K. (1987), *A Simple, Positive Semi-Definite, Heteroskedasticity and
  Autocorrelation Consistent Covariance Matrix* (Econometrica).** The HAC *t*-stat each
  rule reports — [`strategy.hac_tstat`](../data_mining_roulette/strategy.py).
- **Politis, D. & Romano, J. (1994), *The Stationary Bootstrap* (JASA).** The resampling
  scheme behind the Reality Check and the circular block-bootstrap CI — preserves the
  autocorrelation that i.i.d. resampling would destroy.
- **Bonferroni correction / Holm (1979).** The simple family-wise threshold |*t*| ≥
  Φ⁻¹(1 − 0.05/2N) used as the conservative reference bar in
  [`strategy.how_many_pass`](../data_mining_roulette/strategy.py).
- **Short-horizon mean reversion** (the faint *real* effect the roulette finds on SPY):
  Jegadeesh (1990), *Evidence of Predictable Behavior of Security Returns* (Journal of
  Finance); Lo & MacKinlay (1988). Documents that the real tape is not a clean null — which
  is exactly why the Reality Check, not the naive *t*, is the arbiter.

## Data sources used here

- **Yahoo! Finance** (via `yfinance` and the shared `quantlab.data` loader), SPY
  total-return daily closes, 1995–2026. The headline real run is pinned with an as-of date
  (2026-05-28, the last full month dropped of its partial bar) and a content fingerprint
  (see [`docs/results.md`](results.md)). The offline reproducible core and the test-suite
  run entirely on the deterministic
  [`data.synthetic_tape`](../data_mining_roulette/data.py) generator, never the network.

## Related desk studies

- **[Study 350 — Dartboard-Portfolio](../../350-dartboard-portfolio/)**: randomises *which
  stocks* you hold and races the distribution against the index. Study 343 randomises the
  *rule* — the entry/exit logic — and studies the distribution of the best backtest under
  the null. The pair brackets "randomness in markets": random *holdings* vs random *logic*.
- **[Study 301 — Triple-RSI](../../301-triple-rsi/)**: a single viral "90% win-rate" rule
  dismantled. Study 343 generalises the lesson — it is the *factory* that manufactures
  exactly such rules from noise.
- **[Study 97 — Balancing-Act](../../97-balancing-act/)**: the desk's reference for the
  shared inference machinery (HAC *t*, block-bootstrap CIs, excess-vs-excess races).
