# References & literature map — Study 346 (Multiple-Testing)

## The problem under test — many tests, inflated p-values

- **The core fact.** If you run *M* independent tests each at significance α, the chance of
  *at least one* false positive (the family-wise error rate, FWER) is 1 − (1 − α)^M, which
  rushes to 1 as *M* grows. With M = 38 and α = 0.05 that is already ≈ 0.83. Quoting the
  best test's own *t*-stat without saying how many you ran is the original sin of empirical
  finance.
- **In finance specifically.** Harvey, Liu & Zhu (2016), *…and the Cross-Section of Expected
  Returns* (Review of Financial Studies) — argue that with hundreds of published "factors"
  tested, the conventional *t* > 2 hurdle is far too low; they recommend a multiple-testing
  hurdle nearer *t* > 3. Harvey & Liu (2014, 2015, 2020) develop the multiple-testing and
  Bayesianized-p / haircut-Sharpe framework for exactly this.
- **The data-snooping reckoning.** White (2000), *A Reality Check for Data Snooping*
  (Econometrica); Sullivan, Timmermann & White (1999), *Data-Snooping, Technical Trading
  Rule Performance, and the Bootstrap* (Journal of Finance) — the best of ~7,800 technical
  rules evaporates once you correct for the search. Bailey, Borwein, López de Prado & Zhu
  (2014), *Pseudo-Mathematics and Financial Charlatanism* — the expected maximum Sharpe
  grows without bound in the number of trials.

## The correction procedures (the four horses we race)

- **Naive (|*t*| > 2).** No correction — reject each test on its own two-sided *t*. Controls
  the *per-comparison* error rate only; the family-wise rate is uncontrolled. This is the
  count a p-hacker quotes.
- **Bonferroni.** Bonferroni (1936); Dunn (1961), *Multiple Comparisons Among Means* (JASA).
  Reject when *p* ≤ α/M. Controls FWER, but conservatively (it ignores the dependence and
  the alternative distribution).
- **Holm (1979),** *A Simple Sequentially Rejective Multiple Test Procedure* (Scandinavian
  Journal of Statistics). The step-down refinement: sort the *p*-values ascending and reject
  *p*₍ₖ₎ while *p*₍ₖ₎ ≤ α/(M − k). Controls FWER and is **uniformly more powerful than
  Bonferroni** at the same guarantee — there is never a reason to prefer Bonferroni over
  Holm.
- **Benjamini-Hochberg (1995),** *Controlling the False Discovery Rate* (Journal of the
  Royal Statistical Society B). The step-up procedure that controls the *false discovery
  rate* — the expected fraction of rejections that are false — instead of the family-wise
  rate. Trades a small, controlled false-discovery rate for substantially more power; the
  right target when a few false leads are tolerable. (Benjamini & Yekutieli (2001) extend
  the guarantee to dependent tests.)

## The method lineage (the desk's shared engine)

- **HAC / Newey-West *t*-stat.** Newey & West (1987), *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix* (Econometrica) —
  [`strategy.hac_tstat`](../multiple_testing/strategy.py). Each effect's daily-return *t* is
  autocorrelation-robust before any correction is applied.
- **Circular block bootstrap.** Politis & Romano (1992), *A circular block-resampling
  procedure for stationary data* — the CI on each effect's mean
  ([`strategy.block_bootstrap_ci`](../multiple_testing/strategy.py)).
- **Inverse-normal quantile.** Acklam's rational approximation of the inverse normal CDF,
  used as a scipy-free fallback in [`strategy.alpha_to_t`](../multiple_testing/strategy.py).

## The calendar effects in the real battery

The 38 named effects (day-of-week and month-of-year dummies, turn-of-month, first/last days,
quarter-end, Santa rally, sell-in-May, opex-Monday, …) are the folklore this desk
investigates one study at a time. Studying them as a *family* is the point: any one of them
might look significant in isolation, which is exactly why they need a family-wise correction.

## Data sources used here

- **Yahoo! Finance** (via `yfinance` and the shared `quantlab.data` loader), SPY
  total-return daily closes, 1995–2026. All headline numbers are pinned with an as-of date
  (2026-05-29, last full month) and a content fingerprint (see
  [`docs/results.md`](results.md)). The offline reproducible core and the test-suite run on
  the deterministic [`data.synthetic_battery`](../multiple_testing/data.py) generator, never
  the network.

## Related desk studies

- **[Study 343 — Data-Mining-Roulette](../../343-data-mining-roulette/)**: spins random
  *rules* and runs Bonferroni + White's Reality Check on the *single best* of N. Study 346
  is the complementary angle — the **correction-procedure comparison** (FWER vs FDR) on a
  *battery of many named effects with a known truth mix*, not the best-of-N maximum.
- **[Study 344 — Backtest-Overfitting](../../344-backtest-overfitting/)**: grid-searches one
  strategy's parameters with the Deflated Sharpe Ratio + PBO. 346 corrects across an
  *unrelated family* of hypotheses rather than one strategy's tuning grid.
- **[Study 97 — Balancing-Act](../../97-balancing-act/)**: same desk machinery (HAC *t*,
  block-bootstrap CIs, excess-vs-excess) on an allocation rule.
