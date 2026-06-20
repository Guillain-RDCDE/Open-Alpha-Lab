# References & literature map — Study 349 (Regime-Dependence)

## The methodological claim under test

- **The trap.** A strategy that posts a strong *full-sample* Sharpe may owe that number
  to a single favourable regime — and a single Sharpe averages the regimes away, hiding
  the dependence. The testable methodological hypothesis: **decade-by-decade
  (regime-conditional) performance, with a test of the cross-regime *difference*,
  separates a durable edge from a one-era bet that a full-sample number cannot.** This is
  a research-method demo, not a folklore teardown.
- **"Ruling one decade" as the canonical failure mode.** The 60/40 portfolio's spectacular
  2010s (a 40-year bond bull market layered under a US equity bull) is the textbook
  example — see the post-2022 "death of 60/40" coverage after a single bad year exposed how
  much of the record rode falling rates. Study 144 (Permanent-Portfolio) and Study 97
  (Balancing-Act) on this desk test the *strategy*; here the regime split is the *measuring
  instrument*.

## Why regime-conditioning matters — the literature

- **Backtest overfitting / data-snooping.** Bailey, Borwein, López de Prado & Zhu (2014),
  *Pseudo-Mathematics and Financial Charlatanism: The Effects of Backtest Overfitting on
  Out-of-Sample Performance* (Notices of the AMS); López de Prado (2018), *Advances in
  Financial Machine Learning* (Wiley) — the "deflated Sharpe ratio" and the danger of a
  Sharpe that owes its level to selection. Fitting to one era is a special case: the regime
  *is* the in-sample window.
- **Structural breaks & parameter instability.** Andrews (1993), *Tests for Parameter
  Instability and Structural Change with Unknown Change Point* (Econometrica); Pesaran &
  Timmermann (2002), *Market Timing and Return Predictability under Model Instability* — the
  formal case that an edge estimated on one regime need not persist.
- **Regime-switching models.** Hamilton (1989), *A New Approach to the Economic Analysis of
  Nonstationary Time Series and the Business Cycle* (Econometrica) — the canonical
  two-state Markov-switching model; Ang & Bekaert (2002), *International Asset Allocation
  with Regime Shifts* (Review of Financial Studies). The synthetic generator here is a
  deliberately simple, *known-truth* version of a regime-switch process.
- **Time-series momentum (the durable real comparator).** Moskowitz, Ooi & Pedersen (2012),
  *Time Series Momentum* (Journal of Financial Economics) — trend-following's persistence
  *across* markets and decades, the reason it serves as the desk's "regime-robust" example.
- **The 60/40 record's regime dependence.** Asness, Israelov & Liew (2011),
  *International Diversification Works (Eventually)*; the broad post-2022 literature on how
  much of the 60/40's risk-adjusted record was a 40-year decline in interest rates rather
  than a structural property of the mix.

## Method lineage (the desk's shared engine)

- **HAC / Newey-West t-stat.** Newey & West (1987), *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix* (Econometrica) —
  [`strategy.hac_tstat`](../regime_dependence/strategy.py), applied on the level and on the
  drop-best-decade re-estimate.
- **Circular block bootstrap.** Politis & Romano (1992), *A circular block-resampling
  procedure for stationary data* — preserves autocorrelation when CI-ing the cross-regime
  mean gap ([`strategy.block_bootstrap_ci`](../regime_dependence/strategy.py)).
- **Positive control as a harness requirement.** A pipeline that cannot detect a *planted*
  regime-fitted edge proves nothing by finding nothing — the synthetic
  [`data.synthetic_regimes`](../regime_dependence/data.py) plants both a durable edge and a
  one-regime edge so the lens can be scored against ground truth (METHODOLOGY → *The
  inference bar*).

## Data sources used here

- **Yahoo! Finance** (via `yfinance` and the shared `quantlab.data` loader), total-return
  monthly closes for `^SP500TR` (equity) and `IEF` (7–10y Treasuries). Window
  2002-08 → 2026-05, pinned with an as-of date (2026-05-31, last full month) and a content
  fingerprint (see [`docs/results.md`](results.md)). The offline reproducible core and the
  test-suite run on the deterministic [`data.synthetic_regimes`](../regime_dependence/data.py)
  generator, never the network. Adjustment mode is **total return** for both legs (labelled
  everywhere; `^SP500TR` carries dividends, `IEF` is total return).

## Related desk studies

- **[Study 97 — Balancing-Act](../../97-balancing-act/)** and
  **[Study 144 — Permanent-Portfolio](../../144-permanent-portfolio/)**: teardowns of the
  *strategies* (60/40; Browne's four-way mix). Study 349 borrows their tape but asks a
  *methodological* question — is the record stable across regimes, or fit to one?
- **[Study 119 — Real-Rate-Regime](../../119-real-rate-regime/)**: a real-rate *timing rule*
  (a strategy keyed on a regime). Distinct: there the regime drives the trade; here the
  regime split is the measuring instrument applied to *any* strategy.
- **[Study 350 — Dartboard-Portfolio](../../350-dartboard-portfolio/)**: the sibling
  research-method demo from this lot (a planted size-tilt control); same desk machinery.
