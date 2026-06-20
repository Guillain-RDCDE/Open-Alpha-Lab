# References & literature map — Study 328 (Benford-Law)

## The law itself

- **Benford, F. (1938).** *The Law of Anomalous Numbers* (Proceedings of the American
  Philosophical Society). The empirical first-digit law: digit `d` leads with probability
  `log10(1 + 1/d)` — ~30.1% ones down to ~4.6% nines.
- **Newcomb, S. (1881).** *Note on the Frequency of Use of the Different Digits in
  Natural Numbers* (American Journal of Mathematics). The original observation, decades
  before Benford.
- **Hill, T. P. (1995).** *A Statistical Derivation of the Significant-Digit Law*
  (Statistical Science). The modern theorem: the law is the unique scale- and
  base-invariant distribution; mixtures of distributions converge to it. Crucially, it
  holds for quantities spanning *several orders of magnitude on a log scale* — the exact
  property a decades-long price has and a daily return does not.

## The forensic tradition — what Benford is actually used for

- **Nigrini, M. J. (1996).** *A Taxpayer Compliance Application of Benford's Law*
  (Journal of the American Taxation Association). The foundational forensic-accounting
  use: fabricated figures deviate from Benford.
- **Nigrini, M. J. (2012).** *Benford's Law: Applications for Forensic Accounting,
  Auditing, and Fraud Detection* (Wiley). Source of the Mean Absolute Deviation (MAD)
  conformity thresholds we use (close < 0.006, acceptable < 0.012, marginal < 0.015,
  nonconformity above) — implemented in [`strategy.mad`](../benford_law/strategy.py).
- **Varian, H. (1972).** *Benford's Law* (The American Statistician). An early note on
  using the law to check the plausibility of reported data.

## Benford applied to financial / economic data — the claim under test

- **Ley, E. (1996).** *On the Peculiar Distribution of the U.S. Stock Indices' Digits*
  (The American Statistician). The leading digits of the DJIA and S&P 500 *index levels*
  broadly follow Benford — the empirical seed of the "stock prices obey Benford" idea.
- **Corazza, Ellero & Zorzi (2010).** *Checking Financial Markets via Benford's Law: the
  S&P 500 Case.* Tests index returns/prices against the law; finds returns deviate.
- **Nigrini & Miller (2009).** *Data Diagnostics Using Second-Order Tests of Benford's
  Law* (Auditing: A Journal of Practice & Theory). The method extensions behind the
  forensic screen.
- **Shi, Ausloos & Zhu (2018) / and related "Benford for fraud detection in markets"
  literature.** The leap the *trading* folklore makes: that a Benford deviation on a
  name's price flags trouble and should predict lower forward returns. This study tests
  that leap directly — a cross-sectional sort on a trailing-window deviation score, with
  an honest HAC *t* on the long-conforming / short-deviant spread.

## Why a single name's price need not be Benford (and why this matters)

- A quantity is Benford only when it ranges over *several decades* on a log scale. The
  S&P 500 *index over a century* qualifies (Ley 1996). A *single ETF over 30 years* (SPY:
  ~\\$43 → ~\\$760, only ~1.2 decades) does **not** — it dwells in the 400s for years, so
  4s are over-represented. The "deviation" of a single name is therefore largely a
  statement about *the width of its price range*, not its integrity. This is the central
  confound the teardown isolates.

## Method lineage (the desk's shared engine)

- **Newey & West (1987).** *A Simple, Positive Semi-Definite, Heteroskedasticity and
  Autocorrelation Consistent Covariance Matrix* (Econometrica) — the HAC *t* in
  [`strategy.hac_tstat`](../benford_law/strategy.py).
- **Politis & Romano (1994); Künsch (1989).** The block bootstrap — the circular
  block-bootstrap CI in [`strategy.block_bootstrap_ci`](../benford_law/strategy.py),
  which preserves the autocorrelation an i.i.d. resample would destroy.
- **Harvey, Liu & Zhu (2016).** *…and the Cross-Section of Expected Returns* (RFS) — the
  multiple-testing lens on any cross-sectional sort dressed up as a new "anomaly".

## Data sources used here

- **Real:** the desk's shared parquet cache (`quantlab.data`) — SPY/QQQ/… split-adjusted
  daily closes (split mode chosen deliberately: a split rewrites leading digits, so we
  use the level a holder saw between splits), plus the shared `daily_panel.parquet`
  (current-membership large-cap return panel). **Survivorship is named on the Signal
  axis**: that panel is current constituents reconstructed from returns, so any apparent
  cross-sectional edge is contaminated by both survivorship and a fabricated price base.
  Headline numbers are pinned with an as-of date + fingerprint (see [`results.md`](results.md)).
- **Offline core / tests:** the deterministic [`data.synthetic_benford`](../benford_law/data.py)
  positive control and [`data.synthetic_nonbenford`](../benford_law/data.py) anti-control,
  never the network.

## Related desk studies

- Cross-sectional sorts with honest HAC inference and survivorship guards run throughout
  the factor-zoo lots; this study is the forensic/methodology cousin — it asks whether a
  famous *audit* tool carries any *return* information, and finds the apparent edge is a
  price-range-and-survivorship artifact, not a signal.
