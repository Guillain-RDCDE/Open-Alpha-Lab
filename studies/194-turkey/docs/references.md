# References & literature map — Study 194 (Turkey)

## The claim under test

- **Brockman, P., & Michayluk, D. (1998).** *The persistent holiday effect: additional
  evidence.* Applied Economics Letters, 5(4), 205–209. Documents positive pre-holiday
  returns in multiple international markets; Thanksgiving is cited as one of the
  strongest cases in the US pre-holiday calendar. The study motivates testing
  Wednesday-before and Friday-after Thanksgiving specifically.

- **Ariel, R. A. (1990).** *High stock returns before holidays: existence and evidence
  on possible causes.* Journal of Finance, 45(5), 1611–1626. Foundational paper on
  pre-holiday effects across all US market holidays; shows the pre-holiday day delivers
  returns several times the unconditional mean. The Thanksgiving pre-holiday Wednesday
  is a specific instance of this broader claim.

## Replications and disputes

- **Pettengill, G. N. (1989).** *Holiday closing and security returns.* Journal of
  Financial Research, 12(1), 57–67. Early US study showing elevated pre-holiday
  returns; notes that Thanksgiving-week effects are among the more consistent findings
  but do not survive in all sub-periods.

- **Lakonishok, J., & Smidt, S. (1988).** *Are seasonal anomalies real? A ninety-year
  perspective.* Review of Financial Studies, 1(4), 403–425. Uses DJIA 1897–1986;
  documents pre-holiday and turn-of-month effects. The Thanksgiving holiday is one of
  the eight US exchange holidays and is included in their pre-holiday tabulations.

- **Białkowski, J., Etebari, A., & Wisniewski, T. P. (2012).** *Fast profits: Investor
  sentiment and stock returns during Ramadan.* Journal of Banking & Finance, 36(3),
  835–845. Provides cross-cultural context: religious / cultural calendar effects on
  prices are documented in many markets, but magnitudes decay post-publication and are
  absent in the most liquid indices.

## Why holiday effects are expected to be fragile

- **McLean, R. D., & Pontiff, J. (2016).** *Does Academic Publication Destroy Stock
  Return Predictability?* Journal of Finance, 71(1), 5–32. Demonstrates that average
  post-publication return to published anomalies decays by ~58%. Pre-holiday effects
  published in the early 1990s are among the candidates for this post-publication decay.

- **Fama, E. F. (1991).** *Efficient Capital Markets: II.* Journal of Finance, 46(5),
  1575–1617. Discusses calendar anomalies as potential inefficiencies but notes their
  tendency to disappear once exploited or widely known.

## Multiple-comparisons and inference

- **Harvey, C. R., Liu, Y., & Zhu, H. (2016).** *...and the Cross-Section of Expected
  Returns.* Review of Financial Studies, 29(1), 5–68. Documents t-stat inflation in
  the factor zoo; the applicable lesson here is that selecting "Thanksgiving" from a
  broader holiday set without a prior inflation adjustment is a form of calendar mining.

- **Bonferroni, C. E. (1936).** *Teoria statistica delle classi e calcolo delle
  probabilità.* The Bonferroni correction applied here with k=2 (two primary
  hypotheses: Wednesday-before and Friday-after) sets the significance threshold at
  p < 0.025 for α = 0.05.

- **Newey, W., & West, K. (1987).** *A Simple, Positive Semi-Definite, Heteroskedasticity
  and Autocorrelation Consistent Covariance Matrix.* Econometrica, 55(3), 703–708.
  HAC t-stat used in `strategy._hac_tstat` for robust inference on each day-group.

## Related desk studies

- **[Study 95 — Holiday-Cheer](../../95-holiday-cheer/)**: pools all US market
  pre-holidays; a more powerful (but less precise) version of the same broad hypothesis.
  The desk's verdict for that study provides the baseline against which Thanksgiving
  specifically can be compared.

- **[Study 163 — Friday-13th](../../163-friday-13th/)**: another calendar date singled
  out from a pool of similar dates; same small-n reckoning and same Bonferroni approach.

- **[Study 48 — Groundhog](../../48-groundhog/)**: seasonal calendar effect with even
  fewer annual events; illustrates how thin-event-rate studies are systematically
  underpowered regardless of the strength of the prior.

- **[Study 82 — Witching-Hour](../../82-witching-hour/)**: expiration-day calendar
  effect; the quadruple-witching Friday is structurally similar to Black Friday as
  a "special trading session" with anomalous volume and thin activity.
