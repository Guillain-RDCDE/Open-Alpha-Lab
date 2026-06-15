# References — Study 189 (Double-Top / Double-Bottom)

## Primary academic literature

**Bulkowski, T. N. (2005).** *Encyclopedia of Chart Patterns* (2nd ed.). Wiley.
The canonical practitioner reference for double-top and double-bottom statistics.
Bulkowski documents break-even failure rates (~64% for double-tops), average declines
post-confirmation (~19%), and the importance of using the confirmed neckline break —
not just the two-peak formation — as the entry trigger.

**Lo, A. W., Mamaysky, H., & Wang, J. (2000).** Foundations of Technical Analysis:
Computational Algorithms, Statistical Inference, and Empirical Implementation.
*Journal of Finance, 55*(4), 1705–1765.
Formalises the smoothed nonparametric approach to detecting chart patterns (including
double-tops and double-bottoms) in price data and tests their predictive power over
1962–1996 US stock data.  Finds statistically significant patterns in a few cases
but does not account for transaction costs or out-of-sample stability.

**Dawson, E. R., & Steeley, J. M. (2003).** On the existence of visual technical
patterns in the UK stock market. *Journal of Business Finance & Accounting, 30*(1–2),
263–293.  Extends Lo et al. to UK equities; finds less consistent evidence for the
patterns' predictive value.

**Savin, G., Weller, P., & Zvingelis, J. (2007).** The predictive power of "head-and-
shoulders" price patterns in the U.S. stock market. *Journal of Financial Econometrics,
5*(2), 243–265.  Related study on another multi-peak chart pattern using rigorous
statistical testing; finds the pattern's predictive power essentially disappears after
appropriate data-snooping adjustments.

## Cost and frictions literature

**Bessembinder, H. (2003).** Issues in assessing trade execution costs. *Journal of
Financial Markets, 6*(3), 233–257.  Benchmark for realistic one-way transaction cost
estimation relevant to this study's cost sweep.

**Lesmond, D. A., Ogden, J. P., & Trzcinka, C. A. (1999).** A new estimate of
transaction costs. *Review of Financial Studies, 12*(5), 1113–1141.  Establishes
typical round-trip costs (spreads + commissions + market impact) for US equities.

## Statistical methodology

**Newey, W. K., & West, K. D. (1987).** A simple, positive semi-definite,
heteroskedasticity and autocorrelation consistent covariance matrix. *Econometrica,
55*(3), 703–708.  The Bartlett-kernel HAC estimator used for all t-statistics in this
study (implemented inline in :func:`strategy.summarize_pattern`).

**Romano, J. P., & Wolf, M. (2005).** Stepwise multiple testing as formalized data
snooping. *Econometrica, 73*(4), 1237–1282.  The data-snooping correction framework;
we use a simpler Bonferroni adjustment (|t| ≥ 3.0 for 6 tests at α = 5%) as a
conservative bound.

**Sullivan, R., Timmermann, A., & White, H. (1999).** Data-snooping, technical trading
rule performance, and the bootstrap. *Journal of Finance, 54*(5), 1647–1691.
Demonstrates how the uncorrected significance of many technical patterns inflates once
the full universe of rules tested is accounted for via White's Reality Check.

## Related Open-Alpha-Lab studies

- **Study 76 — Rice-Paper:** Candlestick reversal patterns (bullish/bearish engulfing,
  hammer, shooting star, doji) on the same daily OHLCV tape with the same random-day
  control framework.  Same conclusion: Signal = NONE, Tradability = MIRAGE.
- **Study 72 — Loaded-Dice:** Moving-average crossover scalp on 5-minute bars; the same
  "the chart pattern looks like it works until you run the honest symmetric test"
  narrative.
- **Study 127 — Williams %R:** A momentum oscillator study with a comparable analysis
  of popular technical signals against a random placebo.
