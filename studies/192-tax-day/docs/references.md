# References & literature map — Study 192 (Tax-Day)

## The claim under test

- **Ogden, J. P. (1990).** *Turn-of-Month Evaluations of Liquid Profits and Stock Returns:
  A Common Explanation for the Monthly and January Effects.* Journal of Finance, 45(4),
  1259–1272. Documents a turn-of-month return premium and attributes part of it to
  liquidity infusions including IRA contribution deadlines and tax-refund disbursements.
  The mechanism: cash that investors receive or allocate around tax season flows into
  equities, temporarily boosting demand in the days before and around April 15.
- **Gu, A. Y. (2003).** *The Declining January Effect: Evidence from the U.S. Equity
  Markets.* Quarterly Review of Economics and Finance, 43(2), 395–404. Reviews the
  January "IRA top-up" explanation for seasonal anomalies; argues that IRA contribution
  deadlines (originally coinciding with April 15 before the deadline was extended)
  create a mechanical demand shock around the filing deadline date.

## Replications and related work

- **Ritter, J. R. (1988).** *The Buying and Selling Behavior of Individual Investors at
  the Turn of the Year.* Journal of Finance, 43(3), 701–717. Retail investors tend to
  sell in December (tax-loss harvesting) and reinvest in January; the analogous April
  mechanism (contribution-deadline buying) is the direct follow-on.
- **Poterba, J. M., & Weisbenner, S. J. (2001).** *Capital Gains Tax Rules, Tax-Loss
  Trading, and Turn-of-the-Year Returns.* Journal of Finance, 56(1), 353–368. Capital
  gains and tax realisation drive seasonal returns; the April deadline is a secondary
  pressure point for late refund-related purchases.
- **Haugen, R. A., & Jorion, P. (1996).** *The January Effect: Still There after All
  These Years.* Financial Analysts Journal, 52(1), 27–31. Finds that most US tax-related
  return anomalies have weakened or disappeared post-publication, a pattern consistent
  with our NONE verdict for the April window.
- **Kamstra, M. J., Kramer, L. A., & Levi, M. D. (2003).** *Winter Blues: A SAD Stock
  Market Cycle.* American Economic Review, 93(1), 324–343. Tests seasonal investor mood
  effects; spring (April) should be a reversal of winter blues if the mood mechanism is
  real. We find no evidence of a spring premium.

## The multiple-comparisons problem

- **Harvey, C. R., Liu, Y., & Zhu, H. (2016).** *...and the Cross-Section of Expected
  Returns.* Review of Financial Studies, 29(1), 5–68. With the large number of
  calendar anomalies tested in the literature, the hurdle for claiming significance
  should be well above the naive t=1.96. Our Bonferroni correction (k=2) is the
  minimum adjustment; in the broader literature the correction would be larger.
- **Bonferroni, C. E. (1936).** *Teoria statistica delle classi e calcolo delle
  probabilita.* Pubblicazioni del R Istituto Superiore di Scienze Economiche e
  Commerciali di Firenze. The correction applied here: two simultaneous hypotheses
  (pre-window, post-window), threshold 0.05/2 = 0.025.

## Why the effect, if real, would be weak

- **McLean, R. D., & Pontiff, J. (2016).** *Does Academic Publication Destroy Stock
  Return Predictability?* Journal of Finance, 71(1), 5–32. Anomalies decay after
  publication. An April deadline effect that existed in the 1980s would face
  arbitrage pressure as hedge funds and prop desks position in advance.
- **Fama, E. F., & French, K. R. (1988).** *Permanent and Temporary Components of
  Stock Prices.* Journal of Political Economy, 96(2), 246–273. Slow mean reversion
  and small predictable demand shocks produce very small and hard-to-exploit return
  differentials, well within the noise of a 10-day window.

## Method lineage (the desk's shared engine)

- **Newey, W., & West, K. (1987).** *A Simple, Positive Semi-Definite, Heteroskedasticity
  and Autocorrelation Consistent Covariance Matrix.* Econometrica, 55(3), 703–708.
  HAC t-stat used in `strategy._hac_tstat` for robust inference on each window group.
- **Welch, B. L. (1947).** *The Generalization of 'Student's' Problem when Several
  Different Population Variances are Involved.* Biometrika, 34(1/2), 28–35. The
  group-contrast t-test (`scipy.stats.ttest_ind(equal_var=False)`).

## Related desk studies

- **[Study 48 — Groundhog](../../48-groundhog/)**: weather-based seasonal forecast vs
  stock returns — another calendar anomaly with a strong folk narrative and a NONE verdict.
- **[Study 163 — Friday-13th](../../163-friday-13th/)**: daily calendar superstition
  tested on long GSPC history; same methodology, same NONE verdict.
- **[Study 136 — Mark-Twain](../../136-mark-twain/)**: "Sell in May" — the best-known
  monthly seasonal in US equities; methodological sibling to this study.
- **[Study 82 — Witching-Hour](../../82-witching-hour/)**: expiry-day / options
  deadline seasonal effects; a related deadline-driven demand shock.
