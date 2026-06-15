# References & literature map — Study 163 (Friday-13th)

## The claim under test

- **Kolb, R. W., & Rodriguez, R. J. (1987).** *Friday the Thirteenth: 'Part VII' — A Note.*
  Journal of Finance, 42(5), 1385–1387. The original academic finding: a statistically
  significant negative return on Friday the 13th in the Dow Jones Industrial Average
  from 1940 to 1987. The paper that launched a cottage industry of replications — and
  spawned this very study.
- **Lucey, B. M. (2001).** *Friday the 13th and the Philosophical Basis of Financial
  Economics.* Applied Economics Letters, 8(9), 565–568. A conceptual and empirical
  follow-up questioning both the robustness of the original result and its theoretical
  coherence: should a superstition ever produce a systematic return?

## Replications and disputes

- **Coutts, J. A. (1999).** *Friday the Thirteenth and the Financial Times 30 Share Index
  1935–94.* Applied Economics Letters, 6(1), 35–37. Found **no effect** on the FT30
  over 60 years, despite 103 Friday-13ths in sample.
- **Dyl, E. A., & Maberly, E. D. (1988).** *A Possible Explanation of the Weekend Effect.*
  Financial Analysts Journal, 44(3), 83–84. Contextualised Friday-specific anomalies
  within the broader Monday/weekend-effect literature — the "Friday" part of a Friday-13th
  effect can masquerade as the known weekend effect.
- **Patel, M. A. (2018).** Systematic review of calendar anomaly literature; concludes that
  most intraweek and intra-month effects published before 2000 either fail out-of-sample or
  are absorbed by risk premia once properly benchmarked.

## The multiple-comparisons problem — the statistical lesson

- **Harvey, C. R., Liu, Y., & Zhu, H. (2016).** *...and the Cross-Section of Expected
  Returns.* Review of Financial Studies, 29(1), 5–68. Documents t-stat inflation from
  mining calendar anomalies; argues that with the number of factors published in the
  literature, the hurdle for significance should be far above t=2.
- **Bonferroni, C. E. (1936).** *Teoria statistica delle classi e calcolo delle
  probabilità.* Pubblicazioni del R Istituto Superiore di Scienze Economiche e
  Commerciali di Firenze. The correction applied here: with k=4 "special Friday" slots
  tested simultaneously (6, 13, 20, 27), the alpha per comparison is 0.05/4 = 0.0125.

## Why superstitions don't translate to price

- **Fama, E. F. (1970).** *Efficient Capital Markets: A Review of Theory and Empirical
  Work.* Journal of Finance, 25(2), 383–417. For a superstition to create a systematic
  return, either (a) it must change cash flows or discount rates in a predictable way,
  or (b) investors must systematically act on it en masse. The first channel is absent;
  the second would be self-correcting once arbitrageurs exploited the mis-pricing.
- **Shleifer, A., & Vishny, R. W. (1997).** *The Limits of Arbitrage.* Journal of
  Finance, 52(1), 35–55. Even if superstition created mispricing, the limits of
  arbitrage explain why small anomalies can persist — but also why they show up as
  noise-level effects, not significant signals.

## The small-n reckoning

- **McLean, R. D., & Pontiff, J. (2016).** *Does Academic Publication Destroy Stock Return
  Predictability?* Journal of Finance, 71(1), 5–32. Documents post-publication decay in
  anomalies. With ~1.7 events per year, a Friday-13th study accumulates power slowly:
  the 168 events in our 99-year sample still leave the test with low power against any
  effect smaller than ~15 bps/day.

## Method lineage (the desk's shared engine)

- **Newey, W., & West, K. (1987).** *A Simple, Positive Semi-Definite, Heteroskedasticity
  and Autocorrelation Consistent Covariance Matrix.* Econometrica, 55(3), 703–708.
  HAC t-stat used in `strategy._hac_tstat` for robust inference on each day-group.
- **Welch, B. L. (1947).** *The Generalization of 'Student's' Problem when Several
  Different Population Variances are Involved.* Biometrika, 34(1/2), 28–35. The
  group-contrast t-test (`scipy.stats.ttest_ind(equal_var=False)`).

## Related desk studies

- **[Study 80 — Cold-Open](../../80-cold-open/)**: January Barometer — another calendar
  anomaly with ~75 observations; same small-n reckoning and same NONE verdict.
- **[Study 48 — Groundhog](../../48-groundhog/)**: weather-based seasonal forecast vs
  stock returns — the superstition cousin of this study.
- **[Study 136 — Mark-Twain](../../136-mark-twain/)**: "Sell in May" — another monthly
  calendar anomaly tested on long GSPC history.
