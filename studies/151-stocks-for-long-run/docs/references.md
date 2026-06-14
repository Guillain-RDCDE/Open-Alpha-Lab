# References & literature map — Study 151 (Stocks-For-Long-Run)

## The claim under test

- **The canonical source.** Siegel, J.J. (1994, 2002, 2014, 2022), *Stocks for the Long Run*
  (McGraw-Hill). The core empirical claim: over any 20-year period in U.S. history, equities
  have never delivered a negative real return and have never lost to bonds. Siegel uses a real
  total return framework back to 1802 (extending Cowles via various sources). Our study uses
  Shiller's independently-constructed dataset from 1871 as the reference, which gives 150+
  years of robust data with well-documented sources.

- **The data source.** Shiller, R.J. (2000+), *Irrational Exuberance* dataset (online at
  http://www.econ.yale.edu/~shiller/data.htm). Monthly S&P 500 real price, real dividend,
  long-term nominal interest rate, and CPI back to January 1871. Our bond proxy
  (Long Interest Rate / 100 / 12 − monthly CPI) is a simplified real bond return, not a
  total-return bond index; we name this limitation explicitly.

## The equity premium literature

- **The foundational puzzle.** Mehra, R. & Prescott, E.C. (1985), *The Equity Premium: A Puzzle*
  (Journal of Monetary Economics, 15(2), 145–161). The U.S. equity premium (~6%/yr above bonds
  over 1889–1978) is far too large to be explained by standard consumption-based asset pricing
  — "too large to be consistent with reasonable risk aversion." Our empirical mean of +6.66%/yr
  real (30-year horizon) sits directly in this literature.

- **Long-run international evidence.** Dimson, E., Marsh, P. & Staunton, M. (2002), *Triumph of
  the Optimists: 101 Years of Global Investment Returns* (Princeton University Press). Extends
  the equity premium to 21 countries, 1900–2000. The U.S. result holds broadly but the U.S.
  sits near the top of the distribution; some markets (Russia, Argentina, Germany) experienced
  near-total loss. This is the core survivorship-bias caveat for Siegel's claim.

- **The long-run evidence update.** Jordà, Ò., Knoll, K., Kuvshinov, D., Schularick, M. &
  Taylor, A.M. (2019), *The Rate of Return on Everything, 1870–2015* (Quarterly Journal of
  Economics, 134(3), 1225–1298). Broadens to 16 advanced economies. Average real equity return
  ~7%/yr globally, consistent with our U.S. estimate, but with substantial country-level variance.

## Horizon effects and worst-case windows

- **Holding-period risk reduction.** Siegel's own analysis (2022 edition, Chapter 2) shows the
  worst-case rolling N-year real equity return shrinking from large negatives at short horizons
  to near-zero at 20 years and positive at 30 years. Our study replicates this finding on the
  Shiller panel (1871–2023) and finds the same pattern with one bare exception at 20 years.

- **Overlapping windows and inference.** Ang, A. & Bekaert, G. (2007), *Stock Return
  Predictability: Is it There?* (Review of Financial Studies, 20(3), 651–707). Long-horizon
  regressions with overlapping windows suffer from small effective sample sizes and severely
  inflated t-statistics. We flag this: for 30-year windows the effective independent sample
  is ~5 observations (150 years / 30). Our HAC t-stats are directionally meaningful but the
  magnitude should not be taken at face value.

- **The lost decade risk.** Faber, M. (2007), *A Quantitative Approach to Tactical Asset
  Allocation* (Journal of Wealth Management). Documents that even 10-year horizons in U.S.
  history delivered negative real returns multiple times (2000s most recently). Our decade
  breakdown confirms: bonds beat equity in 4 of 15 decades.

## The survivorship-bias caveat

- **U.S. exceptionalism.** Brown, S.J., Goetzmann, W.N. & Ross, S.A. (1995), *Survival*
  (Journal of Finance, 50(3), 853–873). Markets that survive to be studied are not random
  samples; survivorship bias inflates measured returns and worst-case improvements. Siegel's
  U.S. 'never lost' claim is conditional on the U.S. market surviving every episode; Russia
  1917 or Germany 1923 did not survive in a way that allows a Siegel-style analysis.

## Related desk studies

- **[Study 56 — Tide-Table](../../56-tide-table/)**: CAPE as a long-run equity return
  predictor — the Shiller-CAPE signal that identifies (in advance) which 10-year windows
  are likely to be bad for equity.
- **[Study 120 — Excess-CAPE-Yield](../../120-excess-cape-yield/)**: the bond-adjusted
  CAPE yield (1/CAPE − real bond yield) as a predictor of the equity-bond excess return —
  directly the Siegel premium in signal form.
- **[Study 68 — All-Weather](../../68-all-weather/)**: the portfolio diversification answer
  to the long-horizon risk — a multi-asset allocation that reduces the worst-case without
  fully sacrificing the equity premium.
- **[Study 97 — Balancing-Act](../../97-balancing-act/)**: rebalancing premium and its
  role in long-run equity outperformance — the mechanical basis for some of the Siegel edge.

## Method lineage

- **HAC / Newey-West t-stat.** Newey, W.K. & West, K.D. (1987), *A Simple, Positive
  Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix*
  (Econometrica, 55(3), 703–708). Used in `strategy.worst_case_windows` for the per-window
  excess return series.
- **Real total return construction.** Standard methodology: real price pct_change + real
  dividend yield / 12, compounded monthly. See Shiller (2000), *Irrational Exuberance*,
  Appendix 2 for the data construction methodology.
