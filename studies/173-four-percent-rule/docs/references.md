# References & literature map — Study 173 (Four-Percent-Rule)

## The claim under test

- **Bengen, W.P. (1994).** *Determining Withdrawal Rates Using Historical Data.*
  Journal of Financial Planning, 7(4), 171–180.  The founding paper.  Using Ibbotson
  SBBI data 1926–1992 with a 50/50 stock-bond mix, Bengen showed that no historical
  30-year cohort failed at a 4.15% real withdrawal rate — coining what became the
  "4% rule."  The rule is *conservative*: Bengen himself noted that 75% stock allocations
  pushed the SAFEMAX toward 5%.

## Extensions and critiques

- **Cooley, Hubbard & Walz (1998).** *Retirement Savings: Choosing a Withdrawal Rate
  That Is Sustainable.* AAII Journal (the "Trinity Study").  Extended Bengen's analysis
  to varying time horizons and asset allocations, providing the first probability-of-success
  tables.  Confirmed the ~4% rule under common 60/40 and 75/25 portfolios for 30-year
  horizons.

- **Pfau, W.D. (2010).** *An International Perspective on Safe Withdrawal Rates from
  Retirement Savings: The Demise of the 4 Percent Rule?*  Journal of Financial Planning,
  23(12), 52–61.  Extended the analysis to 17 developed-country markets using Shiller-style
  real returns.  Found that 4% failed in many non-US markets, arguing the rule is specific
  to the exceptionally high US equity returns of the 20th century.

- **Pfau, W.D. (2011).** *Safe Savings Rates: A New Approach to Retirement Planning over
  the Lifecycle.*  Journal of Financial Planning, 24(5), 42–50.  Showed that CAPE (PE10)
  at retirement is a robust predictor of safe withdrawal rates, with Q4 CAPE values
  (expensive markets) associated with significantly lower forward SWRs.

- **Kitces, M.E. (2008).** *Resolving the Paradox — Is the Safe Withdrawal Rate Sometimes
  Too Safe?*  The Kitces Report.  Documented that sequence-of-returns risk, not long-run
  mean return, is the dominant driver of failure: a retiree who survives the first decade
  with portfolio intact almost always succeeds even at higher withdrawal rates.

## Sequence-of-returns risk

- **Milevsky, M.A. & Abaimova, A. (2006).** *Sequence-of-Returns Risk and the
  Sustainability of Retirement Income.*  Demonstrates mathematically why the order of
  returns (not just their mean) determines portfolio survival — a 30-year mean of 7%
  with the worst years first routinely fails at 4%, while the same mean with best years
  first leaves a large bequest.

## Why the forward outlook is fragile

- **Shiller, R.J. (2000).** *Irrational Exuberance.* Princeton University Press.
  PE10 (CAPE) is inversely related to subsequent 10-year real returns; high starting CAPE
  implies compressed future returns and therefore a lower safe withdrawal rate.

- **Bogle, J.C. (2012).** *The Clash of the Cultures.* Wiley.  Argued that
  forward US equity real returns would be 4–5% vs. the historical 6.5%+, mechanically
  reducing the SWR below 4%.

## Data source

- **Shiller, R.J.** Online data supplement to *Irrational Exuberance* (various editions),
  available at http://www.econ.yale.edu/~shiller/data.htm.  Monthly US stock prices,
  dividends, earnings, CPI, and long-term interest rates from 1871.  This is the primary
  data source for this study — staged at the repo's shared `_cache/shiller_sp500.parquet`.

## Related desk studies

- **[Study 68 — All-Weather](../../68-all-weather/)**: the Ray Dalio all-weather portfolio —
  a risk-parity allocation evaluated on the same Shiller real-return tape.
- **[Study 97 — Balancing-Act](../../97-balancing-act/)**: the mechanics of annual
  rebalancing — the same 60/40 discipline that underpins the 4% rule.
- **[Study 102 — Free-Rebalance](../../102-free-rebalance/)**: whether rebalancing
  frequency matters — monthly vs. annual vs. threshold — another angle on the same
  retirement construction.
- **[Study 144 — Permanent-Portfolio](../../144-permanent-portfolio/)**: the Browne
  permanent portfolio as an alternative low-vol allocation for retirement drawdown.
