# References & literature map — Study 160 (Skyscraper-Curse)

## The claim under test

- **The Skyscraper Index.** Lawrence, A. (1999). *The Skyscraper Index: Faulty Towers!*
  Property Report, Dresdner Kleinwort Wasserstein Research. The original piece coining
  the term. Lawrence observes that world-record building completions cluster near economic
  peaks and precede recessions. He is self-aware enough to call it a curiosity; later
  commentators hardened it into a prediction rule. The testable version: post-completion
  S&P 500 returns underperform the unconditional baseline.

## Academic treatments of the Skyscraper Index

- **Thornton, D.L. & Wheelock, D.C. (2014).** *Making Sense of Dissents: A History of
  FOMC Dissents.* Federal Reserve Bank of St. Louis Review. Cited in the context of
  credit-cycle indicators; the skyscraper observation is one of several "peak optimism"
  signals discussed in the macro literature.

- **Barr, J., Mizrach, B. & Mundra, K. (2015).** *Skyscraper Height and the Business
  Cycle: Separating Myth from Reality.* Applied Economics, 47(2), 148-160. The most
  rigorous academic treatment. Extends Lawrence's event table to a broader set of
  construction projects; finds the correlation between record-height completions and
  subsequent economic downturns weakens substantially once the sample is expanded beyond
  the cherry-picked "world record" events. Concludes: the narrative is real (credit booms
  drive both construction and economic excesses) but the index has no reliable predictive
  power for specific turning points.

- **Helsley, R.W. & Strange, W.C. (2008).** *A Game-Theoretic Analysis of Skyscrapers.*
  Journal of Urban Economics, 64(1), 49-64. Provides the theoretical mechanism: a
  "winner's curse" in skyscraper competition among developers, which amplifies
  construction during boom periods. This explains the correlation without implying
  predictive power.

## The general problem: small-n spurious signals

- **Meehl, P.E. (1978).** *Theoretical Risks and Tabular Asterisks: Sir Karl, Sir Ronald,
  and the Slow Progress of Soft Psychology.* Journal of Consulting and Clinical Psychology,
  46(4), 806-834. The classic treatment of how small samples and narrative selection
  produce spuriously compelling patterns.

- **Fama, E.F. (1991).** *Efficient Capital Markets: II.* Journal of Finance, 46(5),
  1575-1617. Covers the general problem of over-fitting to historical return anomalies;
  argues that short samples and data-snooping generate false positives at high rates.

- **Harvey, C.R., Liu, Y. & Zhu, H. (2016).** *... and the Cross-Section of Expected
  Returns.* Review of Financial Studies, 29(1), 5-68. Proposes t-stat thresholds of
  3.0+ for newly discovered anomalies to account for multiple testing; the Skyscraper
  Index, even with a "big" t on n=6, cannot come close to this bar.

## The null case: why post-event S&P returns are not special

- **Shiller, R.J. (2015).** *Irrational Exuberance.* 3rd ed. Princeton University Press.
  Provides the long-run S&P 500 data (monthly, 1871-present) used here. Shiller's CAPE
  is a genuine long-horizon market predictor; the Skyscraper Index is not.

- **Schwert, G.W. (2003).** *Anomalies and Market Efficiency.* Handbook of the Economics
  of Finance, 1, 939-974. Documents how many apparent equity market anomalies disappear
  out-of-sample or after transaction costs.

## Method lineage

- **HAC / Newey-West t-stat.** Newey, W.K. & West, K.D. (1987). *A Simple, Positive
  Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix.*
  Econometrica, 55(3), 703-708. Used in `strategy.summarize`.

- **Event study methodology.** MacKinlay, A.C. (1997). *Event Studies in Economics and
  Finance.* Journal of Economic Literature, 35(1), 13-39. The forward-return event-window
  design follows standard event-study practice; the key departure is the explicit power
  analysis, which standard event studies on large samples do not need.

## Data sources used here

- **Shiller S&P 500 monthly dataset** (pre-staged at `_cache/shiller_sp500.parquet`).
  Source: Robert Shiller, http://www.econ.yale.edu/~shiller/data.htm. Monthly 1871-2026,
  includes SP500, Dividend, Earnings, CPI, CAPE (PE10). No network call at run time.

- **CTBUH Tall Buildings Database** (Council on Tall Buildings and Urban Habitat),
  ctbuh.org, accessed 2026-06. Used to verify building heights and completion years.

- **Wikipedia: 'List of tallest buildings'**, accessed 2026-06. Cross-reference for
  building names, heights, and completion years.

## Related desk studies

- **[Study 81 — Four-Year-Itch](../../81-four-year-itch/)**: US presidential election
  cycle and stock returns — another small-n, compelling-narrative claim tested honestly.
- **[Study 83 — Half-Life](../../83-half-life/)**: Bitcoin halving effect — n=4 events,
  same structural power problem, same verdict.
- **[Study 48 — Groundhog](../../48-groundhog/)** and
  **[Study 80 — Cold-Open](../../80-cold-open/)**: calendar-based seasonal claims;
  the desk's other fun-claims tier.
