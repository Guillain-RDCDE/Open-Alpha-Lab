# References & literature map -- Study 223 (Same-Month Seasonality)

## The canonical claim

- **Heston, S. L. & Sadka, R. (2008).** *Seasonality in the Cross-Section of
  Stock Returns.* Journal of Financial Economics, 87(2), 418--445.
  The founding paper: stocks sorted by their historical return in the same
  calendar month outperform low-seasonality stocks by ~40 bps per month in
  their most liquid specification (NYSE 1952-2002).  The effect is distinct
  from momentum, long-term reversal, and January seasonality.  The authors
  attribute it to periodic liquidity or investor-attention shocks that recur
  annually on a stock-specific basis.

- **Heston, S. L. & Sadka, R. (2010).** *Seasonality in the Cross-Section of
  Stock Returns: The International Evidence.* Journal of Financial and
  Quantitative Analysis, 45(5), 1133--1160.
  Extends the 2008 result to international equity markets; confirms the effect
  in 18 out of 20 countries.  Suggests that country-specific earnings
  announcement patterns and fiscal-year structures amplify the effect outside
  the US.

## Mechanisms and explanations

- **Keloharju, M., Linnainmaa, J. T. & Nyberg, P. (2016).** *Return Seasonalities.*
  Journal of Finance, 71(4), 1557--1590.
  A broad examination of return seasonalities across assets and countries.
  Finds strong same-month seasonality in equities, commodities, and currencies.
  The estimated premium shrinks toward recent decades, consistent with
  post-publication arbitrage.

- **Hartzmark, S. M. & Solomon, D. H. (2013).** *The Dividend Month Premium.*
  Journal of Financial Economics, 109(3), 640--660.
  Documents that stocks earn abnormal returns in the calendar months when they
  pay dividends -- a specific mechanism that could drive same-month seasonality
  for dividend-paying firms.  Dividend-month months recur in the same calendar
  months each year by design.

- **Jegadeesh, N. & Titman, S. (1993).** *Returns to Buying Winners and Selling
  Losers: Implications for Stock Market Efficiency.* Journal of Finance, 48(1),
  65--91.
  The momentum baseline.  Same-month seasonality is distinct from 12-month
  momentum: Heston-Sadka control for momentum explicitly and show their
  seasonality effect is incremental.

## Post-publication decay

- **McLean, R. D. & Pontiff, J. (2016).** *Does Academic Research Destroy Stock
  Return Predictability?* Journal of Finance, 71(1), 5--32.
  Documents that anomaly returns decay on average ~58% after academic
  publication.  The Heston-Sadka same-month seasonality effect (published 2008)
  is expected to have weakened since; our sub-period breakdown confirms the
  post-2018 spread (t = 2.15 on a survivorship-biased panel) is materially
  lower than the 1999-2008 era (t = 3.92).

## Related desk studies

- **[Study 24 -- Stampede](../../24-stampede/)**: 12-1 month cross-sectional
  momentum.  Same-month seasonality and momentum both exploit one-year look-back
  windows but target different phenomena: momentum uses the prior 12 months'
  cumulative return (all calendar months), same-month seasonality uses only
  the identical calendar-month return across years.

- **[Study 196 -- Long-Term-Reversal](../../196-long-term-reversal/)**: De Bondt-
  Thaler 36-60 month reversal.  Same-month seasonality and long-term reversal
  both involve looking back more than one year but are distinct signals.

- **[Study 158 -- Super-Bowl](../../158-super-bowl/)** and
  **[Study 174 -- Four-Percent-Rule](../../174-four-percent-rule/)**: other
  calendar-driven cross-sectional or market-level effects in the Calendar family.

## Method lineage

- **HAC / Newey-West t-stat.** Newey & West (1987), *A Simple, Positive
  Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance
  Matrix* (Econometrica).
- **Block-bootstrap Sharpe CI.** Politis & Romano (1994), *The Stationary
  Bootstrap* (JASA).
- **Survivorship notation.** Shumway (1997), *The Delisting Bias in CRSP
  Data* (Journal of Finance).
