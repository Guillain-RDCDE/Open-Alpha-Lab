# References & literature map — Study 248 (Presidential-Honeymoon)

## The claim under test

- **The first-100-days market honeymoon.** The folklore: markets enjoy a brief rally when a new
  president takes office — a honeymoon of investor optimism, policy relief, or partisan
  enthusiasm before the hard work of governing begins.  The claim circulates widely in financial
  media before each inauguration and is sometimes framed as a complement to the four-year
  presidential cycle (study 81): if year 3 is the boom year, do the first 100 days of year 1
  offer a secondary seasonal lift?  We steelman it as: *the S&P 500's cumulative return in the
  first 100 calendar days after each inauguration is meaningfully higher than its return in the
  next 265 calendar days (a within-term comparison), measured across all 25 inaugurations
  since Hoover in 1929.*

## The original and academic claims

- **Hirsch, Y. (annual, since 1972).** *Stock Trader's Almanac.* The presidential-cycle
  framework; the first-100-days observation is a popular derivative of the almanac tradition.
- **Nordhaus, W. D. (1975).** *The Political Business Cycle.* Review of Economic Studies,
  42(2), 169–190.  The canonical political-economy mechanism: administrations stimulate the
  economy before elections.  The honeymoon hypothesis is the *opposite* direction — a natural
  lift at the start of a term rather than at the end.
- **Huang, R. D. (1985).** *Common Stock Returns and Presidential Elections.* Financial
  Analysts Journal, 41(2), 58–61.  Documents year-of-term patterns; does not specifically
  isolate the first 100 days.
- **Hensel, C. R. & Ziemba, W. T. (1995).** *United States Investment Returns During
  Democratic and Republican Administrations, 1928–1993.* Financial Analysts Journal, 51(2),
  61–69.  Comprehensive review of presidential equity effects; no first-100-day premium is
  isolated.
- **Santa-Clara, P. & Valkanov, R. (2003).** *The Presidential Puzzle: Political Cycles and
  the Stock Market.* Journal of Finance, 58(5), 1841–1872.  The most-cited academic study of
  presidential equity effects; focuses on party (Democrat vs Republican) rather than
  inauguration timing.  Finds a robust party premium in raw data that fades after risk
  adjustment.

## Why the claim has intuitive appeal but thin empirical footing

- **Baker, M. & Wurgler, J. (2006).** *Investor Sentiment and the Cross-Section of Stock
  Returns.* Journal of Finance, 61(4), 1645–1680.  Provides the sentiment mechanism: new
  administrations generate optimism among certain investors, temporarily lifting prices.
  Sentiment-driven moves are typically mean-reverting — consistent with the honeymoon
  period underperforming the subsequent window in our data.
- **Shiller, R. J. (2000).** *Irrational Exuberance.* Princeton University Press.  Argues
  that short-term equity moves are driven by narrative and sentiment, not fundamentals — the
  psychological underpinning of any 'honeymoon' story.
- **Brunnermeier, M. K. & Nagel, S. (2004).** *Hedge Funds and the Technology Bubble.*
  Journal of Finance, 59(5), 2013–2040.  Illustrates how even sophisticated investors fail
  to arbitrage away sentiment-driven price moves over horizons of weeks to months — which
  would be the mechanism keeping a honeymoon premium alive.

## The honest critique

- **Sullivan, R., Timmermann, A. & White, H. (1999).** *Data-Snooping, Technical Trading
  Rule Performance, and the Bootstrap.* Journal of Finance, 54(5), 1647–1691.  A pattern
  tested on 25 non-overlapping 100-day windows on one index is almost certainly data-mined
  from the broader universe of calendar effects tested over the same history.
- **Harvey, C. R., Liu, Y. & Zhu, H. (2016).** *… and the Cross-Section of Expected Returns.*
  Review of Financial Studies, 29(1), 5–68.  Argues that the multiple-testing crisis in
  empirical finance demands a t-bar well above 2; for first-publication claims the bar
  should be closer to 3.0.  Our result (t = −0.29) does not come close to any bar.
- **Ferson, W. E. & Harvey, C. R. (1993).** *The Risk and Predictability of International
  Equity Returns.* Review of Financial Studies, 6(3), 527–566.  Documents how calendar
  effects in equities frequently fail out-of-sample; the post-2001 *reversal* in our data
  (honeymoon t = −3.83) is a real-world example of exactly this.

## Method lineage (the desk's shared engine)

- **HAC / Newey-West t-stat.** Newey & West (1987), *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix.* Econometrica — the
  inference method used in [`strategy._hac_tstat`](../presidential_honeymoon/strategy.py).
  Applied to the 25 paired differences (honeymoon minus control per term).
- **Within-term paired design.** Comparing the honeymoon window to the *subsequent* window
  in the *same* term controls for trend and cycle effects — the fair test rather than
  comparing to the full-history mean.
- **Synthetic positive control.** The `data.synthetic_daily` generator plants a known
  honeymoon premium; the test suite confirms the engine detects it when present and reads
  near-zero when absent — so the real-tape "none" verdict is a statement about the data,
  not a defect in the machinery.

## Data sources used here

- **Yahoo! Finance daily bars** (via `yfinance`): ^GSPC price index back to 1928-01-03
  (~24,700 trading days, 25 complete inaugurations from Hoover through Trump's second term).
  No dividends in the price index — the within-term *differential* tests are unaffected.
  Cached under `_cache/`; fingerprinted in [`docs/results.md`](results.md).
- **Inauguration dates.** Hardcoded table in `data.INAUGURATIONS` drawn from the US National
  Archives record of presidential inaugurations (publicly available, zero ambiguity for all
  25 terms since 1929).

## Related desk studies

- **[Study 81 — Four-Year-Itch](../81-four-year-itch/)**: the presidential election cycle
  (year-3 premium) — the structural cousin of this study; also Weak/Mirage and structurally
  limited by ~24 observations.
- **[Study 159 — Presidential-Party](../159-presidential-party/)**: Democratic vs Republican
  presidency equity returns — the party axis of the presidential-effect family; similarly
  fragile statistically.
- **[Study 134 — FOMC-Cycle](../134-fomc-cycle/)**: policy-driven equity drift around
  FOMC meetings — a tighter political-economy identification with more observations.
