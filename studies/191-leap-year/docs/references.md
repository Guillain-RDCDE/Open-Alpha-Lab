# References — Study 191 (Leap-Year)

## Primary sources

**Shiller, R.J. (2005).**  *Irrational Exuberance*, 2nd edition.  Princeton University
Press.  Data appendix available at http://www.econ.yale.edu/~shiller/data.htm.  The
monthly S&P 500 price index series (back to 1871) is the primary data source for this
study.  The repo-staged cache `_cache/shiller_sp500.parquet` is derived from this
dataset.

**Hirsch, Y. and Hirsch, J.A. (2021).**  *Stock Trader's Almanac 2022*, 55th annual
edition.  Wiley.  The Almanac popularises the presidential-cycle ("four-year itch")
pattern and the election-year seasonal, which overlap exactly with the leap-year claim
in the modern era.  The Almanac is the source of the folk belief that "election years
are good for stocks."

**Huang, R.D. (1985).**  "Common stock returns and presidential elections."
*Financial Analysts Journal* 41(5), 58–65.  Early quantitative work on the
presidential-cycle effect; finds no robust election-year premium once unconditional
returns are controlled for.

**Stovall, S. (2012).**  "The four-year presidential election cycle and the US stock
market."  S&P Capital IQ research note.  Documents the year-of-term return differential,
with year 3 (pre-election) often cited as the strongest — inconsistent with a simple
election-year / leap-year story.

## Context and methodology

**Ioannidis, J.P.A. (2005).**  "Why most published research findings are false."
*PLOS Medicine* 2(8), e124.  The multiple-comparisons failure mode at the heart of
calendar-anomaly mining: testing many quadrennial hypotheses (leap year, election year,
Olympics year, Olympic-cycle year) inflates false-discovery risk above any single test's
nominal alpha.

**Sullivan, R., Timmermann, A., and White, H. (2001).**  "Dangers of data mining:
the case of calendar effects in stock returns."  *Journal of Econometrics* 105(1),
249–286.  Applies White's (2000) Reality Check to a wide range of calendar rules and
finds that most disappear when proper multiple-comparisons corrections are applied;
directly relevant to the leap-year claim.

**Newey, W.K. and West, K.D. (1987).**  "A simple, positive semi-definite,
heteroskedasticity and autocorrelation consistent covariance matrix."
*Econometrica* 55(3), 703–708.  The HAC estimator used for the per-group t-statistics.
Annual return series have low autocorrelation but potential heteroskedasticity (the
variance shifts across macro regimes), so Newey-West inference is appropriate.

## Related Open-Alpha-Lab studies

- **Study 81 — Four-Year-Itch:** the presidential-cycle year-of-term effect (the direct
  confound of the leap-year claim); tests whether year-of-term 1–4 generates
  differentiated equity returns.
- **Study 48 — Groundhog:** another folklore-based calendar anomaly (Punxsutawney Phil
  predicts six more weeks of winter → bad for stocks); same NONE verdict from the same
  Shiller tape.
- **Study 163 — Friday-13th:** superstition-driven daily calendar anomaly; also NONE.
- **Study 136 — Mark-Twain:** the "October Effect" monthly seasonal; same family of
  claims, same statistical framework.
