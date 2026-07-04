# References — Study 596 (Bond Tent / Rising Equity Glidepath)

## The claim's source

- **Pfau, W. & Kitces, M. (2014).** *Reducing Retirement Risk with a Rising Equity Glide Path.*
  Journal of Financial Planning, 27(1), 38–45. SSRN: <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2324930>.
  The origin of the tested claim: start retirement bond-heavy (~30% equity) and glide equity
  *up* through retirement; in their Monte Carlo simulations this raises success rates and
  softens failures relative to static and declining paths — most visibly under low-return
  assumptions.
- **Kitces, M. (2016).** *The Portfolio Size Effect And Using A Bond Tent To Navigate The
  Retirement Danger Zone.* Nerd's Eye View. <https://www.kitces.com/blog/managing-portfolio-size-effect-with-bond-tent-in-retirement-red-zone/>.
  Coins the "bond tent" name: equity glides down into retirement and back up after it; this
  study tests the retirement (back) half of the tent, which is the actionable part of the claim.

## The adversarial literature

- **Estrada, J. (2016).** *The Retirement Glidepath: An International Perspective.* Journal of
  Investing, 25(2). SSRN: <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2557124>.
  On historical (not Monte Carlo) data across 19 countries, rising glidepaths are broadly
  *inferior* to static and declining ones — the finding our US tape reproduces.
- **Bengen, W. (1994).** *Determining Withdrawal Rates Using Historical Data.* Journal of
  Financial Planning, 7(4), 171–180. The 4%-rule framework (30-year cohorts, real withdrawals,
  annual rebalance) our simulator follows.
- **Blanchett, D. (2007).** *Dynamic Allocation Strategies for Distribution Portfolios:
  Determining the Optimal Distribution Glide Path.* Journal of Financial Planning, 20(12).
  An earlier historical-cohort comparison of decumulation glidepaths: static paths hold up
  against declining ones.

## Sibling studies on this desk (the dedup guard)

- [Study 172 — Hundred-Minus-Age](../../172-hundred-minus-age/): the *accumulation* glidepath
  ("age in bonds"), including a Pfau-Kitces rising variant during the saving phase. This study
  is its retirement-phase cousin: the claim tested here is the **glidepath shape in
  decumulation** — the bond tent proper — which 172 did not touch.
- [Study 173 — Four-Percent-Rule](../../173-four-percent-rule/): the 4% rule on the same
  Shiller tape (nominal-bond variant, annual cohorts). We inherit its cohort framework and ask
  the *next* question: given the 4% rule, does the tent's shape add safety? (Our SAFEMAX runs
  below 173's because our bond leg is CPI-deflated 10-year and our cohorts are monthly-start —
  a stated data decision, not a contradiction.)

## Data

- **Shiller, R.** *Irrational Exuberance* long-run US dataset (S&P composite price, dividends,
  CPI, 10-year yield, CAPE), monthly 1871+. Homepage: <http://www.econ.yale.edu/~shiller/data.htm>.
  Fetched via the GitHub raw mirror <https://raw.githubusercontent.com/datasets/s-and-p-500/main/data/data.csv>,
  cached at `_cache/shiller_sp500.parquet` (cache-first; same extract staged repo-wide).
- Bond returns: first-order 10-year approximation `y_{t-1}/12 − D·Δy` with modified duration
  D = 7, deflated by realised CPI — standard in long-run asset-allocation research (e.g.
  Campbell-Viceira). Stated as a decision in [results.md](results.md).

## Method

- **Newey, W. & West, K. (1987).** *A Simple, Positive Semi-Definite, Heteroskedasticity and
  Autocorrelation Consistent Covariance Matrix.* Econometrica 55(3) — HAC t on overlapping
  cohort differences, bandwidth forced to the full 360-month overlap.
- **Politis, D. & Romano, J. (1992).** Circular block bootstrap — 120-month blocks on the joint
  monthly tape for distribution-statistic CIs.
