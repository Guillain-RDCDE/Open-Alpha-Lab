# References & literature map -- Study 360 (NAAIM-Exposure)

## The claim under test

- **The folklore.** The NAAIM Exposure Index is widely cited (financial press,
  technical-analysis blogs, NAAIM's own weekly commentary) as a *contrarian
  sentiment gauge*: when active managers report being all-in (extreme high exposure)
  the upside is supposedly exhausted -> sell; when they have fled to cash (extreme
  low exposure) capitulation is near -> buy. The hook: this is the *smart money*
  positioning, not a retail opinion poll, so fading it should be more reliable.
- **The testable version.** (H1) NAAIM extremes carry forward information about the
  next week's SPY return; (H2) the contrarian timing rule beats buy-and-hold net of
  costs; (H3) because it is *professional* positioning, it beats the equivalent
  retail-sentiment gauge.

## The data source (primary, free)

- **NAAIM Exposure Index.** National Association of Active Investment Managers,
  *NAAIM Exposure Index* -- a weekly survey (since 2006-07-05) of member firms'
  reported **current equity exposure**, mean across respondents, on a 0-200% scale
  (0 = all cash, 100 = fully invested, 200 = 2x leveraged long; the mean can dip
  slightly negative when members are net short). NAAIM publishes the **full weekly
  history free** as a since-inception spreadsheet:
  `naaim.org/programs/naaim-exposure-index/`
  (`USE_Data-since-Inception_<date>.xlsx`, column "NAAIM Number"). This study uses
  the genuine published series (cached under `_cache/naaim_weekly.csv`), with a
  compact **real** quarterly fallback (`data.NAAIM_FALLBACK`) so the offline core
  runs if the cache is gone.
- **SPY total return.** SPDR S&P 500 ETF daily closes from yfinance
  (`auto_adjust=True`, dividends reinvested), resampled to the NAAIM survey dates.
  Total return on both legs so a cash-sitting timing rule is honestly charged its
  forgone dividends.

## Why a contrarian sentiment edge is usually weak / priced out

- **Sentiment as a (weak) contrarian predictor.** Fisher & Statman (2000), *Investor
  Sentiment and Stock Returns*, *Financial Analysts Journal* -- individual- and
  newsletter-sentiment relate negatively to subsequent returns, but the relationship
  is weak and unstable out of sample. Brown & Cliff (2004), *Investor Sentiment and
  the Near-Term Stock Market*, *J. Empirical Finance* -- sentiment co-moves with the
  market but has little short-horizon predictive power.
- **The cross-section of sentiment.** Baker & Wurgler (2006), *Investor Sentiment and
  the Cross-Section of Stock Returns*, *J. Finance* -- sentiment effects are real but
  concentrated in hard-to-arbitrage stocks and at long horizons, not in a weekly
  index-timing signal.
- **Managers chase, not lead.** A large literature (e.g. Frazzini & Lamont 2008,
  *Dumb Money*; flow-performance studies) finds active-manager and fund-flow exposure
  is largely *trend-following* -- positioning rises after the market has already
  risen -- which makes "fade the all-in pros" mechanically equivalent to "fade
  recent strength," a weak and pro-cyclical bet.
- **Market efficiency on a public weekly print.** A freely-published weekly gauge that
  thousands watch is, by construction, unlikely to hide a large, stable, tradable
  edge: any robust contrarian alpha would be arbitraged toward the unconditional
  drift, which is what we observe (R-squared ~0.03%).

## Method lineage (the desk's shared engine)

- **Newey-West (HAC) inference.** Newey & West (1987), *A Simple, Positive
  Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance
  Matrix*. The Signal axis is the HAC *t* of the regime long-short and of the
  predictive-regression slope; weekly returns are mildly autocorrelated, so HAC (not
  OLS) standard errors are the honest test. `REAL` needs HAC *t* >= 2 on the real
  tape (METHODOLOGY -> *The inference bar*).
- **Predictive regression / R-squared bar.** Forward return on standardised prior
  exposure; a real contrarian effect needs a negative slope with `|t| >= 2` and a
  non-trivial R-squared (cf. Welch & Goyal 2008, *A Comprehensive Look at the
  Empirical Performance of Equity Premium Prediction* -- most predictors fail out of
  sample with tiny R-squared).
- **Deterministic synthetic control.** A fixed-seed AR(1) exposure tape with a known
  contrarian loading (`data.synthetic_weekly`); the harness must recover the planted
  edge (slope *t* << -2) and read ~zero on the null (*t* ~ 0). A machinery proof,
  never market evidence (METHODOLOGY -> *The inference bar*).

## Related desk studies (the same contrarian template, different crowd)

- **[Study 257 -- AAII-Sentiment](../../257-aaii-sentiment/)** -- the
  *individual-investor* bull-bear survey as a contrarian tool. Same verdict shape
  (WEAK / MIRAGE); this study is the **professional-positioning** twin and the
  head-to-head shows the pros are no more bankable.
- **[Study 261 -- Put-Call Ratio](../../261-put-call-ratio/)** -- the *options-crowd*
  hedging gauge as a contrarian tell.
- **[Study 260 -- Margin-Debt](../../260-margin-debt/)** -- aggregate *leverage* as a
  euphoria/risk gauge. NAAIM is the cleaner real-time positioning read (weekly vs
  margin debt's lagged monthly print), which is exactly why we test it separately.

## Data sources used here

- **NAAIM** since-inception weekly spreadsheet (`naaim.org`), column "NAAIM Number",
  cached under `_cache/`.
- **yfinance** SPY total-return daily closes. All headline numbers are pinned in
  [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).
