# References & literature map — Study 626 (Unemployment-Trend-Timing)

## The claim under test

- **The source.** *Growth-Trend Timing* — Jesse Livingston, **Philosophical Economics**
  (blog), *"Growth and Trend: A Simple, Long-Term Model that Offers Some Answers"*, February
  2016 (philosophicaleconomics.com). The rule: run a trend-following overlay (monthly
  price-vs-moving-average, i.e. Faber timing) **but only obey the sell signal when a macro
  "growth" gauge is deteriorating** — unemployment rising, real retail sales or industrial
  production contracting. The pitch: trend rules earn their keep only inside recessions;
  outside recessions their sell signals are whipsaws, so gating them on a recession
  indicator should keep the crash protection while dumping most of the false exits.
- **The trend base.** Mebane Faber (2007, rev. 2013), *A Quantitative Approach to Tactical
  Asset Allocation*, SSRN 962461 — the 10-month/200-day SMA rule this study filters; graded
  on this desk in [110-faber-timing](../../110-faber-timing/) (Signal Real as a drawdown
  shield, Tradability Fragile).
- **The macro trigger family.** Claudia Sahm (2019), *Direct Stimulus Payments to
  Individuals* (the "Sahm rule": 3-month-average unemployment 0.5pp above its 12-month low
  flags a recession onset) — graded in [268-sahm-rule](../../268-sahm-rule/) (a first-rate
  recession dater, a poor *standalone* sell button because stocks lead the cycle). This
  study asks the complementary question: unemployment not as a sell button but as a **veto**
  on someone else's sell button.

## Method

- **Signals & lags.** Month-end close vs its 200-day SMA; unemployment print (1-month
  **reporting lag** — the month-*m* rate is published in early *m+1*) vs its 12-month SMA.
  A signal formed at the close of month *m* sets the position for month *m+1* — exactly
  **one execution lag**. Newey & West (1987) HAC t on monthly active returns (positions
  persist, so active returns are serially correlated).
- **The exposure-matched placebo.** GTT holds equities ~15pp more of the time than Faber, so
  it mechanically collects the unconditional equity premium in those extra months whatever
  the filter knows. The honest null keeps the filter's *shape* (persistence + duty cycle)
  and destroys its *alignment*: every circular rotation of the `unemp_rising` series, full
  deterministic enumeration (904 offsets), p = share of rotations matching the observed
  GTT-minus-Faber mean. Fisher's randomization logic; cf. the random-timing control of
  study 110.
- **Whipsaw accounting.** A risk-off spell (as traded) is a *whipsaw* if its compounded
  cash-minus-equity P&L is negative, a *save* otherwise — the direct test of "halves the
  whipsaws".
- **Costs.** One-way cost × NAV per switch, 0/5/10/25 bps sweep (Frazzini, Israel &
  Moskowitz 2018, *Trading Costs*, on gross-vs-net discipline). Sharpe races are
  excess-vs-excess over the same T-bill series.

## Data sources

- **Unemployment.** BLS series **LNS14000000** (civilian unemployment rate U-3, 16+, SA,
  monthly, 1948-01→). Fetched once from the keyless BLS public API v2
  (`api.bls.gov/publicAPI/v2/timeseries/data/`, 10-year chunks); the identical numbers ship
  in the flat file `download.bls.gov/pub/time.series/ln/ln.data.1.AllData`. Cached at
  `_cache/unrate_lns14000000.csv`. **Current vintage** — revisions/seasonal-factor updates
  are not point-in-time (named on the Signal axis).
- **S&P 500.** ^GSPC daily closes via yfinance (1927-12→), price-only index; cached
  `_cache/gspc_daily.csv`. Total return built by adding the Shiller monthly dividend
  yield / 12 (Robert Shiller, *Irrational Exuberance* data, via the repo-staged parquet /
  the `datasets/s-and-p-500` mirror); cached `_cache/shiller_divyield_monthly.csv`.
- **Cash.** ^IRX (13-week T-bill discount rate) via yfinance from 1960
  (`_cache/irx_daily.csv`); 1948–1959 hardcoded **annual average 3-month T-bill (new issue)
  rates** from the *Economic Report of the President 2011*, Table B-73
  (govinfo.gov/content/pkg/ERP-2011/pdf/ERP-2011-table73.pdf), constant within each year.

## Related desk studies

- [110-faber-timing](../../110-faber-timing/) — the unfiltered base rule (Real / Fragile):
  a genuine drawdown shield that lags buy-and-hold on return. **This study's new claim is
  the FILTER**, not the trend rule.
- [268-sahm-rule](../../268-sahm-rule/) — unemployment as a *standalone* sell trigger
  (None / Mirage): recessions ≠ market timing when used alone. Here the same series is a
  veto on a price signal instead — a genuinely different construction, and the contrast
  between the two verdicts is the interesting part.
- [67-fed-drift](../../67-fed-drift/) and the macro-announcement family — other
  macro-series-meets-price-signal hybrids on the desk.
