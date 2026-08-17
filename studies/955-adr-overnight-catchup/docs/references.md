# References & literature map — Study 955 (ADR Catch-Up)

## The claim under test

- **The catch-up thesis.** A US-listed ADR is a claim on a share that trades somewhere
  else, on a different clock. Tokyo shuts at 02:00 New York time; London and Frankfurt at
  11:30. So when the ADR opens — or closes — its home market has already priced the day's
  news and gone home. The folk claim, repeated on every trading desk that touches ADRs, is
  that the ADR *owes* that move and pays it back in the US session: watch `^N225` plus the
  yen overnight and you know which way TM and SONY are going. This study asks two nested
  questions of the daily tape: is any of the home move still unpaid at the ADR's own close,
  and can the un-caught-up part be traded.
- **The steelman.** The mechanism is not a forecast, it is an accounting identity waiting
  to settle: the home line and the ADR are (up to the ratio and fees) the same asset in two
  currencies, so a home move that has not yet appeared in the ADR is arbitrage-visible.
  Where the two markets never overlap — Japan — the identity has a full session to close.
- **The honest limit.** Genuine catch-up is *intraday*. A daily-close tape sees only
  close-to-close, so it can measure the residue that survives to the ADR's own close, not
  the session in which the catching-up happens. Named on the Signal axis and repeated in
  `docs/results.md`.

## Why the effect should exist — the mechanism

- **Non-synchronous trading and stale prices.** Scholes & Williams (1977), *Estimating
  Betas from Nonsynchronous Data*, Journal of Financial Economics; Dimson (1979), *Risk
  Measurement When Shares Are Subject to Infrequent Trading*, JFE. The canonical result
  that an asset whose price is stamped at a different clock time loads on *lagged* market
  returns — exactly the `b1` this study estimates, and the reason a naive same-day beta
  understates the true one.
- **Lo & MacKinlay (1990), *When Are Contrarian Profits Due to Stock Market
  Overreaction?*, Review of Financial Studies.** Lead-lag structure across
  differently-traded assets, and the warning that it is easily confused with reversal.
- **Cross-listed price parity.** Gagnon & Karolyi (2010), *Multi-Market Trading and
  Arbitrage*, Journal of Financial Economics — ADR/home deviations are small on average
  (a few tens of basis points) but fat-tailed and mean-reverting; Froot & Dabora (1999),
  *How Are Stock Prices Affected by the Location of Trade?*, JFE — the twin-share evidence
  that location of trade matters more than the identity of the claim would allow.
- **Werner & Kleidon (1996), *U.K. and U.S. Trading of British Cross-Listed Stocks*,
  RFS.** Direct intraday evidence on the London/New York overlap: during the overlapping
  hours the two markets behave as one, and outside it they do not. This is the source of
  the region prediction the study tests — the UK names should have *no* residue, and they
  do not.
- **Overnight vs intraday return decomposition.** Lou, Polk & Skouras (2019), *A Tug of
  War: Overnight Versus Intraday Expected Returns*, JFE — the discipline of splitting a
  daily return by session, and the reason a close-to-close tape is a blunt instrument for
  any session-level claim.

## Why it can fail — and what would fool us

- **The arbitrage is small, fast and crowded.** ADR/home deviations are the most-watched
  spread in cross-border trading, closed by market-makers within the session. Anything
  visible at the daily close is what nobody wanted.
- **Short-horizon reversal wearing a costume.** Any residual `a_t − β·x_t` contains `a_t`,
  so a negative predictive slope may simply be the bid-ask bounce of Roll (1984), *A Simple
  Implicit Measure of the Effective Bid-Ask Spread*, Journal of Finance, plus the
  liquidity-provision reversal of Nagel (2012), *Evaporating Liquidity*, RFS. The study
  plants pure bounce in the synthetic control and shows it fires the residual rule with
  **zero** stale information — which is why the residual test never carries the verdict.
- **The multivariate coefficient is not the tradable one.** `b1` conditions on tomorrow's
  home move, which nobody knows at the trade. Because the home dollar move is negatively
  autocorrelated (ρ₁ = −0.165 for the Nikkei-in-dollars), `b1` overstates the univariate,
  bettable γ by about 2.4× here. Reporting only `b1` is the single easiest way to
  overstate this idea, and `strategy.tradable_gamma` exists to stop it.
- **A regression slope is an average, and averages have owners.** A handful of enormous
  overnight moves can carry a pooled coefficient on their own; the HAC *t* will not tell
  you, because it corrects the standard error for dependence, not for leverage. The
  classical influence diagnostics — Belsley, Kuh & Welsch (1980), *Regression Diagnostics*,
  Wiley; Huber (1964), *Robust Estimation of a Location Parameter*, Annals of Mathematical
  Statistics — say to re-estimate with the extremes trimmed and with them winsorized, and
  read the gap. `strategy.tail_sensitivity` does exactly that, calibrated on a synthetic
  panel where the planted lag *is* linear (there the same knife moves the coefficient by
  1%). On the real Japan slice it moves it by 79%, which is the single most important
  number in this study and the reason the Signal stamp is Weak rather than Mixed.
- **Turnover.** A daily-rebalanced spread turning over ~108% of NAV per session needs a
  gross edge of several basis points a day to survive a penny spread. Zakamulin (2014),
  *The Real-Life Performance of Market Timing…*, Journal of Asset Management, for the
  general lesson that headline rules die in the friction line.

## Related desk studies (dedup)

- **[Study 01 — Overnight Anomaly](../../01-overnight-anomaly/)** and
  **[Study 788 — Overnight/Intraday Tug-of-War](../../788-overnight-intraday-tug-of-war/)**
  decompose a *single* US tape into its overnight and intraday halves. Study 955 is about
  two *different markets* on two clocks: the predictor is a foreign index plus a currency,
  not the same instrument's own gap.
- **[Study 379 — ETF Lead-Lag](../../379-etf-lead-lag/)**,
  **[Study 865 — Credit → Equity Lead-Lag](../../865-credit-equity-lead-lag/)** and
  **[Study 870 — Industry-Leader Lead-Lag](../../870-industry-leader-lead-lag/)** all test
  a lead-lag *within* one trading session, where the lag is informational (who reacts
  first). Here the lag is *mechanical* — the leading market is physically shut — which is
  why the region cut by closing time is the study's discriminating test rather than a
  robustness afterthought.
- **[Study 613 — Currency-Hedged ETF Carry](../../613-currency-hedged-etf-carry/)** and
  **[Study 634 — US Leads the World](../../634-us-leads-the-world/)** use the same
  home-index/FX machinery but ask about a *carry identity* and a *global* directional lead
  respectively; neither touches the ADR-vs-home-line residue.
- **[Study 916 — Withholding Drag](../../916-withholding-drag-international/)** is the
  other study on this desk about international listings seen from a US tape, but its
  subject is a *tax* wedge in the dividend, not a *timing* wedge in the price.
- **[Study 376 — MOC Imbalance](../../376-moc-imbalance/)** is the nearest reversal study:
  it asks whether the closing push reverses overnight in a single market. Study 955's
  reversal control (fade the ADR's own move, look at no home data at all) is the same
  effect, used here as a *null benchmark* the home-informed books have to beat — and do not.

## Method lineage

- **HAC / Newey-West t-stat and HAC OLS.** Newey & West (1987), *A Simple, Positive
  Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix*,
  Econometrica — [`strategy.newey_west_t`](../adr_catchup/strategy.py),
  [`strategy.hac_ols`](../adr_catchup/strategy.py), and
  [`quantlab.analytics.mean_tstat_hac`](../../../quantlab/analytics.py).
- **Fama-MacBeth cross-sectional slopes.** Fama & MacBeth (1973), *Risk, Return, and
  Equilibrium: Empirical Tests*, Journal of Political Economy —
  [`strategy.fama_macbeth`](../adr_catchup/strategy.py). Used because a pooled *t* over
  41,000 rows from eight names that share three home markets is not eight thousand
  independent observations.
- **Circular block bootstrap.** Politis & Romano (1994), *The Stationary Bootstrap*, JASA —
  [`strategy.block_bootstrap_ci`](../adr_catchup/strategy.py),
  [`strategy.beta_lag_bootstrap`](../adr_catchup/strategy.py) (which resamples whole *dates*
  in blocks so a day's cross-section is never split) and
  [`quantlab.stats.sharpe_ci_bootstrap`](../../../quantlab/stats.py).
- **Trimmed and winsorized re-estimation.** [`strategy.tail_sensitivity`](../adr_catchup/strategy.py),
  with [`strategy.lag_bucket_table`](../adr_catchup/strategy.py) as its non-parametric twin
  (sort by yesterday's home move, read what the ADR paid next).
- **Reproducibility stamp.** [`quantlab/repro.py`](../../../quantlab/repro.py) — the as-of
  slice and the content fingerprint that make a rerun verifiable.

## Data sources, proxies and assumptions

- **ADRs** TM, SONY, SAP, NVO, SHEL, BP, HSBC, RIO — daily **total-return** closes via
  `yfinance` (`auto_adjust=True`). These are high-payout names; a price-only tape would
  misstate their returns by several points a year.
- **Home indices** `^N225`, `^GDAXI`, `^FTSE` — **price-only** (Yahoo publishes no
  total-return series for them). Harmless here: the home move is only ever an explanatory
  variable, and the missing dividend yield is a constant that lands in the intercept.
- **FX** `JPY=X` (quoted local-per-USD, therefore **inverted** to the dollar value of the
  yen), `EURUSD=X`, `GBPUSD=X`.
- **PROXY — the cash leg.** `^IRX`, the 13-week bill *discount* yield, converted to a daily
  accrual. It ignores the discount-to-bond-equivalent conversion and any fund expense.
  `BIL` (a real, costed T-bill ETF, from 2007) is the cross-check and moves the basket's
  excess Sharpe from +0.515 to +0.450.
- **PROXY — NVO's home market.** Novo Nordisk is Danish; no Copenhagen index reaches back
  to 2004 in the shared cache, so the headline panel pairs it with `^GDAXI`/`EURUSD=X` (the
  krone is pegged to the euro inside a ±2.25% band). The genuine `^OMXC25`/`DKK=X` pair is
  run as a cross-check from 2016-12 and gives the same null.
- **ASSUMPTION — costs.** 5 bps one-way × NAV on turnover, swept 0–25 bps. The breakeven
  cost is reported directly, so the reader can substitute their own.
- **ASSUMPTION — borrow.** 50 bps/yr on the short leg, accrued daily; swept 0–300 bps,
  which moves the answer by ~1.4 pp and changes nothing.
- **ASSUMPTION — the closing clock.** Home-market closes are taken as fixed New York hours
  (Tokyo 02:00, London and Frankfurt 11:30), ignoring the few weeks a year when the two
  hemispheres' daylight-saving transitions are out of step.
- **Survivorship.** Eight large ADRs still listed in 2026, hand-picked for a long tape and
  a liquid home line. They are the *survivors* of the cross-listed universe; the ADRs that
  delisted, converted or were taken over are absent. Named on the Signal axis.
- **As-of 2026-06-30.** The partial current month is dropped so the sample never creeps.
  The window opens 2004-01-05, where the `EURUSD=X` and `GBPUSD=X` tapes begin.
