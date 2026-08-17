# References & literature map — Study 913 (Tracking-Difference Persistence)

## The claim under test

- **The fund-picker's folklore.** Two ETFs track the same index, yet one quietly returns a
  few basis points more each year. The advice that follows — repeated on every index-fund
  forum and in most "how to choose an ETF" guides — is to look up last year's **tracking
  difference** and buy whoever won it, on the theory that good tracking is a skill and skills
  persist. The competing rule is duller: ignore realised tracking and just buy the lowest
  **expense ratio**, which is published in advance and free to read.
- **Why the distinction matters.** Both rules would agree if realised TD were mostly fee. If
  it is mostly *noise* — dividend-timing conventions, securities-lending revenue, sampling
  error, index-reconstitution execution — then rotating into last year's winner is chasing a
  random draw and paying a spread (and, in a taxable account, a realised capital gain) for it.
- **Terminology.** *Tracking difference* is the level of fund-minus-index return; *tracking
  error* is the volatility of that difference. This study is about the difference, and about
  whether its **rank** is sticky.

## Where the tracking difference actually comes from

- **Sharpe (1966/1991), *The Arithmetic of Active Management*, Financial Analysts Journal.**
  The zero-sum identity that makes cost the only reliably persistent term in a fund's
  relative return. The whole study is an empirical restatement of it at one-basis-point
  resolution.
- **Carhart (1997), *On Persistence in Mutual Fund Performance*, Journal of Finance.** The
  canonical finding that almost all apparent persistence in fund returns is explained by
  expenses and transaction costs rather than skill — the direct ancestor of this study's
  "is it a memory or a level?" permutation test.
- **Elton, Gruber & Busse (2004), *Are Investors Rational? Choices Among Index Funds*,
  Journal of Finance.** S&P 500 index funds are near-perfect substitutes and their return
  differences are almost entirely explained by expense ratios, yet investors persistently
  buy the dearer ones. The single closest paper to the question here.
- **Blume & Edelen (2004), *S&P 500 Indexers, Tracking Error, and Liquidity*, Journal of
  Portfolio Management.** Where the *non*-fee part of tracking difference comes from —
  reconstitution trading, sampling and the cost of exact replication.
- **Petajisto (2017), *Inefficiencies in the Pricing of Exchange-Traded Funds*, Financial
  Analysts Journal.** ETF market price can deviate from NAV, so a TD measured from closing
  prices carries a mark-to-market term that a NAV-measured TD does not. This is why the
  study includes three **NAV-priced** mutual funds and labels them everywhere.
- **Securities lending.** Vanguard and iShares return lending revenue to the fund, which can
  offset several basis points of fee; SPY, as a Unit Investment Trust, may neither lend nor
  reinvest dividends between distribution dates — a structural drag independent of its
  9.45 bp fee. Elton, Gruber, Comer & Li (2002), *Spiders: Where are the Bugs?*, Journal of
  Business, documents the SPY UIT dividend-drag directly.

## Why the persistence test can go wrong

- **The measurement floor.** A TD computed from adjusted closes inherits the vendor's
  dividend-timing convention and the closing print. Our real tape shows a 9.7–10.7 bp
  standard deviation of relative TD — wider than the fee spread inside the ETF trio, and
  wide enough to produce two ~40 bp "outlier" years that fully reverse. Any persistence test
  run without acknowledging this floor will read noise as skill or skill as noise.
- **Level vs memory.** Rank persistence between *consecutive* years is not evidence of
  year-to-year information if the same correlation holds between *any* two years. The
  year-label permutation used here (Fisher's randomisation logic, *The Design of
  Experiments*, 1935) separates the two, and finds a level.
- **Survivorship.** The panel is eight funds that still exist. Trackers that tracked badly
  enough to close are absent, which compresses measured dispersion and flatters the
  cheapest-fund rule. Named on the Signal axis, as the desk requires.
- **Hindsight in the fee sheet.** Expense ratios are read at build time and applied to the
  whole sample, so a "buy the cheapest" rule is choosing its fund with information the early
  years did not have (Fidelity cut FXAIX to 1.5 bp only in 2019). The literature usually
  sidesteps this by using contemporaneous prospectus fees; lacking a point-in-time fee
  series, this study prices the bias instead — `strategy.per_fund_gap_vs_leader` re-asks the
  same question with no fee sheet at all, and the gap halves (+10.64 → +5.21 bp/yr) without
  changing sign.
- **Taxes are not on the tape.** Elton, Gruber & Blake (2001) and the broader after-tax
  literature make the point this study handles with an explicit assumption grid: a switch
  between two near-identical funds in a taxable account realises the embedded gain, which is
  a one-off cost measured in *percent* against a benefit measured in *basis points a year*.

## Related desk studies (dedup)

- **[Study 613 — Currency-Hedged-ETF-Carry](../../613-currency-hedged-etf-carry/)**: the
  closest cousin — two funds holding the *same* stocks, one out-returning the other by a
  mechanical amount. There the gap is the **CIP rate differential** (hundreds of bp) and the
  verdict is Real/Investable; here the gap is the **expense ratio** (single-digit bp) and
  the question is whether *realised* tracking, not a rate identity, is forecastable.
- **[Study 378 — ETF-NAV-Premium](../../378-etf-nav-premium/)**: price-versus-NAV
  dislocation and its mean reversion — an intraday/settlement pricing effect, not the
  annual total-return gap measured here.
- **[Study 379 — ETF-Lead-Lag](../../379-etf-lead-lag/)**: whether one ETF's move predicts
  another's *return*; this study is about the residual *cost* gap between funds that move
  together by construction.
- **[Study 601 — Factor-ETF-Live-Test](../../601-factor-etf-live-test/)**: live factor
  wrappers versus their academic promise — a question about the *strategy* inside the
  wrapper. Study 913 holds the strategy constant (one index) and varies only the wrapper.
- **[Study 139 — AI-Powered-ETF](../../139-ai-powered-etf/)**: an actively managed fund
  against a cheap index fund — a fee gap of ~70 bp, an order of magnitude coarser than the
  1.5–20 bp ladder resolved here.
- **[Study 624 — Buffer-ETF-Cost](../../624-buffer-etf-cost/)** and
  **[Study 619 — BITO-Roll-Drag](../../619-bito-roll-drag/)**: structural drags inside a
  wrapper (option financing, futures roll). Both are tens-to-hundreds of bp effects; this
  study asks whether the *smallest* such drag, the plain expense ratio, is even measurable
  from a public tape.

## Method lineage

- **Spearman rank correlation.** Spearman (1904), *The Proof and Measurement of Association
  Between Two Things*, American Journal of Psychology — [`strategy.spearman`](../td_persist/strategy.py),
  implemented on ranks with no scipy dependency.
- **Permutation / randomisation null.** Fisher (1935), *The Design of Experiments* —
  [`strategy.persistence`](../td_persist/strategy.py) shuffles the order of the calendar
  years, preserving each year's cross-section and destroying only the time linkage.
- **HAC / Newey-West t-stat.** Newey & West (1987), *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix*, Econometrica —
  [`strategy.newey_west_t`](../td_persist/strategy.py) and
  [`quantlab.analytics.mean_tstat_hac`](../../../quantlab/analytics.py).
- **Circular block bootstrap.** Politis & Romano (1994), *The Stationary Bootstrap*, JASA —
  [`strategy.block_bootstrap_ci`](../td_persist/strategy.py), applied in blocks of two
  consecutive *years*.

## Data sources

- **SPY, IVV, VOO** (S&P 500 ETFs), **VFIAX, FXAIX, SWPPX** (S&P 500 index mutual funds,
  NAV-priced), **QQQ, QQQM** (Nasdaq-100 ETFs), **BIL** (1–3M T-bill, the cash leg) and
  **^GSPC** (a price-only index proxy, kept to demonstrate what it cannot measure) — daily
  **total-return** closes via `yfinance` (`auto_adjust=True`), through 2026-06-30.
- **SPLG was requested and is unavailable.** Yahoo! Finance serves a single stale bar
  (2026-07-17) for it and rejects every historical range. Rather than substitute a different
  fund under the same name, it is declared missing in `data.UNAVAILABLE` and named in the
  README and `docs/results.md`.
- **Expense ratios are issuer disclosure, not tape** — an ASSUMPTION *carrying hindsight*
  (build-time levels applied to the whole sample), listed in `data.EXPENSE_RATIO_BPS`, swept
  via the cost grid and the tax break-even table, and bypassed entirely by the
  selection-free control in `docs/results.md`.
- **As-of 2026-06-30**, and calendar-year work additionally drops any partial year, so a
  fund listed in October never contributes a stub year to a persistence test.
