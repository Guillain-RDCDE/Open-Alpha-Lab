# References & literature map — Study 920 (Total Cost of Ownership)

## The claim under test

- **The total-cost identity.** What an index fund costs you is not its expense ratio. It is
  the expense ratio *plus* everything else the wrapper does to your return — securities-lending
  revenue handed back or kept, sampling error, uninvested cash, the legal form of the trust —
  which together show up as the **realised tracking difference**; *plus* the **round-trip
  spread** you pay to get in and out. The first two are a carry, charged per year of holding.
  The third is a toll, charged once. So the cheap-but-wider wrapper and the expensive-but-tighter
  one cross at a **break-even holding period**, and the whole question is where that crossing sits.
- **The specific structural bet.** SPY (1993) and QQQ (1999) are *unit investment trusts*: by
  their legal form they cannot reinvest dividends between distribution dates and cannot lend
  securities. Their modern clones — IVV, VOO, QQQM — are open-end funds that can do both, and
  charge 3 bp and 15 bp against SPY's 9.45 bp and QQQ's 20 bp. The prediction is a realised
  tracking difference of roughly the fee gap, possibly a touch wider because of the cash drag
  the trust form imposes. The prediction the study can falsify is that this shows up on the
  tape at all, at the size the prospectuses imply.

## Why the mechanism is real — the wrapper literature

- **Elton, Gruber, Comer & Li (2002), *Spiders: Where Are the Bugs?*, Journal of Business 75(3).**
  The original diagnosis of SPY's unit-investment-trust drag: dividends sit in a
  non-interest-bearing account until the quarterly distribution, so the fund under-performs its
  index by more than its expense ratio in rising markets. **This is the study's principal
  confound, not a footnote:** SPY and QQQ are the expensive leg in *every* pair that shows a
  gap, so the realised tracking difference we measure is fee gap + trust cash drag + lending
  pass-through, and no pair on the tape separates them. Consistent with the drag, QQQM comes in
  +2.19 bp *wider* than QQQ's stated 5 bp gap — but IVV comes in 0.67 bp *narrower* than SPY's
  6.45, so the tape does not cleanly show the extra drag either. The agreement with the
  prospectus numbers is the sum of at least two effects; treat it as a plausibility check, not
  as a confirmation that the fee is what was measured.
- **Elton, Gruber & Busse (2004), *Are Investors Rational? Choices Among Index Funds*, Journal
  of Finance 59(1).** Same index, different wrappers, different total costs — and investors
  still buy the expensive ones. The paper's finding that future performance across S&P 500
  index funds is almost entirely predicted by expenses is the prior this study measures.
- **Petajisto (2017), *Inefficiencies in the Pricing of Exchange-Traded Funds*, Financial
  Analysts Journal 73(1).** Premiums and discounts to NAV, and how much of an ETF's apparent
  tracking error is a pricing artefact of the closing print rather than the portfolio. This is
  the source of the noise floor that dominates our full-history estimates.
- **Blocher & Whaley (2016), *Two-Sided Markets in Asset Management: Exchange-Traded Funds and
  Securities Lending*.** Lending revenue as a component of realised tracking difference and how
  much of it is passed through — one reason realised TD and stated fee need not coincide, and
  the reason a same-fee pair (VOO vs IVV) can still drift a basis point apart.
- **Hortaçsu & Syverson (2004), *Product Differentiation, Search Costs, and Competition in the
  Mutual Fund Industry*, Quarterly Journal of Economics 119(2).** Why near-identical index
  products sustain very different fees — the market structure that makes this study's question
  worth asking rather than arbitraged away.

## Why the measurement is harder than it looks

- **Closing-print noise.** The daily difference between two S&P 500 wrappers has a standard
  deviation of roughly 5 bp — annualising to ~80 bp against a 6 bp signal. It is *transient*
  (a level error in the close, not a drift), so it telescopes across chained periods: on the
  QQQ/QQQM pair the monthly tracking differences have a standard deviation near 10 bp while
  their twelve-month sums have one near 3. Any monthly-frequency interval therefore overstates
  the uncertainty by an order of magnitude, which is why this study's bootstrap runs on
  complete years. Compare **Roll (1984)**, *A Simple Implicit Measure of the Effective Bid-Ask
  Spread*, Journal of Finance 39(4), on bid-ask bounce as negatively autocorrelated price noise.
- **Small-sample intervals.** Running the bootstrap on complete years costs sample size: the
  common window has *five*. A percentile block bootstrap on five points is materially too
  narrow — it resamples an empirical distribution built from the very numbers whose dispersion
  it is trying to price — so the study reports a **Student-*t* interval beside it** and quotes
  the *t* interval for every pessimistic claim. On IVV/SPY the two are [+3.55, +7.90] and
  [+0.28, +11.28]. See **Efron & Tibshirani (1993)**, *An Introduction to the Bootstrap*, ch. 12–14,
  on percentile-interval coverage failure at small *n*.
- **Adjustment artefacts.** Public total-return series occasionally time a distribution into
  the wrong session, producing a single-year tracking difference of 40–57 bp between two funds
  that charge the *same* fee. The study names these, reports median and trimmed-mean
  sensitivities beside the mean, and uses the same-fee VOO/IVV pair as the placebo that makes
  the artefact visible.
- **Power.** Newey & West (1987), *A Simple, Positive Semi-Definite, Heteroskedasticity and
  Autocorrelation Consistent Covariance Matrix*, Econometrica 55(3) — the HAC *t* used on the
  monthly series and the overlapping holding-period races. Politis & Romano (1994), *The
  Stationary Bootstrap*, JASA 89(428) — the circular block bootstrap behind the tracking-difference
  intervals. Wilson (1927), *Probable Inference, the Law of Succession, and Statistical
  Inference*, JASA 22(158) — the score interval on the holding-period win rates.

## Related desk studies (dedup)

- **[Study 913 — Tracking-Difference Persistence](../../913-tracking-difference-persistence/)**:
  the *same wrappers*, the opposite question. 913 asks whether last year's best tracker is next
  year's best — a **time-series persistence** test of a ranking. Study 920 does not rank or
  time anything: it measures the *level* of the realised tracking difference and converts it,
  with a swept spread assumption, into a **break-even holding period**. 913's answer would not
  change 920's, and vice versa.
- **[Study 378 — ETF-NAV-Premium](../../378-etf-nav-premium/)**: the *transient* premium or
  discount of an ETF to a fair-value **proxy**, and whether it mean-reverts. That is a
  short-horizon pricing wobble; 920 measures the permanent drift underneath it and explicitly
  treats the wobble as the noise floor to be defeated.
- **[Study 379 — ETF Lead-Lag](../../379-etf-lead-lag/)**: whether one ETF's move predicts
  another's — a timing signal. Nothing here predicts anything.
- **[Study 621 — Share-Class Spreads](../../621-share-class-spreads/)**: two *share classes of
  one company* held apart by a conversion bound. 920's pairs are separate funds from separate
  sponsors held together by a shared index, and the quantity of interest is a fee, not an
  arbitrage bound.
- **[Study 624 — Buffer-ETF Cost](../../624-buffer-etf-cost/)** and
  **[Study 622 — Thematic-ETF Curse](../../622-thematic-etf-curse/)**: what an *exotic* wrapper
  costs relative to a plain one. 920 is the plain-vanilla case, where the wrappers are
  genuinely interchangeable and the only difference is the price tag.
- **[Study 601 — Factor-ETF Live Test](../../601-factor-etf-live-test/)**: whether live factor
  funds delivered their backtested factor. That is an implementation-shortfall question about
  a *strategy*; 920's funds implement no strategy at all.

## Data sources & declared assumptions

- **SPY, IVV, VOO, QQQ, QQQM** (index wrappers) and **BIL** (1-3M T-bill, the cash leg) —
  daily **total-return** closes via `yfinance` (`auto_adjust=True`), through **2026-06-30**.
  Total return is mandatory here: on price-only closes a difference in distribution schedule
  would masquerade as tracking difference.
- **SPLG is excluded.** Yahoo's SPLG series carries a first-trade date of 2026-07-17 at build
  time and returns a single session — the history has been reset upstream. Named rather than
  silently dropped; no result depends on it.
- **ASSUMPTION — stated expense ratios** (SPY 9.45, IVV 3.0, VOO 3.0, QQQ 20.0, QQQM 15.0
  bp/yr, from the funds' prospectuses at the as-of date). Used only as a *prior* to compare
  the realised tracking difference against; never as an input to any return.
- **ASSUMPTION — round-trip spreads.** A daily-close tape carries no quotes. The spread
  differential is swept over 0–10 bp end to end and no headline number rests on a single
  value; the indicative per-fund figures in `data.ASSUMED_RT_SPREAD_BPS` exist only to point
  at the plausible region of that sweep.
- **ASSUMPTION — borrow rate.** Only the long/short harvest has a short leg; its borrow cost
  is swept over 0–100 bp/yr, and the sweep is the finding.
- **As-of 2026-06-30**, with partial calendar months and partial calendar years dropped
  everywhere, so the sample cannot creep between reruns.
