# References & literature map — Study 948 (Capture Ratio)

## The claim under test

- **The income-fund pitch.** Every covered-call and high-dividend ETF prospectus and fact
  sheet makes some version of the same promise: the fund "gives up a little upside in
  exchange for income and downside protection", producing a smoother ride than the index.
  Stated precisely, that is a claim about **capture ratios** — the fund should keep a
  larger share of the benchmark's up months than of its down months, i.e. its **capture
  spread** (up-capture minus down-capture) should be **positive**.
- **Why the spread, and not the two ratios separately.** A fund with up-capture 0.50 looks
  cautious and a fund with down-capture 0.50 looks protective, but a fund that is simply
  half the index has *both*, and has delivered nothing a smaller position would not. Only
  the *difference* is a claim about the shape of the payoff rather than its size. Every
  fact sheet quotes the two ratios; almost none quotes the difference, and none quotes a
  confidence interval on it.
- **The steelman.** A covered-call fund collects option premium every month. If the
  premium it collects exceeds the upside it forfeits — the variance risk premium, which is
  positive on average — the fund can in principle deliver a positive capture spread rather
  than merely a lower beta. That is a testable mechanical claim, and this study tests it on
  the real, costed, excess-of-cash monthly tape.

## Where capture ratios come from, and what is wrong with them

- **Morningstar's up/down capture methodology** — the industry standard scorecard, computed
  on monthly returns against a fund's category benchmark and published on every fund page.
  Both the arithmetic (ratio of mean up-month returns) and geometric (ratio of annualised
  compounded up-month returns) conventions are in circulation; this study reports both,
  because a headline that flips between the two was never a finding.
- **Israelov & Nielsen (2015), *Covered Calls Uncovered*, Financial Analysts Journal** — the
  decomposition that names the problem: a covered-call position is a passive equity
  exposure plus a short volatility exposure, and most of its return and risk come from the
  equity leg. What the fund distributes is largely its own capital, not a premium harvested
  on top. Our finding that the capture spread is ~0 while the *kink intercept* is fat is the
  same statement in the capture-ratio language.
- **Israelov & Klein (2016), *Risk and Return of Equity Index Collar Strategies*** — collars
  (NUSI's design) look like they hedge, but their measured downside protection is dominated
  by the reduced equity exposure. Our NUSI row — the panel's largest spread (+0.164) sitting
  on a *negative* convexity coefficient and a CI that straddles zero — is the ratio version
  of that.
- **Henriksson & Merton (1981), *On Market Timing and Investment Performance II*, Journal of
  Business** — the piecewise regression this study leans on:
  `r_f = a + b_up·max(r_b,0) + b_dn·min(r_b,0) + u`. The convexity `b_up − b_dn` is the
  capture spread's well-behaved twin, testable with a standard error and free of the
  ratio's exploding denominator. Implemented in
  [`strategy.hm_regression`](../capture_ratio/strategy.py).
- **Treynor & Mazuy (1966), *Can Mutual Funds Outguess the Market?*, Harvard Business
  Review** — the quadratic ancestor of the same convexity test, and the origin of the idea
  that market timing shows up as curvature in the fund-vs-benchmark scatter.
- **The ratio-estimator problem.** A capture ratio is a **ratio of two sample means** whose
  denominator (the mean down month) is small and negative. Ratio estimators of this kind are
  biased in small samples — the classic result behind Fieller (1954), *Some Problems in
  Interval Estimation*, JRSS-B, and the reason Fieller intervals exist at all. This study
  measures the bias directly on zero-truth synthetic panels
  ([`strategy.null_spread_distribution`](../capture_ratio/strategy.py)) and finds the
  arithmetic capture spread reads **positive 82% of the time when the truth is exactly
  zero**.
- **Conditional vs unconditional bias — the correction we did *not* make.** That +0.111 is
  an *unconditional* bias: it averages over random benchmark paths. A real-tape spread is
  computed on **one** realised path, and conditioning on that path removes most of it, so
  subtracting the synthetic number from the real one would overstate the case by roughly 3×.
  The study therefore quotes a **fund-matched null**
  ([`strategy.matched_null_spread`](../capture_ratio/strategy.py)): each fund is rebuilt
  1,000 times as its own zero-convexity twin — its own beta and residual vol, block-bootstrapped
  residuals, the real benchmark and cash paths — and re-scored with the same estimator. This
  is a parametric-bootstrap reference distribution in the sense of Efron & Tibshirani (1993),
  *An Introduction to the Bootstrap*, ch. 6, and the conditioning argument is the usual one
  for ancillary regressors (Cox, 1958, *Some Problems Connected with Statistical Inference*).
- **Alpha masquerades as convexity.** For a perfectly linear fund, the capture spread is
  `a·(1/mean_up − 1/mean_down)`, and because `mean_down < 0` the two terms *add*. So any
  positive alpha reads as a positive capture spread with no convexity whatsoever, and an OLS
  line fitted to a genuinely convex fund answers with a large positive intercept that
  reproduces that fund's spread. This is why the study's inference bar is Henriksson-Merton
  (free intercept, separates the two) and not the spread itself. Tested in
  `test_alpha_alone_manufactures_a_positive_capture_spread` and
  `test_alpha_beta_twin_reproduces_the_observed_spread`.

## Why the claim can fail

- **Bakshi & Kapadia (2003), *Delta-Hedged Gains and the Negative Market Volatility Risk
  Premium*, Review of Financial Studies** — the variance risk premium is real, but it is
  earned by a *delta-hedged* short-vol position. An unhedged covered-call fund earns it
  jointly with a truncated equity exposure, and the truncation is not free.
- **Whaley (2002), *Return and Risk of CBOE Buy Write Monthly Index*, Journal of
  Derivatives** — the original BXM study, and the source of the "same return, less risk"
  framing. Later out-of-sample work (and every live buy-write ETF since 2013) has been much
  less kind.
- **Fee drag.** These funds charge 0.35%–0.68% a year against benchmarks available for
  0.03%. A fee is a constant subtraction: it lowers alpha without touching the capture
  *shape*, which is exactly the pattern the panel shows (zero convexity, negative alpha
  point estimates, none significant).
- **Multiplicity.** Screening fourteen funds at a nominal |*t*| ≥ 2 gives roughly a
  one-in-two chance of at least one false positive under the null — Harvey, Liu & Zhu
  (2016), *… and the Cross-Section of Expected Returns*, Review of Financial Studies. The
  family-wise Bonferroni bar (|*t*| ≥ 2.91 for fourteen tests) is what a fund screen has to
  clear, and it is what this study quotes.

## Related desk studies (dedup)

- **[Study 62 — Premium-Seller](../../62-premium-seller/)**: races **QYLD against QQQ** on
  total return and *quotes* its 50%/58% up/down capture as the explanation for an 11%/yr
  shortfall. Study 948 turns that descriptive statistic into the **object of inference**
  and generalises it: fourteen funds instead of one, a bootstrap CI and an HM convexity
  *t* on the spread instead of a bare pair of percentages, a family-wise bar instead of a
  nominal one, a rank-persistence test across eras, and — the part study 62 could not have
  seen from a single fund — the demonstration that the capture spread is a **biased
  estimator** which, once its bias is measured on each fund's own sample, turns out to carry
  no information beyond that fund's alpha and beta.
- **[Study 337 — Covered-Call ETF](../../337-covered-call-etf/)**: decomposes the *headline
  distribution* of the buy-write funds into a NAV leg and a return-of-capital share. That
  is a question about where the cash comes from; 948 is a question about the **shape of the
  payoff**, and its answer (no convexity) holds whatever the distribution turns out to be.
- **[Study 658 — Put-Write Premium](../../658-put-write-premium/)**: PUTW vs SPY, i.e.
  whether the variance risk premium survives in a live *put-writing* wrapper. 948 covers
  call-writers and dividend funds and asks only about capture asymmetry.
- **[Study 899 — Cash-Plus-Call](../../899-cash-plus-call/)**: a 90/10 cash-and-call
  structure, the *long*-convexity mirror of these products.
- **[Study 900 — Quality-Income](../../900-quality-income/)** and
  **[Study 206 — Dividend-Aristocrats](../../206-dividend-aristocrats/)**: race dividend
  *sleeves* on returns and Sharpe. 948 includes SCHD/VYM/DVY/SPHD/NOBL only as the
  **no-options control group** for the capture measurement — if the spread were an artefact
  of low beta rather than of option-writing, these five would show it too (they do not
  show it any more than the writers do, which is the point).

## Method lineage

- **HAC / Newey-West standard errors.** Newey & West (1987), *A Simple, Positive
  Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix*,
  Econometrica — [`strategy.newey_west_t`](../capture_ratio/strategy.py),
  [`strategy.hac_ols`](../capture_ratio/strategy.py) and
  [`quantlab.analytics.mean_tstat_hac`](../../../quantlab/analytics.py).
- **Circular block bootstrap.** Politis & Romano (1992, 1994) — blocks of consecutive
  months resampled *jointly* across fund and benchmark, so the pairing and the local
  volatility clustering survive: [`strategy.bootstrap_capture_spread`](../capture_ratio/strategy.py)
  and [`quantlab.stats.sharpe_ci_bootstrap`](../../../quantlab/stats.py).
- **Rank persistence.** Carhart (1997), *On Persistence in Mutual Fund Performance*, Journal
  of Finance — the template for "does this ranking survive into the next period?", applied
  here to the capture spread rather than to returns:
  [`strategy.rank_persistence`](../capture_ratio/strategy.py).
- **Reproducibility stamp.** [`quantlab/repro.py`](../../../quantlab/repro.py) — as-of
  slicing and a content fingerprint on the exact input series.

## Data sources

- **Fourteen income ETFs** — QYLD, JEPQ, NUSI (vs QQQ); XYLD, JEPI, PBP, DIVO, SPYI, SCHD,
  VYM, DVY, SPHD, NOBL (vs SPY); RYLD (vs IWM) — plus **SPY / QQQ / IWM** and **BIL** (cash),
  daily **total-return** closes via `yfinance` (`auto_adjust=True`), resampled to monthly.
  Total return is not optional here: these funds pay most of their return out as
  distributions, and on a price-only tape every capture ratio in the table would be fiction.
- **As-of 2026-06-30**, the last complete calendar month; the partial month is dropped so
  the sample never creeps. Per-fund windows run from each fund's own inception (2007-06 for
  VYM/DVY, gated by BIL's 2007 inception, to 2022-09 for SPYI).
- **PROXY / ASSUMPTION register.** (1) The **fund → benchmark map** is hand-assigned from
  each fund's own stated reference index — swept by re-running the panel with SPY forced for
  every fund. (2) **BIL** is the cash proxy. (3) **Borrow** on the short leg of the traded
  arm is an assumed 0–200 bps/yr — swept. (4) **Trading cost** is an assumed 0–25 bps
  one-way × NAV — swept. (5) The **capture convention** (arithmetic vs geometric) is a
  choice — both are reported. (6) The **unadjusted-corporate-action screen** is an
  assumption about the vendor: a one-day price ratio within 3% of a real share-count factor
  (3:2, 2:1, 3:1, 4:1, 5:1, 10:1 or their reverse twins) is treated as a split and
  back-adjusted, anything else past ±35% is left in the tape and reported. The whitelist is
  deliberately short so the screen can say *no* — a dense list of every n/m would tile the
  number line and smooth a genuine crash away. On this universe it fires exactly once, on
  **NUSI 2025-02-18 (ratio 2.000000)**, and `load_prices(repair_splits=False)` reproduces
  the unrepaired tape. No hardcoded expense ratios or distribution schedules enter anywhere:
  the total-return tape already carries them.
- **Survivorship.** The panel is the set of income ETFs still listed as of 2026-06-30. The
  buy-write and high-dividend funds that liquidated over the same period leave no trace on
  this tape, so the scorecard is biased **in the funds' favour**. Named on the Signal axis.
