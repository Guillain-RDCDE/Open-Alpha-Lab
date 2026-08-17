# References & literature map — Study 935 (Value Averaging)

## The claim under test

- **Edleson's rule.** Michael E. Edleson, *Value Averaging: The Safe and Easy Strategy
  for Higher Investment Returns* (1991; Wiley reissue 2007, foreword by William
  Bernstein). Instead of investing a fixed *amount* every month, fix the *value* the
  portfolio should reach each month and trade the difference. Because the value path
  climbs smoothly while the market does not, the rule buys more after falls and sells
  after rallies — an automatic contrarian schedule. Edleson reports that value
  averaging (VA) beats dollar-cost averaging (DCA) on an internal-rate-of-return
  basis "almost always", across simulated and historical paths.
- **The steelman.** The advantage is not a forecast: it is a mechanical consequence of
  varying the purchase size inversely with price, which lowers the average cost per
  share below the average price. That much is arithmetic and is not in dispute.
- **What is in dispute** — and what this study measures — is whether that lower average
  cost survives the two things the arithmetic quietly assumes: that the extra money VA
  demands in falling markets is *available*, and that the money VA is *not* investing
  in rising markets is counted at all.

## The known objection — the funding problem

- **Marshall (2000), "A Statistical Comparison of Value Averaging vs. Dollar Cost
  Averaging and Random Investment Techniques", *Journal of Financial and Strategic
  Decisions* 13(1).** VA's higher IRR is measured on a cash flow stream that VA itself
  chooses; the comparison is not on equal invested capital.
- **Marshall & Baldwin (1994), "A Statistical Comparison of Dollar-Cost Averaging and
  Purely Random Investing Techniques", *Journal of Financial and Strategic Decisions*
  7(2).** The earlier, blunter version of the same complaint: formation rules that
  vary the amount invested are not comparable to fixed-amount rules on rate-of-return
  metrics.
- **Constantinides (1979), "A Note on the Suboptimality of Dollar-Cost Averaging as an
  Investment Policy", *Journal of Financial and Quantitative Analysis* 14(2).** The
  general result behind both: any *deterministic, sequential* investment schedule is
  dominated by an unconditional policy, because the schedule is not conditioning on
  anything the market has revealed. VA is such a schedule.
- **Leggio & Lien (2003), "An Empirical Examination of the Effectiveness of
  Dollar-Cost Averaging Using Downside Risk Performance Measures", *Journal of
  Economics and Finance* 27(2).** Once the comparison is put on a risk-adjusted
  footing, the averaging rules lose most of their reported advantage.

## Why the residual is not evidence of skill

- **Volatility harvesting / the rebalancing bonus.** Fernholz & Shay (1982),
  "Stochastic Portfolio Theory and Stock Market Equilibrium", *Journal of Finance*
  37(2); Booth & Fama (1992), "Diversification Returns and Asset Contributions",
  *Financial Analysts Journal* 48(3). A rule that mechanically buys the cheaper asset
  and sells the dearer one earns a positive geometric premium **even when prices are a
  pure random walk** — no predictability is required. Our exposure-matched placebo is
  built on exactly this point: a calibrated random walk hands the same residual.
- **Mean reversion, if any, is the only thing that would make VA genuinely skilful.**
  Poterba & Summers (1988), "Mean Reversion in Stock Prices", *Journal of Financial
  Economics* 22(1); Lo & MacKinlay (1988), "Stock Market Prices Do Not Follow Random
  Walks", *Review of Financial Studies* 1(1). Our synthetic control plants precisely
  this (a transitory Ornstein-Uhlenbeck component, verified by a sub-one variance
  ratio) and confirms VA harvests it when it is there.

## Related desk studies (dedup)

- **[Study 934 — Lump Sum vs DCA](../../934-lump-sum-vs-dca/)**: the *other* half of
  the drip-feed question — how fast to deploy a windfall you already hold, with no
  rule that varies the tranche size. Study 935 holds the deployment schedule's
  *length* fixed and varies the tranche *size* by rule, which is where the funding
  problem appears.
- **[Study 101 — Slow and Steady](../../101-slow-and-steady/)**: the plain DCA-versus-
  lump-sum comparison on daily rolling windows with idle cash pinned at 0%. Study 935
  keeps a live T-bill leg and, crucially, an explicitly *finite* buffer.
- **[Study 102 — Free Rebalance](../../102-free-rebalance/)**: rebalancing a fixed
  multi-asset portfolio. Value averaging is a rebalance against a *growing target on a
  single sleeve*, funded by external contributions rather than by a sibling asset.
- **[Study 596 — Bond-Tent Glidepath](../../596-bond-tent-glidepath/)**: a *scheduled*
  change in equity weight around retirement. Study 935's punchline is that VA is an
  *unscheduled* one — the growth rate of the value path is an equity-weight dial that
  the rule never announces.
- **[Study 604 — Month-End Rebalancing Flows](../../604-month-end-rebalancing-flows/)**:
  the market-impact side of month-end rebalancing. Here the month-end is only the
  decision date; no flow effect is claimed.

## Method lineage

- **HAC / Newey-West t-stat.** Newey & West (1987), *A Simple, Positive Semi-Definite
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix*, Econometrica —
  [`strategy.newey_west_t`](../value_avg/strategy.py), lag-truncated at the window
  horizon because that is the span over which two windows share tape.
- **Circular block bootstrap.** Politis & Romano (1994), *The Stationary Bootstrap*,
  JASA — [`strategy.block_bootstrap_ci`](../value_avg/strategy.py).
- **Wilson score interval.** Wilson (1927), *Probable Inference, the Law of Succession,
  and Statistical Inference*, JASA — [`strategy.wilson_interval`](../value_avg/strategy.py).
- **Money-weighted return.** The IRR bisection in
  [`strategy.irr_annual`](../value_avg/strategy.py) is the standard dated-cashflow
  solve; it is computed twice on purpose — once on the equity flows only (Edleson's
  own metric) and once on the whole programme.

## Data sources & non-tape inputs

- **SPY** (equity sleeve), **BIL** (1-3M T-bill, the cash leg), **IEF** and **QQQ**
  (cross-check sleeves) — daily **total-return** closes via `yfinance`
  (`auto_adjust=True`). The common SPY∩BIL window starts at BIL's 2007 inception,
  which is the honest start for any comparison whose buffer must earn a *real*,
  tradable cash rate rather than an assumed one. As-of **2026-06-30**; the partial
  current month is dropped.
- **Non-tape inputs, all labelled and swept.** The value path's growth rate
  (ASSUMPTION: 0%, i.e. Edleson's linear path, swept to 12%); the cash buffer
  (ASSUMPTION: 6 monthly contributions, swept 0-24); the one-way cost (1 bp, swept to
  25 bp); the programme horizon (36 months, swept 24-120). None of these come from the
  tape, and the first of them is the one that decides the sign of the answer.
- **Survivorship.** SPY, IEF and QQQ are index vehicles that have survived the whole
  window; their *underlying* indices are themselves reconstituted, so the sleeve
  return embeds index-level survivorship. That biases the *level* of both arms
  identically and cancels in the VA-minus-DCA difference, which is the headline here.
