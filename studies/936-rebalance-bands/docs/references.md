# References & literature map — Study 936 (Tolerance Bands)

## The claim under test

- **The 5/25 rule.** Larry Swedroe's widely-repeated tolerance-band heuristic (popularised
  in *The Only Guide to a Winning Investment Strategy You'll Ever Need* and by the
  Bogleheads wiki entry *Rebalancing*): rebalance a sleeve when it is off target by **5
  percentage points absolute** or **25% of its own target weight**, whichever binds first.
  The selling point is that a *state*-triggered rule dominates a *calendar*-triggered one —
  you trade only when the book has actually drifted, so you get better weight discipline
  **and** a better risk-adjusted return for **less** turnover than quarterly (or even
  annual) rebalancing.
- **The steelman.** Under mean-reverting relative performance, acting on dispersion is
  strictly better information than acting on the date: bands sell whatever has run and buy
  whatever has lagged *at the moment the gap is largest*, which is when the reversal
  premium is largest. A calendar rule trades on a date chosen by the Gregorian calendar,
  which carries no information about the book at all.

## Where the rule comes from

- **Daryanani (2008), "Opportunistic Rebalancing: A New Paradigm for Wealth Managers",
  *Journal of Financial Planning*.** The canonical quantitative case for tolerance bands:
  checking bi-weekly with ~20% relative tolerance bands is reported to add roughly
  0.5 pp/yr over annual calendar rebalancing. This is the single strongest published
  version of the claim and the one Study 936 tries to reproduce out of sample.
- **Masters (2003), "Rebalancing", *Journal of Portfolio Management*** — the optimal-band
  framing: a no-trade region whose width trades off tracking error against transaction
  cost. Donohue & Yip (2003), *Optimal Portfolio Rebalancing with Transaction Costs*,
  JPM — the same problem solved dynamically.
- **Swedroe & Grogan, and the Bogleheads wiki *Rebalancing* entry** — where "5/25" became
  retail folklore. Note that neither source presents 5/25 as an optimised parameter; it is
  a round number chosen for memorability, which is why this study **sweeps the band width**
  (2/10 through 10/50) rather than treating 5/25 as given.
- **Vanguard (2015/2022), *Getting Back on Track: A Guide to Smart Rebalancing*
  (Zilbering, Jaconetti & Kinniry).** Vanguard's own conclusion after testing monthly,
  quarterly and annual schedules against 1%, 5% and 10% thresholds on a 60/40 book is
  that **no schedule dominates on risk-adjusted return** — the choice matters for cost and
  for weight discipline, not for Sharpe. Our result is a direct, independently-costed
  replication of that conclusion on 2007-2026 ETF total returns.

## Why the calendar-versus-band difference is small in theory

- **Booth & Fama (1992), "Diversification Returns and Asset Contributions", *Financial
  Analysts Journal*** and **Willenbrock (2011), "Diversification Return, Portfolio
  Rebalancing, and the Commodity Return Puzzle", *FAJ*** — the "rebalancing bonus" is a
  variance identity, second-order in the dispersion of returns. Anything that changes only
  the *timing* of the reset within that identity is third-order: the effect a schedule
  choice can possibly have is bounded far below the noise of a twenty-year sample.
- **Pliska & Suzuki (2004), *Optimal tracking for asset allocation with fixed and
  proportional transaction costs*, Quantitative Finance** — the optimal policy under
  proportional costs *is* a no-trade band, but its welfare gain over a reasonable calendar
  rule is small whenever costs are small. Modern ETF costs (a basis point or two) are
  exactly that regime, which is why our cost sweep barely moves the ranking.
- **Cost is not the binding constraint but *tax* may be.** Every study here is pre-tax; in
  a taxable account the higher-turnover schedule realises gains earlier, and that drag is
  first-order relative to the Sharpe differences we measure. It is named as an unmodelled
  assumption in the README rather than guessed at.

## Related desk studies (dedup)

- **[Study 97 — Balancing-Act](../../97-balancing-act/)**: is a fixed **60/40** better than
  **100% equity**? That is an *allocation* question (how much bond sleeve to own) and it
  holds the annual schedule fixed. Study 936 holds the *allocation* fixed and varies only
  the **schedule** — nothing in 936 changes what the book owns on average.
- **[Study 102 — Free-Rebalance](../../102-free-rebalance/)**: is there a **rebalancing
  bonus** — does rebalancing beat *letting the book drift*? That is the rebalanced-vs-drift
  identity. Study 936 takes rebalancing as given (the drift book appears only as a
  reference row) and asks the next question down: **which trigger** — calendar or tolerance
  band.
- **[Study 604 — Month-End Rebalancing Flows](../../604-month-end-rebalancing-flows/)**:
  whether *other people's* rebalancing moves prices at month-end — a market-impact effect
  you would trade against. Study 936 is about your own book's schedule, and charges (rather
  than harvests) the flow.
- **[Study 68 — All-Weather](../../68-all-weather/)** and
  **[Study 890 — Sector Risk Parity](../../890-sector-risk-parity/)**: *volatility*-weighted
  targets that move over time. Study 936's targets are fixed by construction.

## Method lineage

- **HAC / Newey-West t-stat.** Newey & West (1987), *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix*, Econometrica —
  [`strategy.newey_west_t`](../rebal_bands/strategy.py) and
  [`quantlab.analytics.mean_tstat_hac`](../../../quantlab/analytics.py).
- **Return-difference (Sharpe comparison) test.** Jobson & Korkie (1981), *Performance
  Hypothesis Testing with the Sharpe and Treynor Measures*, Journal of Finance; Ledoit &
  Wolf (2008), *Robust performance hypothesis testing with the Sharpe ratio*, Journal of
  Empirical Finance — the HAC-on-the-difference form used in
  [`strategy.diff_tstat`](../rebal_bands/strategy.py).
- **Circular block bootstrap.** Politis & Romano (1992/1994) — the paired resampling in
  [`strategy.bootstrap_diff_ci`](../rebal_bands/strategy.py) and
  [`quantlab.stats.sharpe_ci_bootstrap`](../../../quantlab/stats.py).
- **As-of / fingerprint stamping.** [`quantlab.repro`](../../../quantlab/repro.py).

## Data sources

- **SPY** (US equity), **IEF** (7-10y Treasuries), **GLD** (gold), **BIL** (1-3 month
  T-bills, the cash leg) — daily **total-return** closes via `yfinance`
  (`auto_adjust=True`), read from the shared desk cache `studies/_cache`. Total return, not
  price only, is essential: the bond sleeve's coupon is most of its return, and a price-only
  IEF series would misstate both the level and the drift of the weights.
- **Headline window 2007-05-30 → 2026-06-30**, set by BIL's inception (the excess-of-cash
  race needs a *tradable* cash leg, not a synthetic bill rate). A longer **2003-2026**
  SPY/IEF cross-check runs **gross of cash**, because BIL does not exist before 2007. The
  cash leg cancels exactly in the daily *return difference* (so the HAC *t* is unchanged),
  but a Sharpe ratio is nonlinear and the Sharpe *difference* is **not** cash-invariant, so
  the long window is compared only against the gross-of-cash comparator that `verify.py`
  prints for the headline window (−0.034), never against the excess-of-cash headline
  (−0.026).
- **Fingerprints are taken on the returns frame, not the levels frame.** Dividend-adjusted
  closes are rescaled by every new distribution, so a levels fingerprint churns on refetch
  while every statistic is unchanged; the returns fingerprint is what actually pins the run.
- **As-of 2026-06-30** — the last complete calendar month; the partial current month is
  dropped so the sample cannot creep between reruns.
