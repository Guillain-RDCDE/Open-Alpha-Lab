# References & literature map — Study 921 (Bill Ladder vs ETF)

## The claim under test

- **The do-it-yourself cash thesis.** A cash ETF holds Treasury bills and rolls them. A
  retail investor can hold Treasury bills and roll them, at TreasuryDirect, for nothing. The
  folk conclusion — repeated on every personal-finance forum since short rates came back in
  2022 — is that running your own 3-month bill ladder must beat BIL or SGOV by the expense
  ratio, "free money for ten minutes a quarter". This study asks whether that arithmetic
  survives contact with the actual tape, and what it costs to collect.
- **The steelman.** Unlike almost every claim on this desk, the mechanism here is not a
  behavioural anomaly that can be arbitraged away: it is an accounting identity. If two
  portfolios hold the same instrument and one of them pays a manager, the other must win by
  the fee. The only empirical questions are (a) whether a ladder built from a *published
  rate index* really tracks a bill fund's gross return, and (b) whether the frictions of
  running it are smaller than the fee avoided.

## The instrument and its arithmetic

- **Treasury bill quoting conventions.** U.S. Treasury, *Uniform Offering Circular for the
  Sale and Issue of Marketable Book-Entry Treasury Bills, Notes, and Bonds* (31 CFR Part
  356) — bills are auctioned on a bank-discount basis, actual/360, and the Treasury
  separately publishes the "investment rate" (coupon-equivalent, actual/365). The conversion
  used here, `P = 1 − d·t/360` then `BEY = (1−P)/P · 365/t`, is that standard identity for
  bills of 182 days or fewer. Stigum & Crescenzi, *Stigum's Money Market* (4th ed., 2007),
  ch. 2, is the canonical treatment of why a discount rate systematically understates the
  yield an investor actually earns.
- **^IRX.** The Cboe 13-Week Treasury Bill index tracks the discount rate on the most
  recently auctioned 13-week bill. It is a **secondary-market quote**, not an auction
  stop-out: a real ladder buying at auction would earn the auction's high rate, which
  differs from the screen by a basis point or two in either direction. That is the study's
  main PROXY and the reason the raw-quote floor is reported alongside the converted headline.
- **Bill ladders and amortised cost.** Money-market and bill portfolios held to maturity are
  conventionally carried at amortised cost (SEC Rule 2a-7 permits it for compliant money
  funds). The near-zero measured volatility of a ladder is therefore an artefact of that
  convention; Investment Company Institute, *Report of the Money Market Working Group*
  (2009), discusses the divergence between amortised cost and mark-to-market ("shadow")
  pricing precisely because it hides, rather than removes, risk.

## Why the fee should be the whole story

- **Sharpe (1966) and the fee identity.** In an asset class with no dispersion of holdings,
  gross returns are common and net returns differ only by cost. Bogle (2014), *The Arithmetic
  of "All-In" Investment Expenses*, Financial Analysts Journal, makes the general case; cash
  is the purest instance of it, because a 1-3 month bill fund has essentially no security
  selection left to disagree about.
- **Cost is the one predictor that works.** Carhart (1997), *On Persistence in Mutual Fund
  Performance*, Journal of Finance, and Fama & French (2010), *Luck versus Skill in the
  Cross-Section of Mutual Fund Returns*, Journal of Finance — expenses predict net returns
  approximately one-for-one. This study's gross-of-fee residual of −0.4 bps/yr against BIL
  is that one-for-one relation measured in a setting where it should be exact.
- **Where the identity leaks.** Duffie (1996), *Special Repo Rates*, Journal of Finance, and
  the on-the-run/off-the-run literature: a fund with scale can lend its bills and earn
  specialness a retail ladder cannot; conversely the fund pays a spread on every trade the
  laddered holder avoids by holding to maturity. The measured residual bounds the net of
  these two.

## Related desk studies (dedup)

- **[Study 892 — Corporate-Bond Ladder](../../892-corporate-bond-ladder/)**: the ladder-vs-fund
  question in **credit**, where the fund holds different bonds from the ladder and the answer
  turns on default and rolldown. Study 921 strips that away: at the bill maturity the two
  portfolios hold the *same* instrument, so the comparison isolates the fee itself.
- **[Study 885 — Ultra-Short Credit Pickup](../../885-ultra-short-credit-pickup/)**: whether
  JPST/ICSH/MINT pay a real excess-of-bills pickup — a **credit** question, taking bills as
  the risk-free benchmark. Study 921 turns around and audits the benchmark.
- **[Study 380 — Curve Roll-Down](../../380-curve-roll-down/)**: riding *down* the curve for a
  term premium, i.e. deliberately taking duration. Study 921's ladder takes none — every rung
  is held to maturity — and SHV is included precisely as the duration control that shows what
  that choice forgoes (−16.6 bps of residual).
- **[Study 603 — Treasury Auction Concession](../../603-treasury-auction-concession/)**: the
  price pattern *around* auctions. Study 921 uses auctions only as the ladder's roll clock and
  never trades the concession.
- **[Study 625 — Starting Yield](../../625-starting-yield-bond-decade/)** and
  **[Study 613 — Currency-Hedged Carry](../../613-currency-hedged-etf-carry/)**: the desk's other
  mechanical-identity results. Study 921 belongs to that family — a real effect whose size is
  fixed by arithmetic rather than by a risk premium.

## Method lineage

- **HAC / Newey-West t-stat.** Newey & West (1987), *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix*, Econometrica —
  [`strategy.newey_west_t`](../bill_ladder/strategy.py) and
  [`quantlab.analytics.mean_tstat_hac`](../../../quantlab/analytics.py). Note the direction of
  travel here: the bid-offer bounce in a bill ETF's close makes the return difference
  *negatively* autocorrelated, so HAC **tightens** the standard error rather than loosening it.
- **Bid-ask bounce as negative autocorrelation.** Roll (1984), *A Simple Implicit Measure of
  the Effective Bid-Ask Spread in an Efficient Market*, Journal of Finance — the reason the
  naive *t* (+1.15) is the over-conservative one. Measured directly here:
  [`strategy.acf1`](../bill_ladder/strategy.py) puts the lag-1 autocorrelation of the daily
  difference at **−0.366**, which is the bounce signature and not an assumption.
- **Circular block bootstrap.** Politis & Romano (1994), *The Stationary Bootstrap*, JASA —
  [`strategy.bootstrap_gap_ci`](../bill_ladder/strategy.py) and
  [`quantlab.stats.sharpe_ci_bootstrap`](../../../quantlab/stats.py). Applied to the annualised
  *gap* rather than a Sharpe, since the ladder's amortised-cost denominator is meaningless.
- **Non-overlapping temporal aggregation as a bandwidth-free alternative to HAC.** The
  standard remedy when a HAC bandwidth choice is doing visible work — see e.g. Cochrane
  (2005), *Asset Pricing*, ch. 20 on long-horizon regressions, and the Hansen-Hodrick /
  Newey-West bandwidth-sensitivity literature. Summing a bounce-contaminated daily series
  into **non-overlapping** periods lets the transitory pricing error telescope inside each
  block while the drift accumulates, leaving near-independent observations on which the
  ordinary *t* is valid with nothing to tune:
  [`strategy.nonoverlap_t`](../bill_ladder/strategy.py) and `strategy.horizon_check`.
  **This study's Real stamp rests on that test** (weekly +2.18, monthly +3.27, quarterly
  +3.54), not on the kernel — HAC (+2.75) and the bootstrap merely concur. The HAC bandwidth
  and bootstrap block-length scans are published alongside it
  ([`strategy.hac_bandwidth_scan`](../bill_ladder/strategy.py), `docs/results.md`) precisely
  because both knobs move the result in the study's favour and must not be hidden.

## Data sources

- **BIL** (SPDR Bloomberg 1-3 Month T-Bill), **SGOV** (iShares 0-3 Month Treasury), **SHV**
  (iShares Short Treasury, ≤1 year — the duration control) — daily **total-return** closes via
  `yfinance` (`auto_adjust=True`). Total return is mandatory here, not a nicety: a bill ETF's
  price is flat by construction and its entire return arrives as monthly distributions.
- **^IRX** — Cboe 13-Week Treasury Bill rate index, daily close **in percent on a bank-discount
  basis**, from 1990. Used only to price the ladder's rungs; converted to bond-equivalent as
  documented above.
- **Expense ratios** (BIL 13.54 bps, SGOV 9 bps, SHV 15 bps) are **PROXY / ASSUMPTION** —
  sponsor-published stickers as of 2026, hardcoded in `data.EXPENSE_RATIO_BPS`. They never
  enter a return calculation; they appear only in the gross-of-fee attribution table, and
  SGOV's has changed since its 2020 launch, which the results file calls out.
- **Per-auction friction and reinvestment idle days** are **PROXY / ASSUMPTION** and are swept
  (0-10 bps and 0-5 days respectively) rather than assumed.
- **As-of 2026-06-30.** The partial current month is dropped so the sample never creeps. The
  ^IRX∩ETF common window is the honest tradable window for each fund.
