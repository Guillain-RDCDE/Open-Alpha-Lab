# References & literature map — Study 940 (The Turnover Budget)

## The claim under test

- **The rebalance-frequency folk theorem.** Every practitioner note on momentum carries the
  same throwaway line: rebalance faster and you track the signal better, but you pay for it
  in turnover, so there is an optimal frequency somewhere in the middle. The claim is almost
  never *priced*. This study prices it on one sleeve — cross-sectional 12-1 momentum on the
  eleven Select Sector SPDRs — by running the identical rule at daily, weekly, monthly and
  quarterly clocks and reporting, for each, the **break-even cost per unit of traded
  notional**: the level of friction at which that speed's net excess return hits zero.
- **The steelman.** A momentum signal decays. If its half-life is weeks, a quarterly clock
  holds stale ranks for most of its life and a daily clock holds fresh ones; the gross return
  should fall monotonically as the clock slows. Turnover moves the other way. Somewhere the
  curves cross, and *that crossing point*, not a Sharpe headline, is the deliverable.

## Where the signal comes from

- **Jegadeesh & Titman (1993),** *Returns to Buying Winners and Selling Losers*, Journal of
  Finance — the original cross-sectional momentum sort, and the source of the 12-1
  convention (rank on the trailing twelve months, skipping the most recent one to dodge
  short-term reversal).
- **Moskowitz & Grinblatt (1999),** *Do Industries Explain Momentum?*, Journal of Finance —
  industry momentum, the direct ancestor of a sector-ETF rotation sleeve. They find industry
  momentum is strong at the one-month horizon and largely subsumes individual-stock momentum
  in their sample. Our tape (sector ETFs, 1999–2026, net of the frictions an ETF book pays)
  does not carry it.
- **Chen, Chen, Hsin & Lee (2014),** *Sector Rotation and Monetary Conditions*, and the wide
  practitioner literature on SPDR-based rotation — the applied form of the same idea, and the
  reason the eleven SPDRs are the natural test bed: they are the sleeve people actually trade.

## Why the frequency question is the interesting one

- **Novy-Marx & Velikov (2016),** *A Taxonomy of Anomalies and Their Trading Costs*, Review
  of Financial Studies — the canonical statement that an anomaly's turnover, not its gross
  spread, decides whether it survives; they report net returns per unit of turnover across
  the zoo. This study is a single-sleeve, four-speed instance of their method.
- **Frazzini, Israel & Moskowitz (2015),** *Trading Costs of Asset Pricing Anomalies*,
  AQR/SSRN — real live-execution costs for momentum books, and the observation that
  academic cost assumptions swing net conclusions by more than the signals do. Our cost
  surface reproduces that in miniature: the ranking of speeds inverts at ~1 bp.
- **Korajczyk & Sadka (2004),** *Are Momentum Profits Robust to Trading Costs?*, Journal of
  Finance — the break-even-cost framing itself (the level of friction at which a strategy's
  profit vanishes), which is exactly the number this study reports per frequency.
- **Garleanu & Pedersen (2013),** *Dynamic Trading with Predictable Returns and Transaction
  Costs*, Journal of Finance — the theory behind partial rebalancing: with quadratic costs
  the optimal book trades *toward* the target rather than to it, so a fixed-clock ladder is a
  coarse approximation of an optimal-trading problem. We name that limitation rather than
  claim to have solved it, and report the churn/drift split of turnover so a reader can see
  how much a no-trade band could have saved (here: little — ~92% of trading is rank churn).

## Related desk studies (dedup)

- **[Study 28 — Carousel](../../28-carousel/)** and **[Study 225 — Sector-Rotation](../../225-sector-rotation/)**:
  both ask *whether* a sector-rotation signal pays (chasing the hottest sectors; cycle-phase
  rotation). Study 940 holds the signal fixed and varies only the **rebalance clock**, and its
  deliverable is a break-even cost per unit turnover rather than a Sharpe.
- **[Study 836 — Rebalance Timing Luck](../../836-timing-luck/)**: varies the rebalance *day*
  at a fixed frequency (how much Sharpe is calendar luck). Study 940 varies the *frequency*
  itself and prices it against friction — the orthogonal axis of the same grid.
- **[Study 102 — Free-Rebalance](../../102-free-rebalance/)**: whether rebalancing a static
  allocation adds return. No cross-sectional signal, no turnover budget.
- **[Study 141 — Turnover-Anomaly](../../141-turnover-anomaly/)** and
  **[Study 821 — Turnover Volatility](../../821-turnover-volatility/)**: turnover as a
  *stock characteristic* on the right-hand side of a sort. Here turnover is the strategy's
  own trading, on the cost side of the ledger.
- **[Study 890 — Sector Risk-Parity](../../890-sector-risk-parity/)** and
  **[Study 903 — Sector-Neutral Low-Vol](../../903-sector-neutral-lowvol/)**: the same eleven
  SPDRs, but weighting schemes (equal-risk, low-vol) rather than a momentum cross-section
  timed at four speeds.
- **[Study 632 — Crypto Cross-Sectional Momentum](../../632-crypto-xs-momentum/)**: the same
  12-1 cross-sectional machinery on a different, far higher-turnover universe.

## Method lineage

- **HAC / Newey-West t-stat.** Newey & West (1987), *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix*, Econometrica —
  [`strategy.newey_west_t`](../turnover_budget/strategy.py) and
  [`quantlab.analytics.mean_tstat_hac`](../../../quantlab/analytics.py).
- **Return-difference (Sharpe comparison) t-stat.** Jobson & Korkie (1981), *Performance
  Hypothesis Testing with the Sharpe and Treynor Measures*, Journal of Finance —
  [`strategy.sharpe_diff_tstat`](../turnover_budget/strategy.py), used for the paired
  speed-vs-speed races.
- **Circular block bootstrap.** Politis & Romano (1992), *A Circular Block-Resampling
  Procedure for Stationary Data* — the fixed-length circular scheme actually implemented in
  [`strategy.bootstrap_sharpe_ci`](../turnover_budget/strategy.py) (21-day blocks, wrapped).
  Their 1994 *Stationary Bootstrap* (JASA) randomises the block length; we do **not** do
  that, and the CIs should be read as fixed-block. See also
  [`quantlab.stats.sharpe_ci_bootstrap`](../../../quantlab/stats.py).
- **Reproducibility stamp.** [`quantlab.repro`](../../../quantlab/repro.py) — the as-of slice
  and the content fingerprint quoted in [`docs/results.md`](results.md).

## Data sources

- **XLB, XLC, XLE, XLF, XLI, XLK, XLP, XLRE, XLU, XLV, XLY** (the eleven Select Sector
  SPDRs) and **BIL** (1-3 month T-bill, the cash leg), plus **SPY** as the cap-weight
  reference — daily **total-return** closes via `yfinance` (`auto_adjust=True`), cached in
  the shared desk cache. Total return matters here: sector yields run from ~0.6% (XLK) to
  ~3% (XLU), and a price-only panel would quietly tilt the momentum cross-section toward the
  low-yield names.
- **Staggered inception.** The nine original SPDRs list 1998-12-22; **XLRE** was carved out
  of financials in 2015 and **XLC** out of technology/consumer discretionary in 2018. The
  panel treats a sector as NaN (not investable) before its own inception, so breadth grows
  9 → 11 rather than being back-filled. The GICS committee's decision to create those two
  sectors is itself hindsight the 1999 investor did not have — named on the Signal axis.
- **As-of 2026-06-30.** The partial current month is dropped so the sample never creeps. The
  headline dollar-neutral book runs from 1999-12-23 (once the 12-1 window fills); the
  long-only cross-check runs from BIL's 2007-05-30 inception, because only that arm needs a
  live cash leg.
- **Not tape:** the cost per unit traded notional (5 bps base) and the short-leg borrow
  (40 bps/yr base). Both are assumptions and both are swept in
  [`docs/results.md`](results.md).
