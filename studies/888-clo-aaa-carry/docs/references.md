# References & literature map — Study 888 (CLO AAA Carry)

## The claim under test

- **The structure.** A **collateralised loan obligation (CLO)** pools ~150-250 senior
  secured leveraged loans and issues tranches in a payment waterfall: the **AAA** tranche
  is paid first and absorbs losses last, protected by a thick cushion (~35-40%) of
  subordinate tranches and equity. Historical **realized** impairment on AAA CLO tranches
  is ~nil across cycles (Moody's / S&P structured-finance default studies report zero
  principal losses on originally-AAA US CLO tranches through 2008-09 and 2020).
- **The pitch.** Because the tranche is **floating-rate** (coupon = SOFR + a discount
  margin), it carries almost no *duration*; because it is **senior**, almost no *default*
  risk. Yet it yields a spread over cash and over same-rated (AAA/AA) corporate bonds —
  the "**structural complexity premium**": a securitisation that few asset managers are
  staffed to underwrite, plus a liquidity/novelty premium on a young ETF wrapper.
- **The instruments.** **JAAA** (Janus Henderson AAA CLO ETF, inception 2020-10-16) and
  **ICLO** (Invesco AAA CLO Floating Rate Note ETF, inception 2022-12-08) hold portfolios
  of primarily AAA-rated CLO tranches. See the fund pages: janushenderson.com (JAAA) and
  invesco.com (ICLO). Expense ratios ~0.20-0.26%/yr, already inside the total-return NAV.

## What we measure, and the honesty rails

- **Everything excess-of-cash.** Each leg's daily total return minus **BIL** (SPDR 1-3m
  T-bill ETF), so the comparison is an **excess-vs-excess Sharpe race**, not a raw
  total-return contest. The AAA-CLO carry is precisely the excess-of-cash spread.
- **Honest benchmarks.** **LQD** (iShares IG corporates, ~8y duration) is the "same rating
  bucket, different risk" leg; **IEF** (7-10y Treasuries) is pure duration; **BKLN**
  (Invesco Senior Loan ETF) is the *un-tranched, below-IG* collateral — the sharp control
  for "is it just leveraged-loan beta?"
- **Robust inference.** Newey-West (HAC, Bartlett, 10-lag) *t* on the daily excess
  (`quantlab.analytics.mean_tstat_hac`); a **circular-block bootstrap** 95% CI on the
  annualised Sharpe (`quantlab.stats.sharpe_ci_bootstrap`); a ZIRP-vs-high-rate era cut; a
  costed buy-and-hold harvest and a long/short isolation trade; a seeded synthetic control.
- **Short history — named on the Signal axis.** JAAA has ~5.7y of tape, ICLO ~3.5y; the
  sample spans one rate cycle but **no CLO credit-stress event** (the March-2020 AAA-CLO
  mark-down predates JAAA). Every realized Sharpe is therefore an **upper bound**.

## Shared method citations

- **Newey, W. & West, K. (1987)** — heteroskedasticity- and autocorrelation-consistent
  covariance (the HAC *t* used on the daily excess series).
- **Politis, D. & Romano, J. (1994)** — the stationary/circular block bootstrap behind the
  Sharpe confidence interval (`quantlab.stats.sharpe_ci_bootstrap`).
- **Lo, A. (2002)**, *"The Statistics of Sharpe Ratios"*, Financial Analysts Journal — the
  standard-error framing for an annualised Sharpe on autocorrelated returns.

## Data sources

- **yfinance** daily total-return closes (`auto_adjust=True`) for **JAAA, ICLO, LQD, IEF,
  BKLN, BIL**, 2020-01-02 → 2026-06-30, cached under this study's own `_cache/`.
- All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is NOT)

- [614-clo-equity-yield](../../614-clo-equity-yield/) — the **first-loss EQUITY tranche**
  at the *bottom* of the same CLO stack (the risky residual). This study is the **senior
  AAA tranche** at the *top* — the opposite end of the waterfall, a carry vs a levered bet.
- [340-bank-loans](../../340-bank-loans/) — **un-tranched** senior *leveraged loans* (the
  BKLN collateral itself). Here BKLN is the *control*: the AAA slice's whole point is to
  deliver a better risk-adjusted return than the raw loan pool it is carved from.
- [796-corporate-bond-low-risk](../../796-corporate-bond-low-risk/) — a low-risk anomaly
  *within* the corporate-bond cross-section, not the securitised AAA-CLO wrapper.
- [885-ultra-short-credit-pickup](../../885-ultra-short-credit-pickup/) — the ultra-short
  *corporate* credit pickup over T-bills (a duration-free cash-plus play), a cousin at the
  short end but on unsecuritised corporate paper, not CLO tranches.

None of the siblings test the **senior AAA CLO tranche's excess-of-cash carry** against the
un-tranched collateral and same-rated duration — this study's own axis.
