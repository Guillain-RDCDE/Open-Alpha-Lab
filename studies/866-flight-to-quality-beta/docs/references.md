# References & literature map — Study 866 (Flight-to-Quality Beta)

## The claim under test

- **Flight-to-quality is a cross-asset fact.** In risk-off episodes investors sell equities
  and buy the safest bonds; long Treasuries rally as stocks fall. The literature on
  **flight-to-quality / flight-to-safety** (e.g. **Baur & Lucey (2010)**, *"Is Gold a Hedge or
  a Safe Haven?"*, Financial Review; **Baele, Bekaert, Inghelbrecht & Wei (2020)**,
  *"Flights-to-Safety"*, Review of Financial Studies) documents this negative stock–bond
  co-movement precisely in the left tail of the equity distribution. Our conditioner is exactly
  that regime: **down-SPY days**.
- **The pricing-of-insurance prediction.** If a stock reliably rises with the safe-haven bid
  when the market falls, it is a *good hedge* — and in a CAPM-of-insurance world investors
  overpay for hedges, so such names should carry a **lower** expected return. This is the same
  logic as **downside-risk pricing** (**Ang, Chen & Xing (2006)**, *"Downside Risk"*, Review of
  Financial Studies) and **betting-against-beta** (**Frazzini & Pedersen (2014)**, *"Betting
  Against Beta"*, Journal of Financial Economics) — a security that protects you when it hurts
  most is expensive and under-earns.
- **The specific test here.** For each name we estimate a **flight-to-quality beta**: the OLS
  slope of its daily return on the **TLT** (20+yr Treasury) return, computed **only on days
  when SPY is down**. We then (a) sort the cross-section and measure the forward return of a
  long-low-FTQ / short-high-FTQ book — the "pay-for-the-hedge" premium — with a Newey-West *t*,
  a permutation placebo, a two-era cut, and a costed timer; and (b) test the *other* half of the
  claim directly, comparing the two books' returns on the worst 5% of SPY days (crash
  protection). A seeded synthetic positive control proves the machinery is unbiased.

## What we measure, and the honesty rails

- **Conditional beta, no free model.** `beta_ftq_i = cov(r_i, r_TLT | r_SPY<0) / var(r_TLT |
  r_SPY<0)` over a trailing 252-day window, computed vectorised across the whole cross-section
  (one matrix contraction on the down-day sub-window).
- **Point-in-time sort, one documented lag.** The ranking signal is the FTQ beta **known at
  month-end `t−1`**; the book is held over month `t`. Zero look-ahead.
- **Robust inference.** Newey-West (HAC, Bartlett, 6-lag) *t* on the monthly long-short spread;
  a one-sample *t* and a pooled Welch *t* (low-FTQ book vs high-FTQ book) cross-check; a
  **1,000-permutation placebo** breaks the signal → forward-return link to see whether the
  cross-sectional tilt is a lucky alignment of the sort.
- **Both halves of the claim are graded.** The return-penalty half (the long-short) is the
  **Signal** axis; the crash-protection half (books' returns on the worst SPY days) is reported
  as a separate, informational third axis so the two are never conflated.
- **Survivorship is named on the Signal axis.** The universe is a **current-membership** set of
  ~50 liquid mega-caps, pulled through the `quantlab.universe` survivorship guard
  (`allow_survivorship_bias=True`, an explicit opt-in). Delisted / de-rated names are absent, so
  the magnitudes are an **upper bound**.
- **The timer is graded separately.** Costs are one-way × NAV on both legs, and the short book
  pays borrow — the honest test of whether the thin monthly premium survives friction.

## Shared method citations

- **Newey, W. & West, K. (1987)** — heteroskedasticity- and autocorrelation-consistent
  covariance (the HAC *t* used on the monthly spread series).
- **Wilson, E. B. (1927)** — score interval for a binomial share.
- **Bawa, V. & Lindenberg, E. (1977)** — mean–lower-partial-moment (downside) beta, the
  conditional-beta convention this study borrows and re-points at the Treasury return.

## Data sources

- **yfinance daily OHLC** (`auto_adjust=True`, total-return), 50 liquid US large-caps,
  2010-01-04 → 2026-06-30, cached under `_cache/` through `quantlab.universe`.
- **yfinance daily closes for TLT and SPY** (`auto_adjust=True`), same span, cached under
  `_cache/market_tlt_spy.parquet`.
- All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is NOT)

- [332-downside-beta](../../332-downside-beta/) — beta to the **equity market** on down days
  (Ang-Chen-Xing β⁻). This study conditions on the same down-market regime but measures
  co-movement with the **Treasury (TLT)** return — loading on the *safe haven*, not on the
  falling market itself.
- [238-betting-against-beta](../../238-betting-against-beta/) — the Frazzini-Pedersen tilt on
  **market** beta (low-beta out-earns high-beta). FTQ beta is a **cross-asset** (equity↔bond)
  loading measured only in sell-offs, not a market-beta rank.
- [246-defensive-sectors](../../246-defensive-sectors/) — a **sector label** (staples,
  utilities, health-care) as the defensive proxy. Here "defensive" is *revealed from the tape*
  by each name's own bond co-movement in sell-offs, not assigned by GICS.
- [69-safe-haven](../../69-safe-haven/) — whether a whole **asset class** (gold, bonds) hedges
  equities. This is a *within-equity cross-section* sort on how each individual stock loads on
  that safe-haven bid.

None of the siblings sort on a name's **own conditional beta to Treasuries on risk-off days** —
the flight-to-quality beta — which is this study's own axis.
