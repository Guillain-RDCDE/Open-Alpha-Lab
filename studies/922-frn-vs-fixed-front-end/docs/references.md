# References & literature map — Study 922 (Floating-Rate Front End)

## The claim under test

- **The floating-rate pitch.** A Treasury floating-rate note pays a coupon that resets
  weekly off the 13-week bill auction plus a fixed spread (the *discount margin*), so its
  price barely moves when yields move. The sales case — repeated in every rate-hiking cycle
  since the instrument was introduced — is that a floater is *cash with a pickup*: it earns
  more than a bill ladder because of the spread, and vastly more than short fixed-coupon
  paper when rates rise, with none of the mark-to-market pain. The counter-claim, made just
  as loudly on the way down, is that the floater surrenders the term premium and the capital
  gain that duration hands you in a cutting cycle. This study asks the tape which of the two
  stories the 2014-2026 cycle actually paid for.
- **The steelman for testing it now.** 2022-2026 is the first complete hike-plateau-cut cycle
  in the lifetime of the Treasury FRN market (first auction: January 2014). Before it, the
  question could only be argued from theory.

## The instrument and its mechanics

- **U.S. Treasury Department (2013-14), *Floating Rate Notes: Final Rule* (31 CFR Part 356)**
  — the two-year FRN's coupon is the weekly high-rate of the 13-week bill auction plus a
  spread fixed at issue, accrued daily and paid quarterly. This is the source of the ~0
  duration and of the "discount margin" the study measures as ≈ 15 bps/yr.
- **Fabozzi & Mann, *The Handbook of Fixed Income Securities*** — the standard treatment of
  floating-rate note valuation: a floater's price risk is confined to the *spread* between
  reset dates, so its effective duration is a few weeks, not years.
- **Macaulay (1938) / Hicks (1939) duration arithmetic** — the arithmetic behind the whole
  race: a 1-3 year fixed fund loses ≈ *D × Δy* of price per unit of yield change. With
  *D* ≈ 1.85, a 4.9 pp move in the front end is worth roughly 9 pp of price, which is the
  magnitude our hiking window reproduces (realised +6.45 pp, the residual being the curve
  inversion).

## What the literature expects the race to show

- **Fama (1984), *The Information in the Term Structure*, Journal of Financial Economics**;
  **Fama & Bliss (1987), *The Information in Long-Maturity Forward Rates*, AER** — forward
  rates carry a time-varying term premium, so the fixed leg's compensation for duration is
  *not* constant: it can be negative for years at a stretch. An inverted curve is exactly the
  state in which the floating end out-yields the fixed end, which is what our plateau and
  cutting windows show.
- **Campbell & Shiller (1991), *Yield Spreads and Interest Rate Movements*, Review of
  Economic Studies** — short rates do not move as the expectations hypothesis says they
  should, which is why "the market has already priced the hikes, so the floater cannot win"
  is a weaker argument than it sounds.
- **Cochrane & Piazzesi (2005), *Bond Risk Premia*, AER** — bond excess returns are
  predictable and strongly time-varying; a study that reports one unconditional average for
  a duration decision is answering the wrong question, which is why the headline here is a
  regime **contrast** rather than a mean.
- **Duffee (2002), *Term Premia and Interest Rate Forecasts in Affine Models*, Journal of
  Finance** — the term premium at the very front of the curve is small and hard to
  distinguish from zero in samples of this length. Our *t* = 0.96 on USFR − SHY is the
  applied version of that statement.

## Why our *t*-statistics stay small (and honestly so)

- **One cycle is one observation.** The 2022-2023 tightening contributes 356 trading days,
  but they are not 356 independent draws: a single macro event drives the whole window. The
  HAC (Newey-West) standard errors and the 21-day block bootstrap are there precisely to
  stop the day count from flattering the evidence.
- **Launch-liquidity artefacts.** USFR and TFLO began trading in February 2014 with tiny
  volumes; their 2014-2015 quoted closes contain round trips of 400+ bps that no holder of
  the underlying notes experienced. We report the whole tape and cut it at 2018 rather than
  quietly starting the sample where it flatters the result.

## Related desk studies (dedup)

- **[Study 921 — Bill Ladder vs ETF](../../921-bill-ladder-vs-etf/)**: the *fee* question
  inside the bill sleeve — a home-made ladder against the ETF that charges for it. Study 922
  holds the ETFs fixed and asks a *duration and coupon-type* question instead: floating vs
  fixed, across the cycle.
- **Study 925 — Front-End Trend** (`925-short-rate-momentum-switch`, a sibling candidate in
  this batch — **not built at the time of writing, so this entry is deliberately unlinked**):
  would *trade* the
  front end, switching duration on a short-rate trend signal. Study 922 deliberately trades
  nothing — it races buy-and-hold sleeves and conditions on the rate direction only to
  describe, never to time. The regime label here is a descriptive cut, not a rule.
- **[Study 885 — Ultra-Short Credit Pickup](../../885-ultra-short-credit-pickup/)**: the
  same "cash with a pickup" pitch sourced from *credit* (JPST/ICSH/MINT) rather than from a
  floating Treasury coupon — a spread-risk story, not a duration one.
- **[Study 826 — Treasury Duration BAB](../../826-treasury-duration-bab/)**: betting against
  beta *along* the Treasury curve, a leveraged cross-maturity factor. Study 922 is an
  unlevered, four-fund sleeve choice at the very front end.
- **[Study 625 — Starting Yield, Bond Decade](../../625-starting-yield-bond-decade/)** and
  **[Study 892 — Corporate Bond Ladder](../../892-corporate-bond-ladder/)**: the
  starting-yield identity and the ladder-vs-fund question in *longer* fixed-coupon paper.
- **[Study 16 — Storm SHY](../../16-storm-shy/)**: SHY as a crisis hedge (equity drawdowns),
  a flight-to-quality question rather than a rate-cycle one.

## Method lineage

- **HAC / Newey-West.** Newey & West (1987), *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix*, Econometrica —
  [`strategy.newey_west_t`](../frn_front/strategy.py),
  [`strategy.hac_ols`](../frn_front/strategy.py) (the regime contrast), and
  [`quantlab.analytics.mean_tstat_hac`](../../../quantlab/analytics.py).
- **Circular block bootstrap.** Politis & Romano (1994), *The Stationary Bootstrap*, JASA —
  [`strategy.block_bootstrap_ci`](../frn_front/strategy.py).
- **Reproducibility stamp.** [`quantlab.repro`](../../../quantlab/repro.py) — the as-of slice
  and content fingerprint printed above every headline table.

## Data sources

- **USFR** (WisdomTree Floating Rate Treasury), **TFLO** (iShares Treasury Floating Rate
  Bond), **BIL** (SPDR 1-3 Month T-Bill), **SHY** (iShares 1-3 Year Treasury Bond) — daily
  **total-return** closes via `yfinance` (`auto_adjust=True`). Total return is not optional
  here: these vehicles distribute essentially all of their return as monthly income, so a
  price-only comparison would measure nothing.
- **^IRX** — the CBOE 13-week Treasury bill **discount rate index**. This is a *price-only
  yield series*, not an investable total return; it is used for the regime label and, as a
  declared **PROXY**, for a daily cash accrual (`(irx_{t-1}/100)/252`), whose 252-vs-360 and
  discount-vs-investment-rate approximations are swept in `strategy.cash_proxy_sweep`.
- **Expense ratios** (USFR 0.15%, TFLO 0.15%, BIL 0.1356%, SHY 0.15% at build time) — quoted
  as **context only**; they are already inside the total-return closes and are never added
  to the arithmetic.
- **Rate-cycle windows** — a hardcoded **ASSUMPTION** (`strategy.CYCLE_WINDOWS`) reflecting
  Fed history: hiking 2022-03 → 2023-07, plateau 2023-08 → 2024-08, cutting 2024-09 → onward.
  The mechanical ^IRX-direction regime is reported alongside it, with window and dead band
  swept, so no conclusion rests on the hand-drawn calendar.
- **As-of 2026-06-30.** The partial current month is dropped so the sample never creeps.
