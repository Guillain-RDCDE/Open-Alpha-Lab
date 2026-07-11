# References & literature map — Study 644 (CPI-Day-Drift)

## The claim under test

- **The folklore.** "CPI morning is the market's biggest, most-watched day of the month" — a
  claim that hardened into conventional trading-desk wisdom over 2022-2024, as the Fed's rate
  path became explicitly data-dependent on the print. The claim has two halves that are often
  conflated: (1) CPI mornings *move the tape systematically* (in a bankable direction), and
  (2) CPI mornings are *louder* than an average day (larger realized moves, whichever way).
- **The academic anchor.** Savor & Wilson (2013, *How much do investors care about
  macroeconomic risk? Evidence from scheduled economic announcements*, JFQA) documents that the
  equity premium concentrates on scheduled macro-announcement days (CPI, NFP, FOMC pooled) —
  see sibling study [602-macro-announcement-premium](../../602-macro-announcement-premium/) for
  the direct rebuild. Faust, Rogers, Wang & Wright (2007, *The high-frequency response of
  exchange rates and interest rates to macroeconomic announcements*, JME) and Balduzzi,
  Elton & Green (2001, *Economic news and bond prices*, JFQA) both document that fixed-income
  markets react sharply and quickly to CPI/inflation surprises specifically — the mechanism
  behind this study's finding that TLT, not SPY, carries the certified reaction.
- **The adjacent (distinct) results.** Lucca & Moench (2015, *The Pre-FOMC Announcement
  Drift*, JF) and this desk's [637-fomc-vol-crush](../../637-fomc-vol-crush/) are about the
  **FOMC decision afternoon**, not the CPI morning — a different scheduled event, a different
  time of day, a different mechanism (Fed *decision* vs. Fed *input data*).

## What we measure, and the honesty rails

- **CPI-day return** — close-to-close, SPY and TLT. CPI prints at 8:30 am ET, before the 9:30 am
  open, so the daily bar fully contains the reaction (no intraday-vs-close mismatch to worry
  about, unlike an afternoon-release event). Welch *t* for the split (Welch 1947); a
  **Newey-West (1987)** 5-lag *t* on the CPI-day dummy regression is the autocorrelation-robust
  cross-check; a two-sided 20-seed × 1,000-draw random-calendar placebo (the direction claim has
  no a-priori sign).
- **Realized high-low range**, (H−L)/prev close, tested with a **one-sided** placebo — "louder"
  is an inherently non-negative claim, unlike "moves systematically" which could point either
  way.
- **Actual release dates, not a weekday-pattern reconstruction.** `CPI_DATES` is the identical,
  source-verified table already used by [602-macro-announcement-premium](../../602-macro-announcement-premium/)
  (BLS archived-news-release index cross-checked against the official `histreleasedates.pdf`;
  19/19 overlapping dates agree across the two sources). The known shutdown-driven gap (no
  Nov-2025 release) is carried through and named.
- **The regime split (2022-01-01) is justified, not snooped** — the Fed's 2021-12-15 FOMC
  (accelerated taper, dot plot signaling 2022 hikes) is the date the CPI print became a direct
  input to the hiking-cycle reaction function; chosen from the FOMC calendar, before looking at
  any CPI-day return.
- **The third-axis "biggest day of the month" claim is tested as a rate with a Wilson interval
  and a Welch t of the pre/post DIFFERENCE**, against the honest chance baseline (1/n for an
  n-session month, ≈ 4.8% for a typical month) — never eyeballed off a chart.

## Why the tradable read is graded separately

- SPY's CPI-day return carries **no certified gap** over an ordinary day (Welch *t* = −0.01), so
  a naive "own SPY only on CPI day" timer starts from a non-edge and goes negative the moment
  costs are charged (5 bps one-way × 2 legs).
- The one certified effect — TLT's elevated realized range — has **no directional tilt to bank**
  (TLT's own return gap is *t* = +0.67, uncertified): a bigger range with no net drift nets zero
  on average to a directionless long/short position. Harvesting pure "loudness" needs an
  options/volatility instrument (straddles, variance swaps), which this study does not test and
  does not have free daily data for.
- By the precedent of [637-fomc-vol-crush](../../637-fomc-vol-crush/) — where a real, certified
  implied-vol collapse on FOMC afternoons turned out to be untradable because VIX futures
  pre-price the scheduled event — the working assumption for any *tradable* volatility claim
  here should be equally skeptical until tested directly.

## Data sources

- **SPY daily raw OHLC + adjusted close**, **TLT daily raw OHLC + adjusted close** — yfinance
  (no key), cached under `_cache/` (`cdd_spy.csv`, `cdd_tlt.csv`), 1997-01-02 → 2026-06-30 (TLT
  from its 2002-07-30 inception).
- **Actual CPI (CPI-U) news-release dates 1997 → 2026**, hardcoded in
  [`data.py`](../cpi_day_drift/data.py). Sources: BLS archived news-release index
  (https://www.bls.gov/bls/news-release/cpi.htm) and the official *Historical Release Dates*
  table (https://www.bls.gov/bls/histreleasedates.pdf). Identical table to sibling study 602
  (see below).
- All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is NOT)

- [602-macro-announcement-premium](../../602-macro-announcement-premium/) — the **pooled**
  Savor-Wilson equity premium across FOMC + CPI + NFP announcement days together, and finds it
  traces almost entirely to the FOMC leg (ex-FOMC *t* = 0.58). This study isolates **CPI alone**
  and adds the bond-side (TLT) and realized-range legs that 602 doesn't test.
- [643-payrolls-day-effect](../../643-payrolls-day-effect/) — its direct sibling: the same
  question (return + range + event window + timer) asked of the **Nonfarm Payrolls** morning
  instead of CPI. Same protocol, different macro print, different release-date calendar.
- [637-fomc-vol-crush](../../637-fomc-vol-crush/) — the **FOMC decision afternoon** (2:00 pm ET,
  a policy *decision*), not a pre-market *data release*; tests implied vol (^VIX), not realized
  range, and finds the opposite tradability lesson (a real effect, but pre-priced away).

None of the siblings isolate the **CPI print's** own return and loudness signature the way this
study does — the CPI-day claim is this study's own axis.
