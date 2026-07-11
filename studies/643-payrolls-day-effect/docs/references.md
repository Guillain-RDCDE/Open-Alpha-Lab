# References & literature map — Study 643 (Payrolls-Day-Effect)

## The claim under test

- **The folklore.** "Payrolls Friday moves the market" — the Nonfarm Payrolls (Employment
  Situation) report, released by the BLS at 8:30 am ET on (usually) the first Friday of the
  month, is the single most-watched print on the US macro calendar: futures desks staff up,
  volatility markets price a jump, and financial media treat the number as a market-moving
  event in its own right. The folk version of the claim goes further than "it's loud" — it
  implies a **knowable, tradable direction** (a relief rally once the print clears, or a
  build-up of positioning ahead of it).
- **The academic anchor.** Andersen & Bollerslev (1998, *Deutschmark-Dollar Volatility:
  Intraday Activity Patterns, Macroeconomic Announcements, and Longer Run Dependencies*, JF)
  and Balduzzi, Elton & Green (2001, *Economic News and Bond Prices*, JFQA) document that
  scheduled macro announcements — employment reports prominently among them — produce sharp,
  short-lived jumps in realized volatility. Savor & Wilson (2013, *How Much Do Investors Care
  About Macroeconomic Risk? Evidence from Scheduled Economic Announcements*, JFQA) find a
  positive **average equity premium pooled across CPI/FOMC/employment announcement days** —
  the adjacent, broader claim tested (and found Weak, FOMC-driven) by sibling study 602.
- **The adjacent (distinct) results.** Study [385-jobless-claims-momentum](../385-jobless-claims-momentum/)
  tests **weekly initial-claims** as a *leading* indicator over months, not the NFP print
  itself. Study [602-macro-announcement-premium](../602-macro-announcement-premium/) pools
  **CPI + FOMC + NFP** into one bundled announcement-day premium. Neither isolates what SPY
  does specifically on the **NFP release morning** — this study's own axis.

## What we measure, and the honesty rails

- **SPY close-to-close return on the release day** — the print lands at 8:30 am ET, before
  the 9:30 am open, so the daily bar fully contains the reaction; no intraday-only effect is
  missed. Welch *t* for the split (Welch 1947); the events are single, non-overlapping days,
  and a **Newey-West (1987)** 5-lag *t* on the release-day dummy regression is reported as
  the autocorrelation-robust cross-check.
- **Actual release dates, not a weekday-pattern reconstruction.** 353 dates, 1997-01 →
  2026-06, sourced from the BLS archived-news-release index and cross-checked against the
  official `histreleasedates.pdf` — the identical table already used and documented by
  sibling study 602 (19/19 overlapping dates agree across the two sources). 344/353 fall on a
  Friday; every non-Friday exception (July-4th-week Thursdays, shutdown delays, the 2026-02-11
  schedule slip) is a real, sourced date, not folklore.
- **Hit rate carries a Wilson (1927) interval, benchmarked against the baseline** — SPY's
  already-positive long-run daily drift means a naive ">50%" comparison is misleading; we
  report the release-day hit rate against the **other-day** hit rate (54.1%), not a coin
  flip.
- **Two-sided placebo.** Unlike a study with a signed hypothesis (e.g. the FOMC vol crush,
  which specifically predicts a *drop*), "equities behave systematically" carries no a-priori
  sign, so the 20-seed × 1,000-draw random-calendar placebo tests **|mean| ≥ |observed|**,
  not a one-tailed drop or rally.
- **Realized-range cross-check.** SPY (H−L)/prev-close on the same days, mirroring the
  resolution-of-uncertainty cross-check that made study 637's FOMC vol-crush finding
  real-mechanical: a scheduled announcement morning can be objectively *louder* even when the
  *direction* of that loudness isn't statistically knowable.
- **No multiple-comparison correction claimed as a finding.** The event window scans seven
  offsets; one of them (the cumulative pre-release run-up) nominally clears *t* = 2. It is
  reported and flagged, explicitly, as an uncorrected exploratory hit — not folded into the
  Signal stamp.

## Why the tradable echo is graded separately

- The naive timer (long SPY on the release day only, entered at the prior close — the BLS
  calendar is public months ahead, so this is a zero-look-ahead scheduled entry) is charged
  one-way costs × NAV per leg (5/10 bps) on a 2-leg round trip, 12 events/year. Even the
  uncertified gross edge (+12.43 bps/event) does not clear a realistic cost load.
- SPY itself carries no survivorship (an index-tracking ETF, not a current-constituent
  basket); the timer strategy is single-vehicle and inherits none.

## Data sources

- **SPY daily raw OHLC + adjusted close** — yfinance (no key), cached under `_cache/`
  (`pde_spy.csv`), 1997-01-02 → 2026-06-30.
- **Actual NFP (Employment Situation) release dates 1997 → 2026**, hardcoded in
  [`data.py`](../payrolls_day_effect/data.py) — identical to sibling study 602's
  source-verified `NFP_DATES` table. Sources: BLS archived-news-release index
  (bls.gov/bls/news-release/empsit.htm) and the official BLS *Historical Release Dates*
  table (bls.gov/bls/histreleasedates.pdf).
- All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is NOT)

- [385-jobless-claims-momentum](../385-jobless-claims-momentum/) — **weekly initial claims**
  as a slow-moving, monthly-frequency *momentum/leading-indicator* signal over 1-12 month
  horizons. A different series (initial claims, not the NFP headline print) and a different
  clock (a multi-month trend, not a single release morning).
- [602-macro-announcement-premium](../602-macro-announcement-premium/) — the **pooled,
  generic** CPI + FOMC + NFP announcement-day bundle (923 sessions), which that study shows
  is entirely an **FOMC** effect once decomposed (ex-FOMC *t* = 0.58). This study isolates
  the **NFP leg alone** and asks the question 602's own third axis explicitly does not
  answer for NFP specifically.
- [637-fomc-vol-crush](../637-fomc-vol-crush/) — the **VIX** (not SPY) collapse on **FOMC**
  (not NFP) decision afternoons. Different index, different calendar, different mechanism
  (implied-vol event-premium expiry vs a realized-return/range test on a data release). The
  realized-range cross-check here deliberately mirrors 637's method to make the comparison
  legible: FOMC gets a certified, real vol crush; NFP gets a certified *loudness* bump but an
  uncertified direction.

None of the siblings test what **SPY does specifically on the NFP release morning** — the
payrolls-day claim is this study's own axis.
