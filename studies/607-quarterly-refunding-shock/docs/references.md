# References & literature map — Study 607 (Quarterly Refunding Shock)

## The claim under test

- **The folklore.** After July–November 2023, macro desks began treating the Treasury's
  **Quarterly Refunding Announcement** (QRA) as a first-tier market event: the July 31,
  2023 borrowing-estimate surprise (+$274B) and the Aug 2 size increases were blamed for
  the autumn-2023 long-end selloff, and the Oct 30 / Nov 1 downshift was credited with
  ending it. The strong form of the claim — *QRA day now moves the long end the way an
  FOMC or CPI day does* — is what we test.
- **The institutional mechanics.** Treasury announces marketable-borrowing estimates on
  the Monday (15:00 ET) and the refunding statement — coupon auction sizes for the
  quarter — on the Wednesday (08:30 ET) of the mid-quarter refunding week (early
  Feb/May/Aug/Nov). Statement archive:
  [home.treasury.gov/policy-issues/financing-the-government/quarterly-refunding](https://home.treasury.gov/policy-issues/financing-the-government/quarterly-refunding);
  press releases: [home.treasury.gov/news/press-releases](https://home.treasury.gov/news/press-releases).
- **The 2023-era literature.** Stephen Miran & Nouriel Roubini, *ATI: Activist Treasury
  Issuance and the Tug-of-War Over Monetary Policy* (Hudson Bay Capital, July 2024) — the
  paper that canonised the "Treasury issuance steers the long end" narrative. The
  Treasury Borrowing Advisory Committee (TBAC) minutes and presenting charts accompany
  every statement. Hamilton (2024, blogged commentary) and multiple Fed notes dispute the
  magnitude. Our study asks the narrow, measurable question: does the *announcement day
  itself* carry an outsized move?

## How the event table was built (fetch once → hardcode with source)

- **QRA dates 2000→2026 (106).** The securities sold in the mid-quarter refunding (the
  10-Year note auctioned in Feb/May/Aug/Nov; in the five 2000–2002 quarters with no new
  10Y, the refunding 5-Year/10Y-reopening pair) are *officially announced via the
  refunding statement*, so their `announcementDate` in the TreasuryDirect auction-query
  API **is** the QRA date. Source:
  [treasurydirect.gov/TA_WS/securities/search?type=Note&format=json](https://www.treasurydirect.gov/TA_WS/securities/search?type=Note&format=json)
  (fetched 2026-07-03; derivation kept as
  [`data.derive_qra_dates_from_treasurydirect`](../quarterly_refunding_shock/data.py)).
  Sanity: all 106 dates are Wednesdays; the storied dates match the press archive
  (2023-08-02, 2023-11-01, 2024-01-31, 2024-10-30, and 2001-10-31 — the 30Y-suspension
  statement).
- **FOMC statement days** (the decontamination table): Federal Reserve historical
  calendars,
  [federalreserve.gov/monetarypolicy/fomc_historical_year.htm](https://www.federalreserve.gov/monetarypolicy/fomc_historical_year.htm)
  — same hardcoded table as desk studies 517/602. **35% of QRA days are FOMC statement
  days** (both calendars pick early-quarter Wednesdays; 2023-11-01 included), which is
  why the primary test runs on FOMC-clean QRA days.
- **The 2023 borrowing-estimate episodes** (hardcoded narrative table): Treasury press
  releases of 2023-07-31 ($1,007B vs $733B flagged in May) and 2023-10-30 ($776B vs
  $852B flagged in July).

## Methods

- **Event-day volatility test.** Welch (1947) *t* on |Δy| — event days sit ~63 sessions
  apart, hence serially uncorrelated; no overlapping windows, nothing for Newey-West to
  fix. A 2,000-draw random-calendar placebo (same size, drawn from the baseline days)
  re-asks the question without distributional assumptions — any random baseline is
  averaged over far more than the desk's ≥20-seed floor.
- **Vol-normalised era comparison.** |Δy| / trailing 60-day mean |Δy| (ending the day
  before), so 2023 QRA days are judged against 2023's own noise, not 2004's. Splits at
  2023-01-01 are *the claim's own* split (not snooped): the folklore names 2023 as the
  regime break.
- **The jobs-report collision.** QRA = early-month Wednesday; the BLS Employment
  Situation = first Friday, 08:30 ET ([bls.gov/schedule/news_release/empsit.htm](https://www.bls.gov/schedule/news_release/empsit.htm))
  — i.e. day+2 of the QRA window. Fleming & Remolona (1999, *Price Formation and
  Liquidity in the U.S. Treasury Market: The Response to Public Information*, JF) rank
  the employment report the single largest scheduled mover of Treasury yields — exactly
  what our day+2 diagnostic finds and removes.
- **Execution honesty.** The statement is public at 08:30 ET; the first honest fill is
  the day-0 close (the ONE documented lag). TLT legs pay one-way costs × 2 legs per
  event; TLT is total-return (auto-adjusted).

## Data sources used here

- **TreasuryDirect TA_WS auction records** (announcement dates; the QRA calendar).
- **yfinance**: ^TNX (CBOE 10Y yield index, Yahoo serves percent), ^TYX (30Y), TLT
  total-return closes, cached under `_cache/qrs_*.csv`. All headline numbers pinned in
  [`docs/results.md`](results.md) and reproduced by [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map)

- [603-treasury-auction-concession](../603-treasury-auction-concession/) — the sibling on
  the **execution** side: yields cheapen into the 10Y/30Y **auctions** themselves (the
  supply *hits* the tape). This study is the **announcement** side: the day Treasury
  *reveals the plan*. Distinct events, distinct dates (QRA Wednesdays precede the
  refunding auctions by ~a week), and opposite verdicts — the auctions carry a real
  concession; the announcement day carries nothing.
- [602-macro-announcement-premium](../602-macro-announcement-premium/) — FOMC/CPI/NFP
  announcement-day equity premium; source of the shared FOMC calendar, and the reason we
  knew to decontaminate the QRA Wednesdays.
- [605-vix-settlement-day](../605-vix-settlement-day/) — another "does the calendar day
  itself carry a shock?" study on this bench.
