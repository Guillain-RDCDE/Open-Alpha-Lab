# References & literature map — Study 652 (Index-Deletion-Bounce)

## The claim under test

- **Chen, Noronha & Singal (2004).** *The Price Response to S&P 500 Index Additions and
  Deletions: Evidence of Asymmetry and a New Explanation* — Journal of Finance 59(4),
  1901-1929. The anchor. CNS document that S&P 500 **deletions** get dumped by index funds
  into the effective date and then **reverse** — unlike the inclusion pop, which they found
  had become *permanent*. Their explanation: additions carry an information signal (quality
  screen by the index committee) that deletions do not, so the deletion price move is "pure"
  price pressure with nothing fundamental behind it, and pure price pressure should mean-revert
  once the forced flow ends. This asymmetry — inclusion sticks, deletion reverses — is the
  specific, falsifiable claim this study tests on a fresh 2012-2025 tape.
- **Harris & Gurel (1986)** and **Shleifer (1986)** established the addition-side price-pressure
  finding this study's sibling, [249-index-inclusion](../../249-index-inclusion/), tests
  directly; CNS's contribution is the **deletion mirror** and the **asymmetry** claim.
- **Petajisto (2011), *The Index Premium and Its Hidden Cost for Index Funds*** (JFE 102(3))
  — quantifies the "index premium" arbitrageurs extract from S&P 500 funds on rebalance days
  (both additions and deletions), and documents its **decay** as more capital chased the same
  trade through the 2000s and 2010s — the backdrop against which this study asks whether the
  deletion side of that premium is still alive in 2012-2025.

## What we measure, and the honesty rails

- **The basket.** 70 real S&P 500 deletions, effective 2012-12-11 -> 2025-09-22, restricted
  *by construction* to removals S&P Dow Jones Indices itself coded **"Market capitalization
  change"** — i.e. the company shrank out of the index, the classic CNS distress-deletion
  mechanism — excluding deletions caused by a merger, acquisition, spin-off or bankruptcy
  filing (those tickers vanish outright on the effective date; there is no "long the deleted
  name" trade to test). Source: the "Selected changes to the list of S&P 500 components" table
  on Wikipedia's *List of S&P 500 companies* page, itself sourced row-by-row to S&P Dow Jones
  Indices' own index-news announcement PDFs
  (spglobal.com/spdji/en/documents/index-news/announcements/...); the announce date is the date
  printed on that cited press release, the effective date is the table's own "Effective Date"
  column. The full hardcoded table lives in
  [`index_deletion_bounce/data.py`](../index_deletion_bounce/data.py).
- **The honesty rail that matters most here: 22 of the 70 deletions (31%) have NO usable tape
  left on Yahoo Finance at all**, verified against Yahoo's own chart endpoint, not just a
  `yfinance` retry. Every one of these is a name that later suffered an *unrelated*, later
  corporate death — bankruptcy (Frontier Communications/FTR, Chesapeake Energy/CHK, Windstream/
  WIN, Mallinckrodt/MNK), a take-private or acquisition years after the S&P 500 removal
  (Comerica/CMA, Gap/GPS-era ticker issues, Dish Network/DISH, Fortune Brands Home & Security/
  FBHS, RR Donnelley/RRD, Big Lots/BIG). When that later death happens, Yahoo drops the whole
  history, including the untouched days around **our** 2012-2025 event. This is a **real,
  directional survivorship bias**: the tape that survives is disproportionately the tape of
  companies that did *not* keep declining into oblivion, tilting the usable sample *toward*
  finding a rebound (the true population — including the names that kept sinking — would look
  worse, not better, than what the 48-event usable panel shows). Named on the Signal axis, not
  buried in Tradability, per house style.
- **Market-adjusted CAR.** Daily log return of the stock minus SPY's same-day log return —
  exactly the abnormal-return construction CNS use — cumulated over the event window
  [-5..+40] trading sessions around the effective date (offset 0). One-sample *t* per offset,
  plus a percentile bootstrap CI (event-level resampling — events are cross-sectional, so this
  is the natural resampling unit, not a time-series block bootstrap).
- **Execution / lag.** The single documented convention: a "long the deleted stock" timer
  enters at the close of the **effective date** itself (public days ahead via the S&P press
  release — zero look-ahead) and exits after a fixed 40-session hold; one round trip = 2 x
  one-way cost x NAV (5/10 bps). Because the return is already market-adjusted (stock minus
  SPY), the timer's "gross/net" figures are already an **excess-vs-excess** comparison, per
  house style — no separate cash benchmark needed.
- **Random-day placebo.** For each ticker with usable tape, draw a random anchor day from ITS
  OWN history (excluding a buffer around the true event) and run the identical CAR machinery;
  repeat 300 times. Tests whether any post-window CAR is special to the *deletion event
  specifically*, or just generic behavior of a name that was distressed enough to be removed.
- **Era split.** The 2012-2025 span cut at 2019-01-01 (bisects the sample) — CNS's own claim
  is that the deletion effect, unlike inclusion, does **not** decay; the split is our
  within-sample test of that claim.

## Data sources

- **Per-ticker deletion-window OHLC** and **SPY OHLC** — yfinance (no key), cached under
  `_cache/` (`idb_<TICKER>.csv`, `idb_spy.csv`), 2012-01-03 -> 2026-06-30.
- **The 70-row deletion calendar** (ticker, name, announce date, effective date), hardcoded in
  [`data.py`](../index_deletion_bounce/data.py). Source: Wikipedia *List of S&P 500 companies*,
  "Selected changes..." table, cross-referenced to the cited S&P Dow Jones Indices index-news
  PDFs for the announce date: https://en.wikipedia.org/wiki/List_of_S%26P_500_companies and
  https://www.spglobal.com/spdji/en/documents/index-news/ (per-event announcement PDFs).
- All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is NOT)

- [249-index-inclusion](../../249-index-inclusion/) — the **ADD side** of the exact same S&P
  500 mechanism: does the classic inclusion "pop" survive? (Verdict there: `WEAK`, driven
  entirely by the TSLA outlier — ex-TSLA the pop is statistically zero, and the give-back
  reversal is *absent*, the opposite of what the "sell on inclusion" folklore predicts.) This
  study is the mirror-image claim on the **DELETE side** — CNS's specific point was that the
  two sides are *asymmetric* (deletion persists where inclusion decays); our third axis tests
  that asymmetry directly, using 249's own finding as the inclusion-side data point.
- [320-russell-reconstitution](../../320-russell-reconstitution/) — the **whole-index ETF**
  (IWM) around the annual, mechanical, one-directional Russell reshuffle. Different index
  (Russell 2000, not S&P 500), different instrument (a diversified ETF, not single deleted
  names), different flow direction (that study's reconstitution buys/sells the *whole* small-
  cap universe symmetrically; this study isolates single names that dropped OUT of a
  large-cap index). No ticker or event overlaps between the two baskets.
- [250-reverse-split](../../250-reverse-split/) — a **different, if related, distress
  signature**: a reverse stock split (itself often a precursor to, or consequence of, being
  small enough to face S&P 500 removal) and its own "kiss of death" forward-return test.
  250's basket (17 reverse splits, 2009-2024) and this study's basket (70 S&P 500 deletions,
  2012-2025) are constructed independently and do not share a single hardcoded event, though
  a company could in principle appear in both tables at different points in its decline —
  250 tests the split itself, this study tests the index removal.

None of the siblings test the S&P 500 **deletion** event on its own single-name basket — the
dump-then-rebound claim is this study's own axis.
