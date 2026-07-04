# References & literature map — Study 602 (Macro-Announcement-Day Premium)

## The claim under test

- **The seminal paper.** Pavol Savor & Mungo Wilson, *How Much Do Investors Care About
  Macroeconomic Risk? Evidence from Scheduled Economic Announcements* (2013, **Journal of
  Financial and Quantitative Analysis** 48(2), 343–375). On 1958–2009 US data they find average
  **excess equity returns of ~11.4 bps on scheduled announcement days** (CPI/PPI, employment,
  FOMC) versus **~1.1 bps on non-announcement days** — i.e. a large share of the equity premium
  concentrates on a small set of pre-scheduled macro-news days. They also report higher Treasury
  returns on announcement days and interpret the premium as compensation for macro risk borne
  when the news lands.
- **The mechanism family.** Lucca & Moench (2015, *The Pre-FOMC Announcement Drift*, JF) isolate
  the FOMC leg; Cieslak, Morse & Vissing-Jørgensen (2019, *Stock Returns over the FOMC Cycle*,
  JF) find the premium spreads over the even weeks of the FOMC cycle; Ai & Bansal (2018,
  *Risk Preferences and the Macroeconomic Announcement Premium*, ECMA) supply the theory —
  announcement premia arise under preferences with aversion to (Knightian) news risk.
- **Post-publication decay.** McLean & Pontiff (2016, JF) and the FOMC-drift literature
  (e.g. Kurov, Wolfe & Gilbert 2021) document announcement-day/drift premia weakening after
  publication — exactly what this tape shows post-2017.

## What we measure, and the construction of the calendar

- **Pooled A-day premium.** SPY daily **total-return** close-to-close returns on announcement
  days (CPI + NFP + FOMC union) vs all other days, Welch *t* (group split), plus a same-density
  random-calendar placebo (20,000 draws). CPI/NFP land at 08:30 ET (before the open) and the
  FOMC statement at ~14:00 ET (before the close), so the release-day close-to-close return is
  the session that prices the news; holiday releases (e.g. an NFP on Good Friday) map to the
  next session (10 cases, reported).
- **The calendar is actual release dates, not a pattern.** FOMC: Federal Reserve historical
  calendars (scheduled decisions only, no emergency actions —
  https://www.federalreserve.gov/monetarypolicy/fomc_historical_year.htm), same hardcoded table
  as study [517](../517-pre-fomc-drift/). CPI + Employment Situation: the BLS archived-news-
  release indexes (https://www.bls.gov/bls/news-release/cpi.htm,
  https://www.bls.gov/bls/news-release/empsit.htm), whose archive filenames carry the release
  date, cross-checked against the official BLS *Historical Release Dates* table
  (https://www.bls.gov/bls/histreleasedates.pdf) — 19/19 overlapping 1999-2000 dates agree.
  Construction, shutdown gaps and the one corrected filename quirk are documented in
  [`data.py`](../macro_announcement_premium/data.py); spot-checks are printed by
  [`examples/verify.py`](../examples/verify.py).
- **No look-ahead.** All three schedules are published in advance (Fed: a year ahead; BLS:
  months ahead), so the A-day tag — and the overlay's prior-close entry — use only ex-ante
  information. One execution lag: enter the close before the A-day, exit the A-day close.

## Why Welch + a placebo, and the honesty rails

- **Welch (1947)** for the unequal-variance group split (announcement days are ~40% more
  volatile); daily close-to-close returns carry negligible serial correlation at this horizon,
  and the **random-calendar placebo** (Fisher randomization logic; Efron & Tibshirani 1993) is
  the sharper null for "could any 923 days have looked this special?".
- **Costs against the alpha.** The A-day overlay round-trips ~31×/yr; we charge one-way costs ×
  NAV on both legs (Frazzini, Israel & Moskowitz 2018, *Trading Costs*, on the paper-vs-net
  gap). Raw total-return arithmetic — no T-bill credit on idle cash, an omission that penalises
  the overlay, never flatters it.
- **Synthetic control** with a planted A-day edge, seed-averaged over 100 seeds (the desk's
  ≥ 20-seed rule): the null must not manufacture significance; the planted edge must light up.
  Machinery proof only — never market evidence.

## Data sources used here

- **yfinance** daily adjusted (total-return) closes for SPY and TLT, 1996-12 → 2026-06, cached
  under `_cache/map_prices.csv`. SPY/TLT are broad index ETFs — survivorship-clean vehicles.
- **Hardcoded release calendars** (236 FOMC + 353 CPI + 359 NFP dates) in
  [`data.py`](../macro_announcement_premium/data.py) with per-list source comments.
- All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is *not*)

- [517-pre-fomc-drift](../517-pre-fomc-drift/) — the **pre-FOMC drift only** (Lucca-Moench's
  single session before/into the statement). Found Real-but-decayed (post-2012 *t* ≈ 0.3).
- [67-fed-drift](../67-fed-drift/) — the folklore "Fed drift" teardown, FOMC-window returns.
- [135-fomc-cycle](../135-fomc-cycle/) — Cieslak-Morse-Vissing-Jørgensen **FOMC-cycle weeks**
  (the bi-weekly cycle pattern, not announcement days).
- **This study is the pooled Savor-Wilson claim**: the premium on the *union* of scheduled
  CPI + NFP + FOMC release days — the version in which macro-news risk, not the Fed alone, is
  supposed to earn the equity premium. The third axis tests exactly whether the pooled framing
  adds anything beyond its FOMC sibling (answer on this tape: **no** — CPI/NFP-only *t* = 0.58).
- Structural cousins: [515-earnings-announcement-premium](../515-earnings-announcement-premium/)
  (firm-level scheduled-news premium), [516-dividend-month-premium](../516-dividend-month-premium/)
  (scheduled-payment calendar premium).
