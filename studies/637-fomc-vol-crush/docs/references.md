# References & literature map — Study 637 (FOMC Vol Crush)

## The claim under test

- **The folklore.** "The VIX collapses the afternoon of a Fed decision" — options desks call
  it the **event-vol crush**: a scheduled macro announcement carries an *event premium* in
  implied volatility that must, mechanically, come out of the surface once the announcement
  is public. Since 1994 the FOMC announces its decision the day the scheduled meeting ends,
  mid-afternoon ET (2:15 pm; 2:00 pm since 2013) — **before** the 4:15 pm ^VIX close — so the
  crush should be visible in the plain daily close-to-close ^VIX change.
- **The academic anchor.** Pástor & Veronesi (2013, *Political uncertainty and risk premia*,
  JFE) and **Amengual & Xiu (2018, *Resolution of policy uncertainty and sudden declines in
  volatility*, Journal of Econometrics)** — the latter documents that a large share of sudden
  downward volatility jumps occur on FOMC announcement afternoons: uncertainty *resolution*
  is a first-order driver of implied vol. Fernandez-Perez, Frijns & Tourani-Rad (2017, *When
  no news is good news — the decrease in investor fear after the FOMC announcement*, JEF)
  measure the intraday VIX drop directly.
- **The adjacent (distinct) result.** Lucca & Moench (2015, *The Pre-FOMC Announcement
  Drift*, JF) is about **equity returns before** the announcement — not about what implied
  vol does **on** the day. See the dedup map below.

## What we measure, and the honesty rails

- **ΔVIX on the decision-day close** — close minus previous close, in points and in logs.
  The statement (2:00/2:15 pm ET) predates the ^VIX close (4:15 pm ET) on every decision day,
  so the daily bar *contains* the resolution. Welch *t* for the split (Welch 1947); the
  events are single, non-overlapping days, and a **Newey-West (1987)** 5-lag *t* on the
  decision-day dummy regression is reported as the autocorrelation-robust cross-check.
- **Scheduled meetings only.** Inter-meeting/emergency actions (1998-10, 2001-01/04/09,
  2007-08, 2008-01, 2020-03-03/15…) are *excluded by construction* — they are surprises, the
  exact opposite of "set your watch to it". Known quirk kept for calendar consistency with
  the sibling studies: the scheduled 2020-03-18 date stays although the 2020-03-15 emergency
  cut pre-empted it.
- **Hit rate carries a Wilson (1927) interval**; the placebo is a 20-seed × 1,000-draw
  random-calendar null; the era split (2011-04-27) is *justified, not snooped* — the first
  post-meeting press conference, the structural change in how the decision is communicated.
- **Realized-vs-implied cross-check.** SPY (H−L)/prev-close on the same days: the decision
  day is *louder* than average in realized terms while implied collapses — the signature of
  event-premium expiry, not of a quiet afternoon.

## Why the tradable echo is graded separately

- SVXY (ProShares Short VIX Short-Term Futures) holds **VIX futures**, not spot VIX; the
  front of the curve **pre-prices** the scheduled crush (the event premium sits in the basis
  and decays into the meeting), so the ETP captures only the *surprise* component. Whaley
  (2013, *Trading volatility: at what cost?*) and Alexander & Korovilas (2013) on the
  mechanics and drag of VIX ETPs.
- **Survivorship, named:** SVXY is the *surviving* short-vol ETP. Its −1× twin **XIV was
  terminated after 2018-02-05** ("Volmageddon", an ~−95% day), and SVXY itself cut exposure
  to −0.5× on 2018-02-27 — the vehicle class carries documented gap-to-zero risk, and any
  short-vol ETP backtest is conditioned on the vehicle that lived.
- Costs are charged one-way × NAV per leg (5/10 bps; SVXY spreads are pennies on a ~$40-50
  NAV); the entry is the prior close — the FOMC calendar is public years in advance, so the
  scheduled entry involves zero look-ahead (the study's single documented execution
  convention).

## Data sources

- **^VIX daily OHLC**, **SPY daily raw OHLC** and **SVXY adjusted closes** — yfinance (no
  key), cached under `_cache/` (`fvc_vix.csv`, `fvc_spy.csv`, `fvc_svxy.csv`), 1994-01-03 →
  2026-06-30 (SVXY from 2011-10-04 inception). Cboe VIX index methodology:
  https://www.cboe.com/tradable_products/vix/
- **Scheduled FOMC decision dates 1994 → 2026**, hardcoded in
  [`data.py`](../fomc_vol_crush/data.py). Source: Federal Reserve historical FOMC calendars —
  https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm (and the historical
  materials pages per year). Same table as the sibling studies below.
- All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is NOT)

- [517-pre-fomc-drift](../517-pre-fomc-drift/) — the Lucca-Moench **equity-return drift
  before** the announcement. Returns, pre-day. This study: **implied vol, on the day**.
- [67-fed-drift](../67-fed-drift/) — the decayed post-publication version of that same
  return drift. Again returns, not vol.
- [135-fomc-cycle](../135-fomc-cycle/) — the **week-parity cycle** in equity returns across
  the whole inter-meeting period. Calendar-cycle returns, not the decision-day vol event.
- [322-fomc-blackout](../322-fomc-blackout/) — the **blackout window** before meetings.
  A pre-meeting information regime, not the announcement afternoon.
- [605-vix-settlement-day](../605-vix-settlement-day/) — ^VIX behavior on **derivative
  settlement mornings** (and it *controls FOMC out* as a confounder — this study is exactly
  the confounder it removes, promoted to the object of study).

None of the siblings test what the **VIX does on the decision day** — the vol-crush claim is
this study's own axis.
