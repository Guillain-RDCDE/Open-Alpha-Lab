# References & literature map — Study 606 (OPEC Announcement Effect)

## The claim under test

- **The folklore.** "OPEC meeting days move oil — volatility doubles on decision day and
  the post-decision move keeps going; trade the announcement." A staple of energy-desk
  lore, broker notes and financial-TV coverage since the 1980s, sharpened in the OPEC+
  era when ministerial meetings became near-monthly (2020-2022).
- **The academic record.** The event-study literature broadly finds *announcement-day
  action but decision-dependent, unstable signed effects*:
  - Guidi, Russell & Tarbert (2006), *The effect of OPEC policy decisions on oil and
    stock prices*, OPEC Review 30(1) — OPEC decisions move oil, asymmetrically across
    cut/increase.
  - Demirer & Kutan (2010), *The behavior of crude oil spot and futures prices around
    OPEC and SPR announcements*, Energy Economics 32(6) 1467-1476 — significant
    announcement-window effects, mostly for **cuts**.
  - Loutia, Mellios & Andriosopoulos (2016), *Do OPEC announcements influence oil
    prices?*, Energy Policy 90, 262-272 — 1991-2015 meeting-by-meeting event study;
    impact **evolves over time and by decision type**, differs WTI vs Brent.
  - Schmidbauer & Rösch (2012), *OPEC news announcements: Effects on oil price
    expectation and volatility*, Energy Economics 34(5) — announcement days carry a
    **volatility** effect over and above the mean effect.
  Our contribution is the desk treatment: one pre-registered meeting table 2000-2026,
  three tapes, the vol multiple with a CI (does it *double*?), HAC drift tests, and the
  tradability of the sign-continuation rule — with a planted-effect machinery control.

## The meeting table (the frozen event input)

- **Scope rule (fixed before looking at returns).** Every OPEC Conference meeting
  (Ordinary + Extraordinary) 2000-2016, the production-setting ministerial consultations
  (25 Jul 2001, 31 Jul 2003, Doha 20 Oct 2006), and every OPEC and non-OPEC Ministerial
  Meeting (ONOMM, the OPEC+ decision body) 2016-2026 — 107 decision days. JMMC
  monitoring committees and post-2023 "eight-country" subgroup calls are **excluded**
  (not the full decision body); note this is *conservative* for the vol claim, since
  some excluded events (e.g. the 3 Apr 2023 surprise voluntary cuts) moved oil hard.
- **Sources.** OPEC press-release archive — https://www.opec.org/press-releases.html
  (per-meeting pages, e.g. `/pr-detail/28-05-dec-2024.html`, 38th ONOMM) and the legacy
  archive `www.opec.org/opec_web/en/press_room/` (e.g. `1050.htm` 129th Conference,
  `1009.htm` 145th, `1006.htm` 146th); the OPEC Annual Statistical Bulletin meeting
  chronology (https://www.opec.org/assets/assetdb/asb-2025.pdf); Wikipedia
  world-oil-market chronologies for 2000-2016 cross-checks. Spot-verified dates are
  listed in the [`data.py`](../opec_announcement_effect/data.py) header comment.
- **Date convention.** The decision (press-release) day; the analysis maps every date to
  the first tradable session at-or-after it per asset, so weekend ONOMMs (e.g. the 41st,
  Sunday 7 Jun 2026) and the Thanksgiving-day 166th Conference (27 Nov 2014) land on the
  session that could actually react.

## Method notes

- **Vol inference.** Welch (1947) *t* on |returns| event-vs-baseline; a Brown-Forsythe
  (1974)-style spread test (Welch *t* on absolute deviations from group medians —
  robust to non-normality); a variance ratio; a **bootstrap CI on the vol multiple**
  (events i.i.d. — they are months apart; baseline in circular blocks of 10, Politis &
  Romano logic, to respect volatility clustering); and a 2,000-draw random-calendar
  placebo (Fisher randomization logic).
- **Drift inference.** Newey & West (1987) HAC *t* from a dummy regression of daily
  returns on the day-0..+k event window (lags k+5) — the Signal-axis statistic — plus
  the per-event one-sample *t* (events are months apart; the two overlapping April-2020
  windows are the documented exception).
- **Execution.** The decision lands intraday, hours before the settle; the continuation
  rule enters at the **day-0 close** (the single documented convention) with the
  enter-at-close(+1) full-lag variant reported beside it. Costs 2/5/10 bps one-way —
  CL front-month trades ~1-2 bps wide.
- **No survivorship** — front-month futures and a live ETF; no panel selection. USO's
  structural roll drag affects levels, not the event-vs-baseline *comparisons* used here.

## Data sources used here

- **yfinance** daily OHLC: `CL=F` (NYMEX WTI front month, 2000-08-23→2026-06-30),
  `BZ=F` (ICE Brent front month, 2007-07-30→), `USO` (United States Oil Fund ETF,
  total-return adjusted, 2006-04-10→), cached under `_cache/opec_{cl,bz,uso}.csv`.
  Headline numbers pinned in [`docs/results.md`](results.md), reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup frame)

- [313-geopolitical-shock](../313-geopolitical-shock/) — oil around **wars and crises**:
  unscheduled, exogenous shocks. This study is the opposite corner: **scheduled,
  endogenous policy decisions** with a known calendar.
- [226-crude-seasonality](../226-crude-seasonality/) — **calendar seasonality** in crude
  (month-of-year patterns). OPEC decisions are event-time, not calendar-time: the
  baseline here explicitly excises the meeting windows the seasonality study averages
  over.
- [602-macro-announcement-premium](../602-macro-announcement-premium/) and
  [605-vix-settlement-day](../605-vix-settlement-day/) — the structural siblings:
  scheduled-announcement days with elevated variance and a contested drift.
