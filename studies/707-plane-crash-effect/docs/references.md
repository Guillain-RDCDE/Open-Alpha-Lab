# References & literature map — Study 707 (Plane-Crash-Effect)

## The claim under test

- **The academic anchor.** Kaplanski & Levy (2010, *Journal of Financial and
  Quantitative Analysis*, "Sentiment and Stock Prices: The Case of Aviation
  Disasters") find that major aviation disasters are followed by a small, statistically
  detectable **negative** abnormal return in broad US stock indices over the following
  days, consistent with a mood/dread-driven sentiment shock (Kahneman & Tversky-style
  affect heuristic) rather than any cash-flow-relevant news for the average listed
  firm — and that the effect is stronger for **more severe** disasters (fatalities) and
  partially reverses. The mechanism: a vivid, media-saturated tragedy transiently sours
  investor mood, which the behavioral-finance sentiment literature (Hirshleifer &
  Shumway 2003, sunshine and stock returns; Kamstra, Kramer & Levi 2003, SAD and stock
  returns) has repeatedly linked to next-day return effects far from the underlying
  news' fundamental relevance.
- **The airline-specific corollary.** Believers extend the claim naturally: if a crash
  moves the *whole market* through pure sentiment, airline stocks — the sector with
  actual fundamental exposure (fleet-grounding costs, litigation, demand shock,
  reputational damage) — should fall **harder**, i.e. sentiment plus fundamentals.
- **What this study does differently from the original.** Kaplanski & Levy's sample
  runs 1950–2005 and screens a *global* accident register by fatality count; ours is a
  hand-curated 2000–2025 table of only the most famous, front-page-of-every-outlet
  disasters, tested against **modern, tradable** instruments (SPY and a 4-carrier US
  airline basket) rather than a broad academic index — the honest question for this
  desk is not "did this effect exist in the mid-20th century" but "is it there, and
  tradable, on the instruments a reader could actually buy today". A null result here
  does not contradict Kaplanski & Levy's original finding; it tests whether a
  *contemporary, tradable* version of it survives.

## What we measure, and the honesty rails

- **Crash-day (day 0) abnormal SPY return** — a constant-mean market model (Brown &
  Warner 1985): the "normal" return is the sample mean, the abnormal return is the
  demeaned daily return. One-sample *t* across the 36 independent, non-overlapping
  event dates (the planned primary; events are far apart in time, so no HAC correction
  is needed the way a daily-panel regression would).
- **Event window [−1..+5]** with each offset's own one-sample *t*, tested honestly as a
  **multiple-comparison** exercise — 7 offsets, so roughly 1-in-7 crossing the
  conventional 5% |*t*| ≥ 2 bar by pure chance is expected and is called out as such
  when it happens (offset +2 in this run).
- **Reversal** — the cumulative abnormal return over [+1..+5], testing the folklore's
  "temporary" half directly rather than eyeballing a chart.
- **The airline extra-drop** — a *paired* same-date (airline basket − SPY) day-0
  difference, one-sample *t*'d across events. Pairing removes the common market-wide
  move on the day, isolating the airline-specific incremental reaction the claim
  actually predicts.
- **Hit rate carries a Wilson (1927) interval**; the placebo is a 20-seed × 1,000-draw
  random-calendar null (the same falsification design as `313-geopolitical-shock`).
- **Coverage named, not hidden.** The AAL/DAL/UAL/LUV basket has full 4-carrier
  coverage only from 2008 onward (each carrier's current corporate entity has a
  different trading-start date after bankruptcy/merger); the earliest events fall back
  to 1–3 carriers, and a full-coverage-only subsample (n=27) is reported alongside the
  full n=36 so the reader can see the answer does not hinge on the thin early years.

## Why the "buy the dip" timer is graded separately, and the ethical framing

- The tradable overlay — enter SPY at the crash-day close, hold a few sessions — is
  tested purely as a **statistical falsification exercise**: if the sentiment-dip claim
  were real and tradable, this overlay should show a positive, cost-surviving edge over
  simply holding the index. It does not (every horizon tested underperforms the
  unconditional baseline). **The obvious ethical caveat, stated plainly and kept
  clinical:** this measures whether a pattern exists in public market-price data; it is
  not, and should not be read as, encouragement to structure trades around human
  tragedy. The honest finding here is also the ethically comfortable one — there is
  nothing to bank.
- Costs are one-way × NAV per leg (5/10 bps), one round trip per event, long-only (no
  borrow, no shorting the grief). The entry convention — the crash's calendar date
  snapped to the first tradable NYSE session — is the study's single documented
  execution lag; see `docs/results.md`.

## Data sources

- **SPY** and **AAL / DAL / UAL / LUV** daily total-return closes (`auto_adjust=True`)
  — yfinance (no key), cached under `_cache/` (`pce_spy.csv`, `pce_aal.csv`,
  `pce_dal.csv`, `pce_ual.csv`, `pce_luv.csv`), 2000-01-03 → 2026-06-30 (each airline
  ticker from its current-entity trading start).
- **36 hardcoded major commercial-aviation disasters, 2000 → 2025**, in
  [`plane_crash_effect/data.py`](../plane_crash_effect/data.py). No free,
  machine-readable "major aviation disaster index" exists (unlike, say, the FOMC
  calendar or a GPR index), so this is a hand-built table of the crashes any reasonable
  person would call front-page news — cross-referenced against public aviation-safety
  reporting (ASN, NTSB/ICAO accident summaries, contemporary wire coverage).
- All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is NOT)

- [313-geopolitical-shock](../313-geopolitical-shock/) — wars, invasions and terror
  attacks, including **9/11 and MH17** (both aviation events, but driven by an act of
  war/terror, not an accident) — this study's table **excludes both** to avoid
  double-counting the same market day under two different folklore claims. 313 tests
  geopolitical-shock sentiment; this study tests **accidental**-disaster sentiment.
- [300-sports-sentiment](../300-sports-sentiment/) — the Edmans, Garcia & Norli (2007)
  "loss effect": a national soccer-team elimination souring the *home* market's mood.
  Same behavioral-finance family (a vivid, emotionally salient, non-fundamental event
  moving sentiment) but an entirely different trigger, geography and mechanism.
- [279-geomagnetic-storms](../279-geomagnetic-storms/) — the Krivelyova & Robotti
  (2003) mood channel via **geomagnetic activity**, not news at all — a physiological/
  biological mood hypothesis rather than a media-driven dread response. Same "does
  mood move markets" research question, entirely different (non-newsworthy) trigger.

None of the siblings test what **a major air crash does to the market and to
airline stocks specifically** — that is this study's own axis.
