# References & literature map — Study 733 (Kentucky-Derby-Effect)

## The claim under test

- **The folklore, two flavours.** (1) *A market seasonal.* The Kentucky Derby — the "Run
  for the Roses" — is run on the **first Saturday in May**, which sits exactly on the
  "Sell in May and go away" calendar boundary; almanac-style seasonality lore treats
  early May as a market inflection. (2) *A gambling name.* The one US-listed stock with a
  direct, marquee exposure to the race is **Churchill Downs Incorporated (`CHDN`)**, which
  owns and operates Churchill Downs racetrack and the Derby itself — so if any single
  equity should "pop" around the first Saturday in May, it is CHDN. Both are
  financial-media / retail-forum folklore, not a peer-reviewed anomaly.
- **The academic anchor for the mechanism (a different trigger).** The idea that mood- and
  attention-events move asset prices is real and well-identified: Edmans, García & Norli
  (2007, *Sports Sentiment and Stock Returns*, Journal of Finance) find national markets
  fall after World Cup soccer **elimination**; Kaplanski & Levy (2010, *Sentiment and
  Stock Prices: The Case of Aviation Disasters*, JFQA) find a broad market dip after major
  air crashes. Both isolate the mood channel off a *negative* shock in an event with deep
  national reach. The Kentucky Derby borrows the "an event moves sentiment" premise and
  swaps in a *positive*, single-day, spectator sporting event whose direct listed exposure
  is one mid-cap company.
- **The "Sell in May" strand.** Bouman & Jacobsen (2002, *The Halloween Indicator, "Sell
  in May and Go Away"*, American Economic Review) document lower May–October equity
  returns across many markets; Jacobsen & Zhang (2018, *The Halloween Indicator…
  Everywhere and All the Time*) revisit it. The Derby's first-Saturday-in-May date makes
  it a natural event-resolution probe of whether that boundary is a tradable turn — this
  study is the desk's Derby-anchored slice of that question.
- **What nobody has published.** There is no peer-reviewed study of a Kentucky Derby stock
  effect (on CHDN or the broad market) that we are aware of — the desk starts this one
  with a low prior, exactly as with the Eurovision study (708). The interest is
  methodological: a marquee, calendar-fixed event with a *directly exposed* listed stock
  and full price history is the cleanest possible setting to show what "no effect" looks
  like when there is no survivorship funnel to blame.

## What we measure, and the honesty rails

- **The calendar is hardcoded** (`data.py`, `EVENTS`) from Wikipedia's "Kentucky Derby"
  and "List of Kentucky Derby winners", each date cross-checked (the Derby is always the
  first Saturday in May). The **2020** running was postponed by COVID-19 from 2 May to **5
  September 2020** — run, not cancelled — and is flagged `ran_in_may=False`: it stays in
  the 26-event **CHDN** sample (the marquee event happened) but is dropped from the
  25-event **market seasonal** sample (a September date is not a first-Saturday-in-May
  observation). Named, not hidden.
- **No survivorship funnel — but exposure dilution, named on the Signal axis.** CHDN has
  traded continuously since the 1990s, so every event resolves with full tape coverage
  (contrast 708, where half the winners had no ETF). The honest caveat here is different:
  CHDN in 2000 was a near-pure racetrack operator, whereas CHDN today is a diversified
  gaming company (regional casinos, the TwinSpires online-wagering platform,
  historical-racing machines) for which the Derby is a *shrinking* share of revenue. The
  direct exposure the folklore assumes has faded across exactly the window we test — a
  bias that points *against* finding a modern effect, reasoned about explicitly rather
  than buried.
- **One documented execution lag.** The Derby runs Saturday — a non-trading day — so the
  result cannot be acted on before markets reopen. day(-1) = last close before the race
  (does not know the outcome); day(0) = first close after (fully public). The **signal**
  windows run day(-1)→day(-1)+k (the full reaction, including the un-tradable weekend
  jump); the **capture** enters day(0). Because the *date* is fixed years in advance, the
  **run-up** window day(-6)→day(-1) is *calendar-known* and tradable with no result
  look-ahead — tested as its own tradable cut.
- **Inference unit.** Each Derby year is one independent, non-overlapping event — the
  correct test is a **one-sample t** of the abnormal return across events (like 708/707),
  not a daily panel. For the **market seasonal** leg the raw one-sample *t* is
  drift-contaminated, so the **random-window placebo** (same-length windows at random
  points in the same tape, which carry the same drift) is the primary test — several
  nominally-interesting one-sample *t*'s here do **not** survive it, which is exactly the
  gap the desk exists to surface.
- **Costs are honest.** One-way × NAV per leg, charged twice per round trip, at 10 bps
  (a mid-cap single stock, wider than an ETF) with a 5 bps cross-check. The study flags
  the classic trap it walks into: charging costs against an *already-negative* seasonal
  return makes the *t*-stat look *more* significant, not less — a red flag, not a signal.

## Why CHDN, and why SPY

- **`CHDN`** (Churchill Downs Inc., Nasdaq) is the only US-listed equity with direct
  ownership of the Derby; newer gambling names (DraftKings `DKNG` IPO'd 2020, etc.) are
  too young and have no pure Derby exposure, so a single clean instrument beats a noisy
  basket here. Its full history back to the 1990s is what lets the study make the "no
  survivorship excuse" point.
- **`SPY`** is the broad-market total-return proxy for both the seasonal leg and the
  β=1 benchmark CHDN's abnormal return is measured against.

## Data sources

- **Daily adjusted (total-return) closes** for `SPY` and `CHDN` — yfinance (no key),
  cached under `_cache/`.
- **Kentucky Derby dates, 2000→2025** — hardcoded in
  [`data.py`](../kentucky_derby_effect/data.py). Sources: Wikipedia, "Kentucky Derby"
  (https://en.wikipedia.org/wiki/Kentucky_Derby) and "List of Kentucky Derby winners"
  (https://en.wikipedia.org/wiki/List_of_Kentucky_Derby_winners); 2020 postponement
  cross-checked against "2020 Kentucky Derby"
  (https://en.wikipedia.org/wiki/2020_Kentucky_Derby).
- All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is NOT)

- [708-eurovision-effect](../../708-eurovision-effect/) — the closest sibling in shape (a
  hardcoded cultural-event calendar, one-sample *t* across independent yearly events, a
  random-window placebo, a zero-look-ahead capture). Different in a key way: Eurovision's
  story is *survivorship* (half the winners have no tradable ETF); this study has full
  coverage and a *directly exposed* single stock, so its null is cleaner and the caveat is
  exposure dilution, not missing tape.
- [707-plane-crash-effect](../../707-plane-crash-effect/) — the same event-study /
  random-date-placebo / costed-overlay machinery on a *negative* shock (Kaplanski-Levy
  aviation disasters), a market-wide index plus a directly-exposed basket (airlines). This
  study mirrors that "market + directly-exposed name" pairing (SPY + CHDN) for a *positive*
  scheduled event.
- [158-super-bowl](../../158-super-bowl/), [235-world-cup-effect](../../235-world-cup-effect/),
  [709-world-series-effect](../../709-world-series-effect/) — folklore sporting-event
  calendar signals on broad indices. None pairs a single-national-index seasonal with the
  one listed company that *operates* the event — the CHDN angle, and the "no survivorship
  excuse, still nothing" finding, is this study's own contribution.
- Any "Sell in May" / Halloween-indicator study on the bench: this one tests that boundary
  at *event resolution* (the Derby week specifically), not as an aggregate six-month
  seasonal.
