# References & literature map — Study 730 (Ferrari-F1)

## The claim under test

- **The folklore.** Ferrari (NYSE: `RACE`) is the rare listed company whose brand *is* a
  Formula 1 team, so the "tifosi" fan base and the halo of a Grand Prix win are supposed
  to lift the stock — a Ferrari-specific brand-sentiment "pop" on the Monday after the
  Scuderia wins. It appears every season in retail-investor and motorsport-business
  commentary ("Ferrari stock and on-track success"), the Ferrari-flavoured version of the
  broader sports-sentiment idea.
- **The academic anchor (a different mechanism, a real effect).** Edmans, García & Norli
  (2007, *Sports Sentiment and Stock Returns*, Journal of Finance 62(4):1967-1998) find a
  robust next-day *national market* decline after a country is **eliminated** from
  soccer's World Cup — a genuine mood-to-market channel, identified off a *loss* shock at
  the index level. Ashton, Gerrard & Hudson (2003, *Economic impact of national sporting
  success*, Applied Economics Letters 10:783-785) find a same-day England-football-result
  effect on the FTSE. The Ferrari claim borrows the mechanism (mood → risk appetite) but
  moves it to a *win* signal on a **single stock** rather than a national index.
- **The single-stock, single-team precedent.** Sports-result effects on individual listed
  clubs are the closest genre: Renneboog & Vanbrabant (2000) and Palomino, Renneboog &
  Zhang (2009, *Information salience, investor sentiment, and stock returns: the case of
  British soccer betting*, Journal of Corporate Finance 15:368-387) show listed football
  clubs' shares react to match results (up on wins, more on losses) — but a club's *entire*
  cash flow is its sporting results, whereas Ferrari's F1 team is a marketing line item
  against a global luxury-goods business. Bell, Brooks, Matthews & Sutcliffe (2012,
  *Over the moon or sick as a parrot?*, Applied Economics 44:3435-3452) is the same genre.
- **Why the prior is low for Ferrari specifically.** Ferrari is valued as a
  high-margin luxury manufacturer (Hermès-like multiples, tiny fixed unit volumes,
  waitlists) — analyst notes key on shipments, price/mix and margin, not race results.
  There is no peer-reviewed study of a Ferrari-`RACE` F1-win stock effect that we are aware
  of; this is folklore, and the desk starts it with a lower prior than a national-index
  sports-sentiment test.

## What we measure, and the honesty rails

- **The calendar is hardcoded** (`data.py`, `EVENTS`) — every Ferrari Grand Prix
  **victory** in the RACE-listed era, 24 wins across 2017→2024, with the exact Sunday race
  date and winning driver. Ferrari were **winless in 2016, 2020, 2021 and 2025** (2025
  their first winless campaign since 2021, with Hamilton's China *Sprint* win the season's
  only trip to the top step — a sprint, not a Grand Prix, and excluded). Sources below.
- **One documented execution lag.** Every Grand Prix runs **Sunday** — a non-trading day —
  so nobody can act on the result before markets reopen. day(-1) = the last close before
  the race (does not know the result); day(0) = the first close after (fully public,
  ~12-40h old). The **signal** measurement runs day(-1)→day(-1)+k (the full reaction,
  including the un-tradable weekend jump); the **tradability** measurement enters at
  day(0)'s close — zero look-ahead by construction, and the honest gap between "the effect
  exists" and "you could have banked it."
- **Inference unit.** Each Ferrari win is one independent event → the correct test is a
  **one-sample t** of the abnormal return across wins, not a daily panel regression. At the
  day(0) horizon all 24 events are distinct Mondays (exact independence); at the 1-week
  horizon three back-to-back pairs overlap and are named and re-run dropped. A
  **random-calendar placebo** (drawing the same number of random, non-race anchors from
  RACE's own history) checks whether the observed mean sits outside RACE's ordinary
  week-to-week noise against SPY.
- **Fundamentals vs sentiment, pre-registered.** The wins are tagged **contender** (2017-18,
  a win updated a live title fight) vs **sporadic** (2019, 2022-24, one-off wins) *before*
  the test, so the contrast that separates "the market repriced a championship" from "fans
  cheered" is not a post-hoc slice.

## Why SPY, not an auto/luxury peer

RACE is a USD, NYSE-listed line, so `SPY` (SPDR S&P 500) is the plainest fair market
counterfactual — it removes the market-wide component of any given Monday move. A tighter
control would net out sector/luxury beta too (RACE vs a European luxury basket — LVMH,
Hermès, or an ETF proxy), which would sharpen the "Ferrari-*specific*" reading. That is a
deliberate sequel, flagged here rather than silently swapped in; SPY keeps the headline
comparison transparent and reproducible with two tickers.

## Data sources

- **Daily adjusted (total-return) closes** for `RACE` and `SPY` — yfinance (no key),
  cached under `_cache/`. RACE history begins at its 2015-10-21 NYSE listing.
- **Ferrari F1 victories, 2017→2024** — hardcoded in
  [`data.py`](../ferrari_f1/data.py). Sources: STATS F1, "Ferrari — Wins"
  (https://www.statsf1.com/en/ferrari/victoire.aspx); Wikipedia, "Ferrari Grand Prix
  results" (https://en.wikipedia.org/wiki/Ferrari_Grand_Prix_results); each race date
  cross-checked against the official Formula1.com season results pages.
- **Ferrari N.V. listing facts** — Ferrari N.V. Form 20-F (FY2015) and the NYSE
  first-trade date (2015-10-21, USD 52); Milan (MTA) listing 2016-01-04.
- All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is NOT)

- [708-eurovision-effect](../../708-eurovision-effect/) — a national-market "feel-good
  bump" keyed to winning/hosting a *cultural* contest, tested as a per-country
  abnormal-return panel. Same event-study template and one-sample-t machinery; different
  trigger (song contest vs motor race) and unit (a panel of country ETFs vs a single
  stock).
- [235-world-cup-effect](../../235-world-cup-effect/) — the Edmans-style national-sentiment
  effect for football World Cup windows on the S&P 500. The closest real academic anchor
  to this study's *mechanism*, but a single national index against a global tournament,
  not a single company against its own team's results.
- [707-plane-crash-effect](../../707-plane-crash-effect/) — the same event-study + random-
  calendar placebo + costed timer shape applied to a *dread* sentiment shock. Shares the
  method (hardcoded event table, one-sample t, placebo, synthetic control); different
  sentiment sign and instrument.
- [158-super-bowl](../../158-super-bowl/), [709-world-series-effect](../../709-world-series-effect/),
  [234-olympic-year](../../234-olympic-year/) — other sports-folklore calendar/mood claims,
  each tested the same honest way, all on national indices rather than the one stock that
  *is* a race team.

None of the siblings test a **single listed company's abnormal return keyed to its own
sports team's wins** — the "the brand *is* the team" angle, and the fundamentals-vs-fans
reattribution, is this study's own contribution.
