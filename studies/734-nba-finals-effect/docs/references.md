# References & literature map — Study 734 (NBA-Finals-Effect)

## The claim under test

- **The folklore.** When a team **loses** the NBA Finals, its home city's local market
  should dip (a deflated-fanbase mood shock); when a team **wins**, its city gets a feel-good
  pop. This is the sports-radio / local-business-page cousin of a real academic finding,
  aimed at basketball's single biggest night.
- **The academic anchor (a different sport, a real effect).** Edmans, García & Norli (2007,
  *Sports Sentiment and Stock Returns*, **Journal of Finance** 62(4):1967-1998) find a robust
  **next-day decline of ~49 bps** in a country's national stock index after that country is
  **eliminated** from the soccer World Cup (smaller but present for elimination from the Euros,
  Copa América, and other continental tournaments) — a genuine mood-to-market channel,
  identified off a *loss* shock, in the sport with the deepest national following. Crucially,
  they find the effect is **asymmetric**: losses move markets, wins essentially do not. Their
  identification *requires each competitor to have its own national stock market*.
- **The supporting local-sentiment literature.** Ashton, Gerrard & Hudson (2003, *Economic
  impact of national sporting success: evidence from the London Stock Exchange*, **Applied
  Economics Letters** 10(12):783-785) find a same-day England-football-result effect on the
  FTSE. On the *local* channel this study actually probes, Kaplanski & Levy (2010, *Sentiment
  and stock prices: The case of aviation disasters*, **JFQA** 45(2):473-496) is the general
  "vivid-mood-shock moves prices" template; and the behavioral-finance "local bias" line
  (Coval & Moskowitz 1999, *Home Bias at Home*, **Journal of Finance** 54(6):2045-2073;
  Grinblatt & Keloharju 2001) motivates *why* a city's own investors might, in principle,
  react to a civic mood shock — the premise the proxy test leans on.
- **What nobody has published.** There is no peer-reviewed study of an NBA-Finals *home-city*
  stock effect that we are aware of — this is folklore, not a tested academic claim. That is
  itself a data point: the desk starts this one with a low prior, and with a structural
  reason for skepticism (below) that has nothing to do with the data.

## Why the mechanism (probably) can't transfer — stated before the test

EGN's effect is identified off **cross-country** variation: an eliminated country's market
falls *relative to the rest of the world*. The NBA Finals, in 2000→2025, is **25 times out of
26 a USA-vs-USA event** — both the winning and losing city trade the *same* national tape
(`SPY`). A market-wide "Finals mood" shock therefore **nets out** (one US city elated, one
deflated), which is exactly what the broad-`SPY` cross-check finds. The only Finals in the
window that spans two national markets is **2019** (Toronto Raptors, Canada, over Golden
State, USA) — and one event is not a test. This is the null-by-design argument, announced up
front, that the teardown then confirms.

## What we measure, and the honesty rails

- **The calendar is hardcoded** (`data.py`, `EVENTS`) from Basketball-Reference / NBA.com
  official Finals results, cross-checked per year against Wikipedia for the exact
  series-clinching game date. 2020 clinched in the Orlando bubble in October; 2021 was
  COVID-delayed into July — both named, neither a cancellation (unlike Eurovision 2020 in the
  sibling study 708).
- **The "civic proxy" is a LABELLED, coarse proxy** — the load-bearing caveat. No US city has
  a stock index, so each metro maps to a single **real, tradable** hometown large-cap (a
  regional bank like `CFR`/`KEY`, an iconic local employer like `F`/`LLY`/`DIS`, or the local
  utility `PEG`); Toronto maps to the `EWC` Canada ETF. These are genuine instruments you
  could trade, but a single stock's daily return is dominated by *its own business*, not civic
  mood — so a real home-market pulse would be buried in idiosyncratic noise. The null we
  report is consistent both with "no effect" and with "an effect too small to see through a
  single-stock proxy," and we say so.
- **One documented execution lag.** The clinching game tips ~9pm and ends ~11:30pm local — a
  non-trading time — so nobody can act before markets reopen. day(-1) = the last close on or
  before the game date (does not know the result; for a weekday game this is that day's own
  4pm close, five hours before tip-off); day(0) = the first close after. The **signal**
  measurement runs day(-1)→day(-1)+k (the full announcement reaction, including the
  un-tradable overnight jump); the **tradability** measurement enters at day(0)'s close
  instead — zero look-ahead by construction.
- **Inference unit.** Each Finals is one independent, non-overlapping event — the correct test
  is a **one-sample t** of the abnormal return across events, not a daily panel regression. A
  random-window placebo (drawing many non-Finals k-session windows from the *same* proxies)
  checks whether the observed mean sits outside the proxies' ordinary tracking noise against
  `SPY`.

## Data sources

- **Daily adjusted (total-return) closes** for every mapped metro proxy (`DIS`, `LLY`,
  `CMCSA`, `PEG`, `CFR`, `F`, `CCL`, `T`, `KEY`, `STT`, `DRI`, `DVN`, `WFC`, `EWC`, `ROK`,
  `RSG`, `DVA`) and the `SPY` US benchmark — yfinance (no key), cached under `_cache/`.
- **NBA Finals champions, runners-up and clinching-game dates, 2000→2025** — hardcoded in
  [`data.py`](../nba_finals_effect/data.py). Sources: Basketball-Reference "NBA Finals" index
  (https://www.basketball-reference.com/playoffs/) and NBA.com, dates cross-checked against
  each season's Wikipedia "20xx NBA Finals" page.
- All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is NOT)

- [708-eurovision-effect](../../708-eurovision-effect/) — the same EGN-borrowing,
  per-entity-abnormal-return event-study *shape* (winner/host country ETFs vs a regional
  benchmark), but for a **song contest across separate national markets**, where the
  cross-country design actually has teeth. This study is its US-sports twin, where the shared
  tape collapses the design — the exact contrast worth reading side by side.
- [709-world-series-effect](../../709-world-series-effect/) — the MLB "champion's league/city
  is an omen for *next year's* market" claim: a **next-year directional omen on ^GSPC**, tested
  with a permutation test against the market's own up-rate. Different mechanism entirely (a
  year-ahead omen, not a days-long event-window sentiment shock) and a different unit.
- [235-world-cup-effect](../../235-world-cup-effect/) — the *original* Edmans mechanism
  (football World Cup) tested on the S&P 500. The closest real academic anchor to this study's
  claim, but a single-market, single-tournament test, not a per-city abnormal-return panel.
- [158-super-bowl](../../158-super-bowl/) / [720-super-bowl-advertiser](../../720-super-bowl-advertiser/)
  — the Super Bowl Indicator (NFC/AFC omen) and the advertiser-stock angle: US football, a
  single national index or an ad-exposed basket, not a champion/loser home-city sentiment
  panel.

None of the siblings test a **per-city abnormal-return panel keyed to the NBA Finals'
champion and runner-up home markets** — the basketball angle, including the "one shared US
tape collapses the cross-country EGN design" finding, is this study's own contribution.
