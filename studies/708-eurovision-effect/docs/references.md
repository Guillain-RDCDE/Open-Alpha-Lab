# References & literature map — Study 708 (Eurovision-Effect)

## The claim under test

- **The folklore.** A country that wins — or hosts — the Eurovision Song Contest gets a
  "feel-good bump": a burst of national pride and global attention that is supposed to
  lift consumer and investor sentiment enough to show up in the country's stock market
  the week or month after the Grand Final. It is the tabloid-financial-media cousin of a
  real academic finding, applied to Europe's silliest, most beloved television night.
- **The academic anchor (a different sport, a real effect).** Edmans, García & Norli
  (2007, *Sports Sentiment and Stock Returns*, JF) find a robust next-day market decline
  after a country is **eliminated** from soccer's World Cup — a genuine mood-to-market
  channel, identified off a *loss* shock, in the single sport with the deepest national
  following. Ashton, Gerrard & Hudson (2003, *Economic impact of national sporting
  success*, Applied Economics Letters) find a same-day England-football-result effect on
  the FTSE. Eurovision borrows the mechanism (mood -> risk appetite) but swaps the
  trigger for a *win* / *hosting* signal in a contest whose fanbase, TV reach and
  emotional stakes are a fraction of a World Cup's.
- **What nobody has published.** There is no peer-reviewed study of a Eurovision stock
  effect that we are aware of — this is pure financial-media / social-media folklore
  ("Eurovision economy" pieces appear every May), not a tested academic claim. That
  itself is a data point: the desk starts this one with a lower prior than 235 (which at
  least has Edmans-style backing for the underlying mechanism).

## What we measure, and the honesty rails

- **The calendar is hardcoded** (`data.py`, `EVENTS`) from Wikipedia's "List of
  Eurovision Song Contest winners" and "...host cities," cross-checked per-year for the
  exact Saturday Grand Final date. 2020 is COVID-cancelled (no winner, no market event);
  2021's host (Netherlands) rolled over from 2020; 2023's host (United Kingdom) stood in
  for war-torn Ukraine — both quirks are named, not hidden.
- **Selection, named loudly.** Of 25 non-cancelled winner-country events and 25 host-
  country events (50 possible role-events, 2000->2025), only **12 winner** and **13
  host** events have BOTH a mapped single-country US-listed ETF AND VGK benchmark
  coverage around the final. The sample that survives is systematically the **richer,
  more-developed half of Europe** (Germany, Sweden, Austria, Switzerland, UK, Italy,
  Netherlands, Denmark, Israel, Portugal, Norway) — Estonia, Latvia, Serbia, Azerbaijan
  and (three-time winner) Ukraine never had a tradable single-country vehicle at all,
  and Russia's ERUS was effectively erased from the data record after the 2022
  sanctions. A "does the winner's market rally" test that structurally can only ever
  answer for the EU-core / G7-adjacent subset of winners is not testing the full
  folklore — it's testing the tradable slice of it, and that slice is disclosed here.
- **One documented execution lag.** The Grand Final airs Saturday night — a non-trading
  day — so nobody can act on the result before markets reopen. day(-1) = the last close
  before the final (does not know the winner); day(0) = the first close after (fully
  public, ~36-60h old). The **signal** measurement runs day(-1)->day(-1)+k (the full
  announcement reaction, including the un-tradable weekend jump); the **tradability**
  measurement enters at day(0)'s close instead — zero look-ahead by construction, and
  the honest gap between "the effect exists" and "you could have banked it."
- **Inference unit.** Each Eurovision year is one independent, non-overlapping event —
  the correct test is a **one-sample t** of the abnormal return across events (like
  235's per-edition t-test), not a daily panel regression. A random-window placebo
  (drawing many non-Eurovision k-session windows from the *same* tickers) checks
  whether the observed mean sits outside the tickers' own ordinary week-to-week
  tracking noise against VGK — several nominally ">= 2" one-sample t's here do **not**
  survive that placebo, which is exactly the kind of gap the desk exists to surface.

## Why VGK, not FEZ

FEZ (Euro Stoxx 50) is Eurozone-only. Five of our included countries — the United
Kingdom, Switzerland, Sweden, Norway and Denmark — are European but **not** in the
euro, and are core members of our winner/host sample. VGK (Vanguard FTSE Europe) spans
both euro and non-euro Europe and is the only one of the two benchmarks that is a fair
counterfactual for every country in this study. VGK's own inception (2005-03-10) is a
second, independent floor on how far back the test can reach — a real constraint,
disclosed rather than patched with an index proxy.

## Data sources

- **Daily adjusted (total-return) closes** for every mapped country ETF (`EDEN`, `TUR`,
  `GREK`, `EFNL`, `NORW`, `EWG`, `EWD`, `EWO`, `PGAL`, `EIS`, `EWN`, `EWI`, `EWU`,
  `EWL`) and the `VGK` Europe benchmark — yfinance (no key), cached under `_cache/`.
- **Eurovision winners and host cities/dates, 2000->2025** — hardcoded in
  [`data.py`](../eurovision_effect/data.py). Sources: Wikipedia, "List of Eurovision
  Song Contest winners" (https://en.wikipedia.org/wiki/List_of_Eurovision_Song_Contest_winners)
  and "List of Eurovision Song Contest host cities"
  (https://en.wikipedia.org/wiki/List_of_Eurovision_Song_Contest_host_cities), Grand
  Final dates cross-checked against "History of the Eurovision Song Contest"
  (https://en.wikipedia.org/wiki/History_of_the_Eurovision_Song_Contest).
- All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is NOT)

- [158-super-bowl](../../158-super-bowl/) — the NFC/AFC "Super Bowl Indicator" on the
  S&P 500. Same shape (a folklore calendar signal, tested with the correct baseline and
  a permutation test) — different sport, different market (a single national index, not
  a per-country abnormal-return panel).
- [235-world-cup-effect](../../235-world-cup-effect/) — the Edmans-style national-
  sentiment effect for **football World Cup** windows on the S&P 500. The closest real
  academic anchor to this study's mechanism, but tests a single market (US) against a
  single global tournament window, not a per-winner/per-host country panel. That study
  finds the signal confounded by macro crises coinciding with tournament windows; this
  study's confound is different (selection into which countries even have a tradable
  ETF).
- [709-world-series-effect](../../709-world-series-effect/) — the same "home-market
  feel-good bump" shape applied to MLB's World Series. Different sport, different
  (single-country) market; no cross-country panel.
- [234-olympic-year](../../234-olympic-year/) — "stocks rally in Olympic years," a
  single-market annual-frequency calendar effect, not an event-window, not a per-
  country panel. Distinguishes the *Olympic year itself* (a macro-frequency claim) from
  *this contest's Grand Final* (a single-week event with a single execution lag).

None of the siblings test a **per-country abnormal-return panel keyed to a specific
non-sporting cultural contest** — the Eurovision angle, including the "half of Europe's
folklore has no tradable market at all" finding, is this study's own contribution.
