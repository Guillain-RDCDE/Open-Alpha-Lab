# References & literature map — Study 648 (Grain Seasonality)

## The claim under test

- **The folklore.** Grain desks have talked about an "old-crop/new-crop" calendar for a century:
  prices carry a **spring planting-scare premium** — weather risk during planting, pollination
  (corn silking, soybean pod-fill) or, for winter wheat, the spring green-up/freeze window — and
  give it back as a **harvest-time discount**, when the new crop floods elevators and inventories
  rebuild. Working's theory of storage (Working 1949, *The theory of price of storage*, AER) gives
  it a textbook mechanism: near-harvest carrying charges and convenience yield move with the
  scarcity of *old-crop* stocks, which is lowest right before harvest and highest right after.
- **The academic anchor.** Anderson & Danthine (1983, *The time pattern of hedging and the
  volatility of futures prices*, REStud) and the broader "seasonality in commodity futures"
  literature (e.g. Fama & French 1987, *Commodity futures prices: some evidence on forecast power,
  premiums, and the theory of storage*, JB) document a systematic seasonal pattern in grain-futures
  basis and volatility tied to the storage cycle — evidence *for the mechanism*, not a guarantee
  the calendar effect survives in a tradable ETF three decades later.
- **The adjacent (distinct) result.** [639-gasoline-rvp-seasonality](../639-gasoline-rvp-seasonality/)
  tests a *statutory* calendar (a hard EPA blend-switch deadline) on a *refined product*, not a raw
  agricultural commodity with no legal deadline behind it — see the dedup map below for why none of
  the softs/energy seasonality siblings test this claim.

## What we measure, and the honesty rails

- **Month-of-year mean returns, per grain** — 3 grains × 12 calendar months = 36 cells, one-sample
  naive and **Newey-West (1987)** HAC *t*-stats. A **Bonferroni** correction (α = 0.05/36) is the
  honesty rail against reporting the one lucky cell out of 36 draws as "the" seasonal.
- **Best/worst month vs rest** — Welch *t* of the single highest- and lowest-mean month against
  every other month pooled, per grain — the number a chart-watching trader would actually act on.
- **Spring vs harvest, per grain and pooled** — each grain's own hardcoded USDA crop-progress
  window (planting / weather-scare / harvest; see `GRAIN_CALENDAR` in
  [`grain_seasonality/data.py`](../grain_seasonality/data.py)), Welch *t* plus a circular
  block-bootstrap CI (5,000 draws, 12-month blocks per grain, respecting the annual seasonal
  structure) on the pooled spread.
- **Calendar-known execution.** The seasonal timer sets its position from the fixed USDA calendar
  alone — the planting/harvest windows repeat every year and are known well in advance, so the
  study's single documented execution convention needs **no signal-to-trade lag** (unlike a
  data-driven signal, a calendar rule is knowable at the start of the year).
- **Costs charged one-way × NAV per leg** (5/10 bps typical; we use 10 bps here, 4 legs/yr),
  spread evenly across the 12 months, exactly the convention used by
  [307-coffee-seasonality](../307-coffee-seasonality/).

## Why the ETF-vs-futures roll gets its own axis

- CORN/WEAT/SOYB (Teucrium) do not hold spot grain — each rolls a **weighted basket of near,
  second and third contract months**, and pays whatever that roll costs (or, occasionally,
  benefits from) every single month. ZC=F/ZW=F/ZS=F, as pulled from a generic data vendor, are a
  **roll-naive front-month splice** — never a tradable continuous series (nobody can roll for
  free) — used here strictly as a spot-price proxy to size the ETF's own roll drag.
  Geman (2005, *Commodities and Commodity Derivatives*) and the CME's own grain-futures roll
  mechanics documentation cover the construction.
- **Survivorship:** none named — CORN, WEAT and SOYB are all still listed and continuously traded
  since inception; no basket conditioning is involved.
- The roll-drag test (ETF monthly return minus futures-splice monthly return, one-sample *t*)
  answers a narrower, honest question: *given the ETF's own roll mechanics, how much of any
  putative seasonal survives contact with the only vehicle you can actually buy?*

## Data sources

- **CORN, WEAT, SOYB** daily adjusted closes (Teucrium single-commodity grain ETFs) and
  **ZC=F, ZW=F, ZS=F** daily raw closes (CBOT corn/wheat/soybean front-month futures splice) —
  yfinance (no key), cached under `_cache/` (`gs_etf_*.csv`, `gs_fut_*.csv`), 2011-10-03 →
  2026-06-30 (WEAT's 2011-09-19 inception sets the common start across the three ETFs).
- **USDA crop-progress calendar**, hardcoded in
  [`grain_seasonality/data.py`](../grain_seasonality/data.py) (`GRAIN_CALENDAR`). Source: USDA
  NASS "Usual Planting and Harvesting Dates for U.S. Field Crops" —
  https://usda.library.cornell.edu/concern/publications/8336h188j.
- All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is NOT)

- [307-coffee-seasonality](../307-coffee-seasonality/) — the frost/harvest calendar on Arabica
  coffee (KC=F). Same *shape* of test (Welch, HAC, Bonferroni, block-bootstrap, calendar timer),
  a **different crop and a different mechanism** (a single frost event, not a planting/pollination
  weather stretch) — and reaches the same honest `NONE`/`MIRAGE` conclusion independently.
- [308-cocoa-squeeze](../308-cocoa-squeeze/) — a supply-shock/squeeze dynamic in cocoa, not a
  recurring calendar seasonal.
- [309-oj-frost](../309-oj-frost/) — the Florida-frost tail-risk story in orange juice futures.
  Tail risk, not a smooth calendar pattern.
- [226-crude-seasonality](../226-crude-seasonality/) — WTI's own driving-season calendar. Energy,
  not agriculture; no planting/harvest mechanism.
- [227-natgas-winter](../227-natgas-winter/) — the winter-heating-demand calendar in natural gas.
  Weather-demand seasonality, not a storage-cycle/crop calendar.
- [639-gasoline-rvp-seasonality](../639-gasoline-rvp-seasonality/) — a *statutory* calendar (the
  EPA's May-1/September-15 RVP blend-switch deadline) on a refined product. This study's closest
  methodological cousin (ETF-vs-futures roll-drag test, calendar-known execution), but a **legal
  deadline** rather than an agronomic weather/storage cycle — and it reaches the opposite Signal
  stamp (`REAL`, because a statute is a hard date, not a soft weather stretch).
- [651-sugar-seasonality](../651-sugar-seasonality/) — the Brazilian/Indian cane-crush calendar in
  raw sugar. Same *family* (softs/ags calendar seasonality), a different crop and crush cycle.

None of the siblings test the **corn/wheat/soybean planting-scare-to-harvest calendar** on the
three Teucrium grain ETFs specifically — this study's own axis.
