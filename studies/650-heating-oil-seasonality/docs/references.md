# References & literature map — Study 650 (Heating-Oil-Seasonality)

## The claim under test

- **The folklore.** Energy-desk shorthand: "buy heating oil ahead of winter" — distillate
  demand (home heating, diesel-adjacent) peaks with cold weather, so the futures price should
  **build** through the autumn (the market pricing the coming season ahead of time) and hold or
  extend the rally through the winter **draw** (physical inventories fall as furnaces run,
  EIA's weekly distillate stocks report). It's the same seasonal-demand mechanism that motivates
  [227-natgas-winter](../227-natgas-winter/) for natural gas — this study asks whether the
  parallel claim holds for the other classic winter-demand commodity.
- **The physical anchor.** The U.S. Energy Information Administration publishes weekly
  distillate fuel oil stocks (EIA Weekly Petroleum Status Report,
  https://www.eia.gov/petroleum/weekly/), which do show a genuine seasonal draw-down pattern
  each winter — the demand mechanism itself is real. Fama & French (1987, *Commodity futures
  prices: some evidence on forecast power, premiums, and the theory of storage*, JB) is the
  classic reference for how storable-commodity seasonals show up (or fail to) in futures prices
  once a market prices the calendar in advance.
- **The adjacent (distinct) results on this desk.** [226-crude-seasonality](../226-crude-seasonality/)
  tests WTI crude's spring-driving-season pattern — a different commodity, a different season.
  [639-gasoline-rvp-seasonality](../639-gasoline-rvp-seasonality/) tests a *statutory* calendar
  (the EPA's May-1 low-RVP blend deadline) on the gasoline-crude spread — a law, not a demand
  story, and it *is* Real (on the spread) even though it's still a Mirage once a real holder
  pays the roll. [306-crack-spread](../306-crack-spread/) tests whether the crack level
  *predicts* refiner equities — a completely different mechanism (predictive vs coincident).
  [227-natgas-winter](../227-natgas-winter/) runs the closest parallel test — winter demand,
  wrong-signed on the tape — for natural gas.

## What we measure, and the honesty rails

- **Two claim halves, tested separately.** "Autumn build" (Sep–Nov) and "winter draw"
  (Dec–Feb) are the two stages believers name; testing them as one blended "heating season"
  window would hide a case where they disagree (they do — see Results). Both are Welch-tested
  against the same off-season control (Mar–Aug).
- **Per-month table carries a Bonferroni-12 bar**, not the naive |*t*| ≥ 2 — testing 12 months
  is 12 chances to find one "significant" by luck (same convention as
  [226-crude-seasonality](../226-crude-seasonality/) and
  [639-gasoline-rvp-seasonality](../639-gasoline-rvp-seasonality/)).
- **The roll/contango caveat is structural, not bolted on.** HO=F on Yahoo is a continuous
  front-month chain **without back-adjustment** — the price jump on a roll day *is* the
  term-structure cost (contango) or gain (backwardation) a real futures roller experiences
  rolling the same day. Unlike a back-adjusted "clean spot" series, this tape already contains
  the friction a trader would actually pay; the study does not need a separate synthetic
  roll-cost model to make that point.
- **Execution.** The seasonal timer enters Sep 1 and exits end-February on a fixed calendar
  rule — no forecast, no look-ahead by construction. Costs are one-way × NAV per switch (2
  switches/yr).

## Why the tradable echo is graded separately

- **UHN (United States Heating Oil Fund, USCF)** is the vehicle that actually let retail
  express this trade: it held front-month HO=F futures directly, 2008-04-10 → 2018-09-11.
  USCF wound the fund down for lack of assets — it is a **discontinued product**, not merely a
  historical curiosity, and a reader today has no way to buy it at any price. This is named
  explicitly rather than buried: any UHN-based backtest necessarily stops in 2018, and the
  absence of a successor product is itself part of the tradability verdict.
- **Paired holder-vs-splice gap.** UHN lost to the raw HO=F splice in 8 of 10 heating seasons
  (mean gap −3.18%/season) — the ETF wrapper (expense ratio + tracking slippage) stacks a
  *second* drag on top of whatever roll cost the futures chain already embeds. The sample is
  small (n=10 non-overlapping seasons); the *t* (−2.58) is reported honestly as uncertifiable
  at this size, and the point estimate is never cited as if it were.
- Costs are charged one-way × NAV per leg on the futures-based timer (5/10 bps).

## Data sources

- **HO=F daily raw OHLC**, **UHN adjusted closes** and **^IRX closes** — yfinance (no key),
  cached under `_cache/` (`hos_ho.csv`, `hos_uhn.csv`, `hos_irx.csv`), 2000-09-01 → 2026-06-30
  (UHN 2008-04-10 → 2018-09-11, its full trading life). CME NY Harbor ULSD futures
  specification: https://www.cmegroup.com/markets/energy/refined-products/heating-oil.html
- **EIA Weekly Petroleum Status Report** (distillate stocks, the physical demand anchor):
  https://www.eia.gov/petroleum/weekly/
- The heating-season calendar (autumn-build Sep–Nov, winter-draw Dec–Feb) is hardcoded in
  [`data.py`](../heating_oil_seasonality/data.py) — a calendar fact, not a fitted parameter.
- All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is NOT)

- [227-natgas-winter](../227-natgas-winter/) — the parallel winter-demand claim for **natural
  gas** (UNG). Different commodity, same demand mechanism, same wrong-signed conclusion — this
  study is heating oil's own tape, not a rerun of that one.
- [639-gasoline-rvp-seasonality](../639-gasoline-rvp-seasonality/) — a **statutory** calendar
  (the EPA's RVP blend-switch deadline) on the gasoline-crude **spread**. A law with a
  provable mechanism, real on the spread, mirage once a holder pays the roll — this study finds
  no real signal to begin with, on a *demand* story with no legal anchor.
- [306-crack-spread](../306-crack-spread/) — does the crack **spread level** *predict*
  refiner-equity returns day to day? A same-day-coincident-vs-predictive question, not a
  calendar seasonal.
- [226-crude-seasonality](../226-crude-seasonality/) — WTI crude's **spring** driving-season
  pattern (Weak, regime-dependent). A different commodity and a different season from this
  study's autumn/winter window.

None of the siblings test HO=F's own autumn-build / winter-draw calendar — that is this
study's own axis.
