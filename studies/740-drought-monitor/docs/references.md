# References & literature map — Study 740 (Drought-Monitor)

## The claim under test

- **The data source.** The **US Drought Monitor** (USDM) — a joint product of the
  National Drought Mitigation Center at the University of Nebraska–Lincoln, USDA and
  NOAA — has published a weekly map of US drought severity (categories **D0
  abnormally-dry → D4 exceptional drought**) every **Thursday morning (~8:30 ET, data
  through the prior Tuesday) since 2000** (droughtmonitor.unl.edu). Its summary
  time-series (the **Drought Severity and Coverage Index**, DSCI, and the % of area in
  each category) is the canonical, cited, machine-published measure of US drought. The
  Monitor is watched closely by USDA (it triggers automatic disaster-relief and crop-
  insurance provisions) and by agricultural commodity desks.
- **The folklore / steelman.** The tradable story writes itself: severe drought across
  the US crop belt → smaller corn/wheat/soybean harvests → higher grain prices → good
  for the ag complex (farm-equipment maker **Deere**, fertilizer maker **Mosaic**,
  processor/trader **ADM**, the agribusiness ETF **MOO**) and for grain itself (the
  broad-ag **DBA**, corn **CORN**, wheat **WEAT** ETFs). If you read the Thursday
  Drought Monitor and it shows a bad, worsening picture, buy the drought. This is the
  agricultural cousin of any weather-driven supply-shock trade.
- **The academic backdrop.** There is a real literature that **weather and drought move
  agricultural commodity prices** — e.g. work on ENSO/El Niño and crop prices (Ubilava,
  2018, "The role of El Niño Southern Oscillation in commodity price movement and
  predictability", *American Journal of Agricultural Economics*), and event studies of
  USDA WASDE crop-report surprises on grain futures (e.g. Adjemian, 2012, "Quantifying
  the WASDE announcement effect", *American Journal of Agricultural Economics*). The
  open question this study asks is narrower and more skeptical: is the *public, weekly,
  pre-scheduled* Drought Monitor print itself a **tradable event** for US-listed ag
  equities and ETFs — or is the drought already fully in prices by the time the Monitor
  confirms it? Efficient-markets priors (Fama, 1970) say a widely-watched public index
  of a slowly-evolving, continuously-forecast physical condition should carry little
  surprise on release.

## What we measure, and the honesty rails

- **Print-day (day 0) abnormal return** — abnormal = equal-weight basket return **minus
  SPY** (a beta-1 market model), so a result is "the ag complex outperformed the market
  on the print", not "stocks moved". One-sample *t* across the **21 independent,
  non-overlapping Thursday prints** (the planned primary; events are far apart in time,
  so no HAC correction is needed the way a daily-panel regression would). Up-rate carries
  a **Wilson (1927)** interval.
- **Event window [−1..+5]** with each offset's own one-sample *t*, read honestly as a
  **multiple-comparison** exercise — 7 offsets, so ~1-in-7 crossing |*t*| ≥ 2 by chance
  is expected and is called out as such when it happens (offset +5 in this run).
- **Post-print drift [+1..+5]** with a random-calendar placebo *p* and an **event
  bootstrap** CI, testing the "the ag complex drifts up after a drought print" half of
  the claim directly rather than eyeballing a chart.
- **The grain-vs-equity third axis** — a *paired* same-date (grain basket − ag-equity
  basket) day-0 difference, one-sample *t*'d across events, isolating whether the more
  directly weather-exposed grain/commodity vehicles react harder than the equities.
- **The random-calendar placebo** is a 20-seed × 1,000-draw null (the same falsification
  design as `707-plane-crash-effect` and `313-geopolitical-shock`).
- **Coverage named, not hidden.** The ag-equity basket has full 4-name coverage only
  from 2007 onward (Mosaic IPO'd 2004, MOO launched 2007); the earliest events fall back
  to DE/ADM. The grain basket exists only from 2007 (DBA), with CORN from 2010 and WEAT
  from 2011 — so the grain test runs on the 16 post-2007 events, and the 5 earliest are
  dropped, not zero-filled.
- **The regime test** splits months by the labelled drought proxy **known at the month's
  start** (one `shift`, no look-ahead) — the single documented convention there, distinct
  from the event study's Thursday-release snap.

## The labelled proxy — stated as a decision, not hidden

The monthly **D2+ severe-drought coverage %** series in
[`data.py`](../drought_monitor/data.py) (`DROUGHT_PROXY`) is a **hand-digitised
approximation** of the US Drought Monitor's public time-series, not a machine pull of the
USDM archive. It is used **only** for the context chart and the drought-regime split —
never for the event study, which keys off the hardcoded Thursday-release dates. This is
the same labelled-proxy discipline the desk uses for any macro/alt-data series not on
yfinance (`358-watch-index`'s auction-price index, `708-eurovision-effect`'s hardcoded
calendar): a small, clearly-cited, approximate series, disclosed as approximate, never
dressed up as a real tape. The event-study verdict does not depend on it.

## Data sources

- **SPY, DE, MOS, ADM, MOO, DBA, CORN, WEAT** daily total-return closes
  (`auto_adjust=True`) — yfinance (no key), cached under `_cache/` (`dm_spy.csv`,
  `dm_de.csv`, …), 2000-01-03 → 2026-06-30 (each ETF from its inception).
- **21 hardcoded major US drought-intensification episodes, 2000 → 2025**, in
  [`drought_monitor/data.py`](../drought_monitor/data.py) — hand-curated Thursday USDM
  release dates cross-referenced against the US Drought Monitor archive
  (droughtmonitor.unl.edu) and contemporary drought reporting.
- All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is NOT)

- [707-plane-crash-effect](../../707-plane-crash-effect/) — the sibling news-event study
  (Kaplanski-Levy aviation-disaster sentiment). Same event-study machinery (a hardcoded
  shock calendar, a random-calendar placebo, a paired sector-extra-move test, a costed
  timer) — but a *sentiment* shock on the broad market, not a *supply-shock* signal on a
  named sector via a scheduled public index.
- [313-geopolitical-shock](../../313-geopolitical-shock/) — wars/invasions/terror on the
  broad market; a shock calendar, but a geopolitical trigger, not an agricultural physical
  condition, and not a pre-scheduled weekly index release.
- [637-fomc-vol-crush](../../637-fomc-vol-crush/) — a *scheduled public release* event
  study (FOMC), the closest structural cousin (a known-in-advance calendar of official
  prints), but on index volatility, not an ag-sector directional bet.

None of the siblings test whether **a scheduled weekly public drought index is tradable
news for the agricultural equity + grain complex** — that is this study's own axis.
