# References & literature map — Study 651 (Sugar-Seasonality)

## The claim under test

- **The folklore.** Raw-sugar desks talk about a "crush calendar" the way grain desks talk about
  "old-crop/new-crop": Brazil's Center-South mills — the world's largest single raw-sugar supply
  source — crush cane roughly **April through November**, and India, the world's second-largest
  producer, crushes roughly **October through April**. Between the two, old-crop stocks are
  supposedly scarcest in the Northern-Hemisphere winter, right before Brazil's new crush gets into
  full swing, giving prices a **pre-harvest-tight premium** that unwinds every spring as the
  Brazilian crush floods the market with new-crop supply — a **crush-glut discount**.
- **The academic anchor.** The general storage-cycle mechanism is the same one grain desks invoke —
  Working's theory of storage (Working 1949, *The theory of price of storage*, AER): near-harvest
  carrying charges and convenience yield move with the scarcity of old-crop stocks. For sugar
  specifically, the USDA's *Sugar and Sweeteners Outlook* and the International Sugar Organization's
  *Sugar Year Book* document the seasonal shape of the Brazilian and Indian crush and its effect on
  world raw-sugar supply timing, without claiming the calendar survives as a tradable price pattern
  in any specific instrument.
- **The adjacent (distinct) result.** [648-grain-seasonality](../648-grain-seasonality/) tests the
  identical *shape* of claim — a planting-scare premium unwinding into a harvest-time discount — on
  corn, wheat and soybeans, a different crop family with a genuinely different mechanism (a weather
  *scare* during pollination/pod-fill, not a slow multi-month crush). See the dedup map below for
  why none of the softs/grains seasonality siblings test this specific claim.

## What we measure, and the honesty rails

- **Month-of-year mean returns** — 12 calendar-month cells, one-sample naive and **Newey-West
  (1987)** HAC *t*-stats. A **Bonferroni** correction (α = 0.05/12) is the honesty rail against
  reporting the one lucky cell out of 12 draws as "the" seasonal — the same discipline used by
  [307-coffee-seasonality](../307-coffee-seasonality/) for a single-instrument 12-cell grid (as
  opposed to 648's 36-cell three-grain grid, which needs a stricter /36 bar).
- **Best/worst month vs rest** — Welch *t* of the single highest- and lowest-mean month against
  every other month pooled — the number a chart-watching trader would actually act on.
- **Pre-harvest tight vs crush-glut** — the claimed Jan–Mar "tight" window vs the claimed Apr–Jul
  "crush" window (see `SUGAR_CALENDAR` / `TIGHT_MONTHS` / `CRUSH_MONTHS` in
  [`sugar_seasonality/data.py`](../sugar_seasonality/data.py)), Welch *t* plus a circular
  block-bootstrap CI (5,000 draws, 12-month blocks, respecting the annual seasonal structure) on the
  spread.
- **Calendar-known execution.** The seasonal timer sets its position from the fixed crush calendar
  alone — the Brazilian and Indian crush windows repeat every year and are known well in advance, so
  the study's single documented execution convention needs **no signal-to-trade lag**.
- **Costs charged one-way × NAV per leg** (10 bps here, 4 legs/yr), spread evenly across the 12
  months, exactly the convention used by [648-grain-seasonality](../648-grain-seasonality/) and
  [307-coffee-seasonality](../307-coffee-seasonality/).

## Why the ETF-vs-futures roll gets its own axis

- CANE (Teucrium) does not hold spot raw sugar — it rolls a **weighted basket of near, second and
  third ICE No.11 contract months**, and pays whatever that roll costs (or, occasionally, benefits
  from) every single month. SB=F, as pulled from a generic data vendor, is a **roll-naive
  front-month splice** — never a tradable continuous series (nobody can roll for free) — used here
  strictly as a spot-price proxy to size the ETF's own roll drag.
- **Survivorship:** none named — CANE has been continuously listed and traded since its 2011-09-19
  inception; no basket conditioning is involved.
- The roll-drag test (CANE monthly return minus SB=F-splice monthly return, one-sample *t*) answers
  a narrower, honest question: *given the ETF's own roll mechanics, how much of any putative
  seasonal survives contact with the only vehicle you can actually buy?*

## Data sources

- **CANE** daily adjusted closes (Teucrium Sugar Fund) and **SB=F** daily raw closes (ICE No.11 raw
  sugar front-month futures splice) — yfinance (no key), cached under `_cache/`
  (`ss_etf_cane.csv`, `ss_fut_sb.csv`), 2011-10-03 → 2026-06-30 (CANE's 2011-09-19 inception sets
  the start).
- **Crush calendar**, hardcoded in [`sugar_seasonality/data.py`](../sugar_seasonality/data.py)
  (`SUGAR_CALENDAR`). Sources: USDA FAS *Sugar: World Markets and Trade*
  (https://www.fas.usda.gov/data/sugar-world-markets-and-trade), UNICA Brazil crush-progress
  reports (https://unicadata.com.br), and the International Sugar Organization's published crop
  calendars.
- All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is NOT)

- [307-coffee-seasonality](../307-coffee-seasonality/) — the Brazilian-frost/harvest calendar on
  Arabica coffee (KC=F). Same *shape* of test (Welch, HAC, Bonferroni-12, block-bootstrap, calendar
  timer), a **different crop and a different mechanism** (a single tail-risk frost event, not a
  multi-month crush) — and reaches the same honest `NONE`/`MIRAGE` conclusion independently.
- [308-cocoa-squeeze](../308-cocoa-squeeze/) — a supply-shock/squeeze dynamic in cocoa, not a
  recurring calendar seasonal.
- [648-grain-seasonality](../648-grain-seasonality/) — the "old-crop/new-crop" planting-scare-to-
  harvest calendar in corn, wheat and soybeans. This study's closest methodological cousin (Welch/
  HAC month table, Bonferroni, block-bootstrap CI, ETF-vs-futures roll-drag test, costed calendar
  timer, coin-flip hit-rate test) — but a **different crop family and mechanism** (a spring weather
  scare during planting/pollination, not a slow multi-month Southern-Hemisphere cane crush) — and it
  independently reaches the same `NONE`/`MIRAGE` verdict.

None of the siblings test the **raw-sugar Brazil/India crush calendar** on CANE specifically — this
study's own axis.
