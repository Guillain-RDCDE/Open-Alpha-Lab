# References & literature map — Study 556 (Electricity-Demand)

## The claim, at full strength

- **The "electricity is the pulse of GDP" folklore.** A long-standing macro-alt-data thesis:
  because almost all real activity — manufacturing, data centres, commercial floor space,
  households — runs on the grid, **aggregate electricity demand is a hard, hard-to-fake nowcast
  of real output**, printed monthly (and, in some regions, near-daily). The trading corollary is
  that *demand-growth* should lead equity or, more mechanically, utility-sector returns.
- **He, Lin & Wei / the electricity–GDP nexus literature** — a large empirical body (e.g. the
  "electricity consumption and economic growth" causality literature spanning dozens of
  countries) establishes that power consumption and GDP are **cointegrated and tightly
  co-move**, with causality running in *both* directions and the lead/lag varying by economy.
  The key nuance this study leans on: the relationship is largely **coincident**, not leading —
  electricity tracks activity contemporaneously more than it forecasts it.
- **EIA *Electric Power Monthly*** — the U.S. Energy Information Administration's monthly release
  of net generation and retail sales by sector, the canonical settled series this study proxies
  (series `ELEC.GEN.ALL-US-99.M`). The release lands ~8 weeks after the reference month — the
  publication lag this study honours with a 2-month signal shift.
- **The "alt-data nowcast" wave** — satellite night-lights, freight, card spend, and power-grid
  load as real-time GDP proxies. The recurring finding on this desk (see the neighbours below) is
  that a coincident macro proxy rarely survives the jump from *nowcast* to *tradable forward
  return*, because the market has already discounted the activity it measures.

## The measure we build

- **Demand-growth momentum** = year-over-year % change of monthly U.S. net generation. The YoY
  transform is deliberate: monthly generation is dominated by a summer air-conditioning peak and
  a winter heating bump (~±15% of the level), which carry no cyclical information; comparing each
  month to the same month a year earlier removes the seasonal and isolates the business-cycle
  pulse. "Hot" = growth above its trailing 24-month median. The signal is lagged 2 months (1 for
  the EIA publication delay, 1 for the standard signal→return convention) so it is strictly
  public at trade time.

## Neighbours on this bench (the dedup map)

- **[Study 385 — Jobless-Claims-Momentum](../../385-jobless-claims-momentum/)** — the closest
  cousin: another *hard-data macro pulse* (initial claims) tested as a leading equity signal,
  with the same conditional-split + placebo + lead/lag machinery. There the pulse turns out to
  *lag* the market; here electricity demand is coincident and its one real reading is a COVID
  artefact. Different series, same "coincident echo dressed as a leader" conclusion.
- **[Study 269 — Baltic-Dry](../../269-baltic-dry/)** — the Baltic Dry shipping index as a
  real-economy leading indicator. Same alt-data-nowcast family (freight vs power); both land
  `Weak`/`Mirage`.
- **[Study 387 — Economic-Surprise-Index](../../387-economic-surprise-index/)** — macro
  data-surprise as an equity signal; the surprise angle rather than a raw level/growth series.
- **[Study 245 — Oil-Equity-Correlation](../../245-oil-equity-correlation/)** /
  **[Study 305 — Gold-Oil-Ratio](../../305-gold-oil-ratio/)** — commodity/energy-price macro
  signals; this study is the *physical-quantity* (grid load) counterpart to those *price* signals.

## Shared method

- **Newey & West (1987)** — the HAC (heteroskedasticity- and autocorrelation-consistent)
  standard errors on the predictive-regression slope, with lags = the forecast horizon to correct
  the overlap when h > 1.
- **Welch (1947)** — the unequal-variance two-sample *t* for the hot-minus-cold forward-return
  split (used only on the non-overlapping 1-month horizon for clean inference).
- **Label-shuffle / permutation testing** (Fisher 1935; Good 2005) — the placebo null: shuffle
  the hot/cold labels against forward returns and read the spread's tail probability.
- House methodology: [`METHODOLOGY.md`](../../../METHODOLOGY.md) — the inference bar (a real-tape
  autocorrelation-robust *t* ≥ 2 for `REAL`, else `WEAK`), one execution lag, gross-vs-net
  labelling, and the seed-robust synthetic control.
