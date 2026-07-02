# References & literature map — Study 555 (OpenTable-Reservations)

## The claim, at full strength

- **OpenTable "State of the Industry" seated-diners dashboard** (2020-2022). During the pandemic
  OpenTable published a widely-cited public dashboard of **year-over-year seated diners** (people
  seated via online reservations, phone and walk-ins at a large sample of restaurants) versus 2019.
  It became the canonical real-time proxy for dining recovery — and the seed of the "reservations
  nowcast" idea: if diners are coming back, restaurant revenue (and stocks) should follow. The feed
  was a moving HTML widget, repeatedly restated, and later discontinued — **not** a stable,
  cache-able CSV/API, and covering only a single non-stationary recovery episode. That is exactly
  why this study is synthetic-only (stated on the SIGNAL axis).

## Alt-data nowcasting — the method the claim rides on

- **Bok, Caratelli, Giannone, Sbordone & Tambalotti (2018)**, *"Macroeconomic Nowcasting and
  Forecasting with Big Data,"* Annual Review of Economics 10. The framework for turning
  high-frequency alt-data flows into a real-time read on slow-moving fundamentals — the general
  form of "reservations nowcast revenue."
- **Croushore (2011)**, *"Frontiers of Real-Time Data Analysis,"* Journal of Economic Literature
  49(1). Why real-time / point-in-time data (and vintage restatements) matter — directly relevant
  to why a restated, discontinued dashboard is not a tradable panel.
- **Da, Engelberg & Gao (2011)**, *"In Search of Attention,"* Journal of Finance 66(5). The
  archetype for "consumer-behaviour alt-data (search volume) nowcasts asset prices" — the same
  logical shape as reservations → restaurant stocks, and a caution on how fragile such signals are
  out of sample.
- **Katona, Painter, Patatoukas & Zeng (2023)** and the broader satellite/foot-traffic alt-data
  literature — geolocation and foot-traffic panels used to nowcast retail revenue. Reservations
  are a foot-traffic cousin; the same edge-decay-once-widely-known caveat applies.

## Why the signal is orthogonalised (the confound we remove)

- A naive regression of restaurant returns on *raw* reservations YoY would mostly rediscover
  **market beta** (both rise in good times). The strategy therefore builds a *surprise* —
  reservations YoY with its own short trend and the contemporaneous market factor removed — so the
  nowcast coefficient measures information **beyond** the market. This mirrors the standard
  alt-data practice of residualising the signal against known factors before claiming incremental
  predictive power.

## Neighbours on this bench (the dedup map)

- **[Study 273 — Lego-Returns](../../273-lego-returns/)**, **[Study 275 — Whisky-Cask](../../275-whisky-cask/)**,
  **[Study 276 — Sneaker-Resale](../../276-sneaker-resale/)** — the desk's other **synthetic-only**
  studies, where the real free data does not exist and the honest move is to prove the machinery on
  a seeded synthetic world and cap the Signal axis below `REAL`. Study 555 is the alt-data-**nowcast**
  instance of that pattern (a *predictive* weekly regression, not a long-horizon collectible index).
- Other alt-data / sentiment nowcast studies on the bench test whether an external attention/traffic
  proxy leads the tape; this one is specifically **dining reservations → restaurant basket / XLY**,
  with the OpenTable seated-diners feed as the (unavailable) real source.

## Shared method

- **Newey & West (1987)** — the heteroskedasticity- and autocorrelation-consistent (HAC) standard
  errors used for the predictive-regression slope *t* (overlapping weekly forward returns make plain
  OLS *t* too optimistic).
- **Label-shuffle / permutation testing** (Fisher 1935; Good 2005) — the placebo null: shuffle the
  reservations surprise against the forward returns and read the slope-*t*'s tail probability.
- House methodology: [`METHODOLOGY.md`](../../../METHODOLOGY.md) — the inference bar (a HAC *t* ≥ 2
  on a **real** tape for `REAL`; literature-plausible-but-untestable reads `WEAK`), one execution
  lag, gross AND net labelled, shorts pay borrow, and the ≥ 20-seed synthetic-control rule.
