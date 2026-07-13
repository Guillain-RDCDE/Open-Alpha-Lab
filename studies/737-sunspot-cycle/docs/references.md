# References & literature map — Study 737 (Sunspot-Cycle)

## The claim under test

- **The original.** **W. S. Jevons (1878)**, "Commercial Crises and Sun-Spots"
  (*Nature*, vol. 19; and his 1875/1878 addresses to the British Association) proposed
  that the ~11-year sunspot cycle drives terrestrial weather → harvests → commodity
  prices → the commercial (trade) cycle, and hence markets. It is one of the earliest
  explicit "exogenous physical cycle drives the economy" theories, and the direct
  ancestor of every "solar-cycle investing" newsletter since. The harvest-transmission
  chain was already contested in Jevons's lifetime and is not defended today; what
  survives in market folklore is the reduced-form headline — **equity returns run on an
  ~11-year solar clock** (active Sun → good returns).
- **The modern restatements.** The claim recurs in popular market-cycle writing and in a
  thin academic literature that periodically re-tests it. Representative peer-reviewed
  entries: **Robotti & Krivelyova / Krivelyova & Robotti (2003, Federal Reserve Bank of
  Atlanta WP 2003-5), "Playing the Field: Geomagnetic Storms and the Stock Market"** —
  a *different* solar-activity channel (geomagnetic disturbance → human mood → returns),
  but the closest rigorous cousin; and periodic notes testing sunspot number directly
  against index returns (e.g. work summarised in the "calendar/space-weather anomalies"
  reviews). The consistent finding across the credible literature is **no robust,
  tradable 11-year return cycle** — this study reproduces that on the longest S&P tape.
- **Why it is an unusually clean thing to test.** The sunspot cycle is **exogenous and
  known in advance** — set by solar dynamo physics, not by anything traders do — so
  there is no reverse-causation and no date-mining escape hatch. A null here is not "we
  couldn't find the trade"; it is "the 11-year period is not in the returns."

## The solar data

- **WDC-SILSO, Royal Observatory of Belgium** — the *Sunspot Index and Long-term Solar
  Observations* centre, custodian of the international sunspot number. Cycle minima /
  maxima dates and smoothed peak amplitudes (version-2 series) are taken from SILSO's
  published "Sunspot cycles data" and cross-checked against **NOAA Space Weather
  Prediction Center (SWPC)** "Solar Cycle Progression". The v2 recalibration is
  documented in **Clette, Svalgaard, Vaquero & Cliver (2016), "Revisiting the Sunspot
  Number", *Space Science Reviews* 186**.
- Only the **turning-point dates and peak amplitudes** are hardcoded (cycles 16–25); the
  monthly series used in the tests is a **labelled cosine reconstruction** pinned to
  those turning points, standing in for the *phase* of the cycle — explicitly a proxy,
  not the raw daily SILSO file (the same labelled-proxy discipline as
  [`358-watch-index`](../../358-watch-index/) and [`708-eurovision-effect`](../../708-eurovision-effect/)).

## What we measure, and the honesty rails

- **Constant-mean market model** (Brown & Warner 1985): the abnormal return is the
  monthly return demeaned by its own full-sample mean, so a "solar" effect is measured
  against the market's ordinary up-drift rather than on top of it.
- **Two units of analysis, on purpose.** Turning points are **independent events**
  (≥ 5 yr apart), so the forward-return study uses a **one-sample / Welch *t* across
  events** plus a **random-calendar placebo** per group — never a HAC panel regression.
  The **monthly panel** is autocorrelated and the regime label is persistent, so the
  phase regression carries **Newey-West (HAC, 12-lag)** *t*'s and the regime spread a
  **circular block bootstrap** (12-month blocks), with a **phase-shift placebo** (roll
  the whole solar calendar to a random start, preserving its 11-year shape but breaking
  its alignment). Hit rates carry **Wilson (1927)** intervals.
- **One documented lag.** The solar-clock timer acts on the phase **lagged 6 months** —
  the SILSO smoothing lag, because a turning point is only *known* to be one once the
  smoothed sunspot number is published. The forward-return event study, by contrast,
  uses the *true* turning-point dates and is **flagged generous / retrospective**: it
  gives the folklore perfect hindsight of the cycle you would never have live, and it
  *still* finds nothing.
- **Costs & labelling.** Timer costs are one-way × NAV per switch; the overlay is
  long-or-flat (no shorting, no borrow), cash earns nothing (a single explicit
  assumption — no risk-free double count). The tape is **price-only** and labelled so
  everywhere; gross and net timer figures are both reported.

## Data sources

- **^GSPC** (S&P 500 **price** index) daily close, `auto_adjust=False` (a price index
  has no dividend to reinvest) — yfinance (no key), cached under `_cache/sunspot_gspc.csv`,
  1927-12-30 → 2026-06-30.
- **Solar cycles 16–25** (turning points + peak amplitudes) hardcoded in
  [`sunspot_cycle/data.py`](../sunspot_cycle/data.py) from SILSO / NOAA-SWPC.
- All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is NOT)

- [279-geomagnetic-storms](../../279-geomagnetic-storms/) — the Krivelyova & Robotti
  (2003) channel: **geomagnetic disturbance → human mood → returns**, a *physiological*
  hypothesis on short (days) horizons. Same star, entirely different mechanism and time
  scale; this study tests the **11-year sunspot-cycle → returns** (Jevons) macro claim,
  not a mood channel.
- [280-solar-eclipse](../../280-solar-eclipse/) — an event-day *superstition* around
  eclipses, not the multi-year activity cycle.
- [278-sunshine-effect](../../278-sunshine-effect/) (Hirshleifer & Shumway 2003) and
  [150-sad-effect](../../150-sad-effect/) (Kamstra, Kramer & Levi 2003) — daylight /
  seasonal-mood channels. Same "the sky moves markets" family, but weather/daylight
  mood, not the sunspot cycle.

No sibling tests the specific **11-year sunspot-cycle → equity-return** (Jevons) axis —
that is this study's own.
