# References — Study 639 (Gasoline RVP Seasonality)

## The claim's source — the law itself

- **EPA, "Gasoline Reid Vapor Pressure"** — the regulatory calendar under test: federal summer
  volatility standards (9.0 / 7.8 psi RVP) apply **May 1 – September 15** at refineries and
  terminals and **June 1 – September 15** at retail; winter blends (cheap, butane-rich,
  high-RVP) are legal outside that window.
  <https://www.epa.gov/gasoline-standards/gasoline-reid-vapor-pressure>
- **40 CFR § 1090.215** (ex-§ 80.27) — the codified summer-gasoline volatility standards and
  their dates. <https://www.ecfr.gov/current/title-40/chapter-I/subchapter-U/part-1090>
- **EIA, "Gasoline explained: factors affecting gasoline prices"** — the plain-language version:
  summer-grade gasoline costs more to make, and the spring specification transition (plus
  refinery maintenance timed to it) lifts gasoline prices relative to crude every year.
  <https://www.eia.gov/energyexplained/gasoline/factors-affecting-gasoline-prices.php>

## Instruments

- **CME Group — RBOB Gasoline futures (RB) contract specs** — deliverable grade switches with
  the season: summer-spec RBOB is deliverable for May–September contracts, which is exactly how
  the RVP calendar gets *into the futures curve* months in advance.
  <https://www.cmegroup.com/markets/energy/refined-products/rbob-gasoline.html>
- **USCF, United States Gasoline Fund (UGA)** — the investable front-month holder used for the
  curve test: UGA holds near-month RBOB futures and rolls them each month, paying the calendar
  spread (~0.75%/yr expense ratio; collateral in T-bills).
  <https://www.uscfinvestments.com/uga>

## Key papers

- **Girma, P. B. & Paulson, A. S. (1999), "Risk arbitrage opportunities in petroleum futures
  spreads,"** *Journal of Futures Markets* 19(8) — documents systematic seasonality in the
  gasoline crack spread.
- **Borovkova, S. & Geman, H. (2006), "Seasonal and stochastic effects in commodity forward
  curves,"** *Review of Derivatives Research* 9 — the third axis's theory: for seasonal
  commodities the *forward curve itself* carries the seasonal, so calendar effects visible in
  spot need not be earnable by a futures holder.
- **Hartzmark, M. — no relation** to gasoline; the desk's canonical "calendar you can see in
  advance" replication is [516-dividend-month-premium](../../516-dividend-month-premium/), where
  the predictable calendar *is* capturable — the instructive contrast to this study, where the
  curve prices it away.

## Data sources

- Yahoo! Finance via `yfinance` (no key): `RB=F` (RBOB front-month chain, 2005+), `CL=F` (WTI
  front-month chain, the control leg), `UGA` (total-return ETF, 2008+), `^IRX` (13-week T-bill
  yield). Spliced-chain caveat: front-month chains include roll jumps no holder earns — treated
  explicitly as a **spot-price proxy** and paired against `UGA`, the real holder.

## Named siblings (the dedup guard)

- [**226-crude-seasonality**](../226-crude-seasonality/) — WTI **outright** calendar months
  (does crude itself rally in spring?). This study is the **spread**: gasoline *over* crude, so
  the oil-price level nets out and only the blend-spec calendar remains.
- [**306-crack-spread**](../306-crack-spread/) — the crack **level** as a timing signal for
  refiner *stocks* (coincident vs predictive). This study never uses the level: it tests the
  **dated RVP calendar** on the gasoline-crude *return spread* and whether the futures curve
  pre-prices it.

## Shared method

- Welch, B. L. (1947), "The generalization of 'Student's' problem when several different
  population variances are involved," *Biometrika* 34 — the across-years group split.
- Desk house rules: [`METHODOLOGY.md`](../../../METHODOLOGY.md) (inference bar, one-lag rule,
  excess-vs-excess races, synthetic controls as machinery proofs only).
