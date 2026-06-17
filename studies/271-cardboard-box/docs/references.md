# References & literature map — Study 271 (Cardboard-Box)

## The claim under test

**"Dr. Cardboard" / the box-and-freight leading indicator.** A piece of market
folklore, recurring in financial journalism and on trading desks: because almost
every manufactured and retail good moves inside a corrugated box and on a freight
train, the year-over-year growth in **box (containerboard) shipments** and **rail
carloads** is a clean, early read on the real economy — and therefore on the
stock market. The colloquial name mirrors the better-known "Dr. Copper." The
testable version: *this year's box/freight growth forecasts next year's S&P 500
return.*

## Why the story is seductive — and why it usually fails as a forecast

- **Coincident, not leading.** Box and rail output track GDP in *real time*. The
  stock market, by contrast, is forward-looking and has already discounted
  expected GDP. A coincident gauge can confirm where the economy *is*; it rarely
  tells you where the *market* is going, because the market moved first. This is
  the central distinction the study makes (a same-year regression vs a one-year-
  lagged forecasting regression).

- **The market's upward drift.** U.S. equities rise in roughly three-quarters of
  calendar years unconditionally. Any rule that is "long in good years" inherits
  that drift; the honest benchmark is buy-and-hold net of costs, not a coin.

- **Tiny n.** Annual macro data gives ~55 observations over 1970–2024. With ~16%
  annual equity volatility and a single regressor, only a fairly large forecasting
  correlation (|r| ≳ 0.37) is detectable at 80% power — a high bar that spurious
  slopes rarely clear, and that a genuinely weak signal cannot.

## The family of physical-flow indicators

- **The Baltic Dry Index (BDI).** Dry-bulk shipping rates, widely cited as a
  leading indicator of global trade and growth. Like cardboard, it is largely
  coincident-to-slightly-leading for the economy and a poor forecaster of equity
  returns at horizons traders care about.
- **"Dr. Copper."** Copper demand as an economic barometer. Same logic, same
  limitation: a real-economy gauge that is mostly already in equity prices.
- **Trucking tonnage / Cass Freight Index / ATA Truck Tonnage.** Freight volumes
  as activity proxies — coincident indicators in the Conference Board sense.

## Method lineage

- **Predictive regression with HAC errors.** Regress $r_{t+1}$ on $g_t$ and read
  the slope's t-stat under a **Newey-West** (Bartlett-kernel) heteroskedasticity-
  and-autocorrelation-consistent covariance.
  - Newey, W. K. & West, K. D. (1987). "A Simple, Positive Semi-Definite,
    Heteroskedasticity and Autocorrelation Consistent Covariance Matrix."
    *Econometrica*, 55(3), 703–708.
- **Predictive-regression caveats.** Small-sample bias and the difficulty of
  detecting weak predictability are well documented:
  - Stambaugh, R. F. (1999). "Predictive Regressions." *Journal of Financial
    Economics*, 54(3), 375–421.
  - Goyal, A. & Welch, I. (2008). "A Comprehensive Look at the Empirical
    Performance of Equity Premium Prediction." *Review of Financial Studies*,
    21(4), 1455–1508. Many "predictors" fail out of sample.
- **Coincident vs leading indicators.** The Conference Board's Business Cycle
  Indicators framework formalises the distinction; freight/production measures sit
  in the coincident and lagging composites, not the leading one.
- **Data-snooping discipline.** Harvey, C. R., Liu, Y. & Zhu, H. (2016).
  "… and the Cross-Section of Expected Returns." *Review of Financial Studies*,
  29(1), 5–68 — the t ≥ 2 bar is a floor, not a pass mark, given how many macro
  series get tested.

## Data sources

- **Box / containerboard shipments.** Fibre Box Association *Annual Report* box-
  shipment series and the American Forest & Paper Association (AF&PA)
  containerboard statistics. The YoY growth series in `data.py` is a curated,
  rounded reconstruction (one decimal); the *direction and rough magnitude* per
  year — which is all the folklore actually trades on — are what the study tests.
- **Rail freight carloads.** Association of American Railroads (AAR), *Rail Time
  Indicators* / weekly and annual carload reports (Class-I total carloads).
- **S&P 500 (^GSPC).** Daily closes from Yahoo Finance via `yfinance`, aggregated
  to December-to-December **price** returns (price-only, dividends excluded),
  cache-only by default at `_cache/gspc_annual.parquet`.

## Related desk studies

- Other macro/folklore leading-indicator teardowns in the lab share this
  coincident-mistaken-for-leading structure; see the bench map for the family of
  "real-economy gauge as market timer" studies.
