# References & literature map — Study 309 (OJ-Frost)

## The claim under test — the *Trading Places* trade

- **The film.** *Trading Places* (1983, dir. John Landis). The climax turns on cornering
  the frozen-concentrate orange-juice (FCOJ) futures market ahead of a (falsified) USDA
  crop report: a freeze would devastate the Florida crop, OJ futures would spike, and
  whoever was positioned first would clean up. The film cemented "OJ + freeze = the trade"
  in popular finance folklore — and FCOJ futures have been the canonical academic example
  of a weather-driven commodity ever since.
- **OJ futures and the weather.** Roll (1984), *Orange Juice and Weather* (American
  Economic Review). The foundational study: FCOJ futures prices react to freezing-weather
  forecasts in the Florida citrus belt, and the futures market even contains information
  about future temperature beyond the National Weather Service forecast. This is the
  steelman — there *is* a real, documented freeze-to-price channel. The question this study
  asks is narrower and harsher: **on the modern, tradable `OJ=F` tape, around the specific
  hard-freeze dates we can list, was there a window you could have bought and made money?**
- **Boudoukh, Richardson, Shen & Whitelaw (2007)**, *Do Asset Prices Reflect Fundamentals?
  Freshly Squeezed Evidence from the OJ Market* (Journal of Financial Economics). A
  follow-up to Roll that finds OJ futures' reaction to temperature is real but that a large
  fraction of price moves are *not* explained by fundamentals — i.e. the freeze signal is
  noisy and partly behavioural, which is exactly why a naive event trade can disappoint.

## Why the trade is so hard in practice

- **Data availability / survivorship of the lore.** The famous freezes that built the
  folklore — the January 1977 snow in Miami, the 1981–1985 cluster, the 1983 "Christmas
  freeze," the 1989 freeze — all predate the Yahoo `OJ=F` continuous-contract tape (which
  begins in 2001). The trade is remembered through events no modern retail data feed
  contains. Naming this is the central methodological point: the sample that built the
  belief is not the sample you can test or trade.
- **Crop-damage uncertainty and the "buy the rumour" problem.** A freeze *forecast* moves
  the market before the cold night; by the time damage is confirmed the move is often done
  or reversing (a freeze that turns out milder than feared sells off). A reactive trade that
  enters *after* the freeze is systematically late — the look-ahead-free version of the
  trade is structurally disadvantaged. Hence the lag=0 "perfect foresight" ceiling in the
  results: it bounds how much was ever there to capture.
- **Microstructure of a thin contract.** FCOJ is one of the smallest, widest-spread futures
  markets. Even a real edge faces large round-trip costs and minimal capacity — the
  Tradability axis is constrained before the Signal axis even reports.

## Seasonality

- **Commodity seasonality, generally.** Commodities with weather-dependent supply often
  show calendar seasonality in price and volatility; for OJ the natural hypothesis is a
  Dec–Feb "freeze-risk premium." We test the simplest version (mean DJF daily return vs the
  rest of the year, HAC *t* on the difference) and find it absent on this tape — a useful
  null against the more elaborate seasonal stories.

## Method lineage (the desk's shared engine)

- **Event-study methodology.** MacKinlay (1997), *Event Studies in Economics and Finance*
  (Journal of Economic Literature) — the abnormal-return-around-an-event design and its
  small-sample fragility. With only four in-tape events the cross-event *t* is the textbook
  case where the asymptotics simply do not apply; we say so rather than quoting it.
- **HAC / Newey-West t-stat.** Newey & West (1987), *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix* (Econometrica) —
  [`strategy.hac_tstat`](../oj_frost/strategy.py).
- **Block bootstrap.** Politis & Romano (1994), *The Stationary Bootstrap* (JASA), and the
  circular-block variant — resampling i.i.d. would destroy the serial dependence the
  inference must respect. [`strategy.block_bootstrap_ci`](../oj_frost/strategy.py).
- **Placebo / random-date control.** The "same window on random non-event dates"
  distribution nets out OJ's ambient drift and the window length, so the freeze excess is
  measured against the market's own background, not zero.

## Data sources used here

- **Yahoo! Finance daily close for `OJ=F`** (via `yfinance`), continuous front-month FCOJ
  future, auto-adjusted, history from 2001-09. Freeze dates: the hardcoded
  [`FREEZE_EVENTS`](../oj_frost/data.py) table of severe documented Florida citrus-belt
  freezes. As-of 2026-05-31; the partial June 2026 month is dropped; headline numbers are
  pinned with a content fingerprint (see [`docs/results.md`](results.md)). The offline
  reproducible core and the test-suite run on the deterministic
  [`data.synthetic_oj`](../oj_frost/data.py) generator, never the network.

## Related desk studies

- **[Study 35 — Contango](../../35-contango/)**: commodity carry / roll yield — the other
  half of "is there money in commodity futures," and another Weak/Mirage.
- **[Study 29 — Hedgers-Toll](../../29-hedgers-toll/)**: are speculators paid for taking the
  other side of producers' hedges — the structural-premium cousin to a one-off event trade.
- **[Study 281 — El-Nino](../../281-el-nino/)**: another weather-to-markets folklore trade
  (does ENSO move equities/commodities), also stamped None/Mirage. Weather narratives are a
  recurring source of plausible-but-untradable lore on this desk.
