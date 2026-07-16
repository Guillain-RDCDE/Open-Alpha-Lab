# References & literature map — Study 787 (Heatwave-Utilities)

## The claim under test

- **The folklore.** "Own the utilities through the peak of summer — the hottest weeks of
  the year drive record air-conditioning load, electricity demand spikes, and utility
  revenues (and utility stocks) rally with it." A perennial retail / financial-media
  seasonality trade for the defensive Utilities sector (XLU), told as a fundamental
  cooling-demand story.
- **Why it's a clean calendar test.** The window is **known in advance** — peak US heat is
  a fixed climatological feature, not an announced event — so "hold XLU across the peak-heat
  weeks" is calendar-known and zero-look-ahead by construction. Because of the ~4-5 week
  *seasonal temperature lag*, the hottest average temperatures across the contiguous US fall
  in mid-to-late July, not on the June-21 solstice; we anchor every year on a fixed
  **July 22** ([`data.py`](../heatwave_utilities/data.py)), a published climatology
  convention rather than a data-mined "hottest day of that year."
- **The efficient-markets prior.** A seasonal demand pattern this well-known and this
  regular is exactly what a semi-strong-efficient market should already price into the
  sector — see Fama (1970, *Efficient Capital Markets*, JF). The desk's prior is that any
  cooling-demand premium is arbitraged away long before the heat arrives.

## What the literature actually says

- **Weather, temperature and electricity load.** There is a large, robust engineering /
  economics literature linking temperature to electricity demand via heating- and
  cooling-degree-days (e.g. Engle, Granger, Rice & Weiss, 1986, *JASA*, on nonlinear
  temperature–load models; and the EIA's cooling-degree-day demand framework). This
  establishes that *load* rises with summer heat — but load is not utility *profit*
  (regulated rates, fuel pass-throughs, decoupling), and utility profit is not the same as
  utility *stock* abnormal return.
- **Weather and stock returns.** Saunders (1993, *AER*) and Hirshleifer & Shumway (2003,
  *JF*, "Good Day Sunshine") find weather/sunshine correlates with market returns via mood,
  not fundamentals — a different mechanism from a sector cooling-demand story and a famously
  fragile, contested effect.
- **Calendar & seasonal anomalies.** The "Sell in May / Halloween indicator" (Bouman &
  Jacobsen, 2002, *AER*) and the broad seasonality-anomaly record show that most published
  calendar effects are weak, sample-specific, and shrink out of sample — the right prior for
  a summer utilities pattern.
- **Sector-rotation seasonality.** Practitioner "seasonal sector rotation" lore (e.g.
  Stock Trader's Almanac, Hirsch) places defensives like utilities as a summer/autumn hold;
  the academic support for tradable sector-calendar timing net of costs is thin.

## Data & method

- **Real tape:** `XLU` (Utilities Select Sector SPDR, inception Dec 1998) and `SPY` daily
  adjusted (total-return) closes via [yfinance](https://github.com/ranaroussi/yfinance),
  one combined panel. XLU's low (~0.3-0.5) beta to SPY is why we measure the *abnormal*
  return `XLU − SPY`, netting out a summer that was simply a strong tape for everything.
- **Statistics:** one-sample *t* of the abnormal return across independent, non-overlapping
  summers (the correct unit — not a daily panel); Wilson hit-rate interval; a 20-seed ×
  200-draw random-window placebo per cut; a leave-one-out jackknife; a costed net leg.
- **Synthetic positive control:** a seeded paired (asset, benchmark) world with a *planted*
  into-the-heat run-up (and optional post-peak fade) — the detector must recover a planted
  bump and stay quiet on the null. See [`strategy.py`](../heatwave_utilities/strategy.py).

*Fama, E. (1970). Efficient Capital Markets. **Journal of Finance**. · Engle, R., Granger,
C., Rice, J. & Weiss, A. (1986). Semiparametric Estimates of the Relation Between Weather
and Electricity Sales. **JASA**. · Saunders, E. (1993). Stock Prices and Wall Street
Weather. **American Economic Review**. · Hirshleifer, D. & Shumway, T. (2003). Good Day
Sunshine. **Journal of Finance**. · Bouman, S. & Jacobsen, B. (2002). The Halloween
Indicator. **American Economic Review**.*
