# References & literature map -- Study 278 (Sunshine-Effect)

## The canonical claim

- **Hirshleifer, D. & Shumway, T. (2003).** *Good Day Sunshine: Stock Returns
  and the Weather.* The Journal of Finance, 58(3), 1009--1032.
  The founding paper. Using morning cloud-cover data for 26 international stock
  exchanges (1982--1997), the authors find that sunshine is significantly and
  positively associated with same-day stock returns, consistent with a
  psychological "mood misattribution" channel. Critically, they also report that
  a weather-based trading strategy does **not** beat buy-and-hold after
  transaction costs -- the effect is statistically detectable but not
  economically exploitable. This study reproduces both halves of that finding
  for NYC and the S&P 500.

- **Saunders, E. M. (1993).** *Stock Prices and Wall Street Weather.* American
  Economic Review, 83(5), 1337--1345.
  The precursor: New York City cloud cover is negatively related to NYSE and
  AMEX index returns, 1927--1989. Saunders established the NYC-specific result
  that Hirshleifer-Shumway later globalised. Our NYC focus follows Saunders'
  exchange-city design.

## Mechanism: mood and misattribution

- **Schwarz, N. & Clore, G. L. (1983).** *Mood, Misattribution, and Judgments of
  Well-Being: Informative and Directive Functions of Affective States.* Journal
  of Personality and Social Psychology, 45(3), 513--523.
  The psychology behind the claim: people misattribute weather-induced mood to
  unrelated judgments (including, the finance literature argues, asset values).
  This is the proposed transmission channel from sky to price.

- **Kamstra, M. J., Kramer, L. A. & Levi, M. D. (2003).** *Winter Blues: A SAD
  Stock Market Cycle.* American Economic Review, 93(1), 324--343.
  A sibling mood anomaly: seasonal affective disorder (daylight-driven) and
  equity returns. Same behavioural family as the sunshine effect; same caveats
  about economic significance after costs.

## Skeptics and robustness

- **Goetzmann, W. N. & Zhu, N. (2005).** *Rain or Shine: Where is the Weather
  Effect?* European Financial Management, 11(5), 559--578.
  Find that individual investors do not trade differently on cloudy vs sunny
  days, casting doubt on the behavioural channel and suggesting the index-level
  result may be driven by market-maker behaviour or be partly spurious.

- **Jacobsen, B. & Marquering, W. (2008).** *Is it the Weather?* Journal of
  Banking & Finance, 32(4), 526--540.
  Argue that several weather-return correlations are fragile and may reflect
  data-snooping or seasonal confounds rather than a robust mood effect --
  motivating our explicit de-seasonalisation of the cloud series.

## Method lineage

- **HAC / Newey-West t-stat.** Newey, W. K. & West, K. D. (1987), *A Simple,
  Positive Semi-Definite, Heteroskedasticity and Autocorrelation Consistent
  Covariance Matrix* (Econometrica, 55(3), 703--708). Daily index returns are
  mildly autocorrelated and heteroskedastic; we use the Bartlett-kernel HAC SE
  (rule-of-thumb lags `floor(4*(n/100)^(2/9))`) on both the regression slope and
  the sunny-cloudy mean contrast. This is the right standard error for the Real
  signal bar (|t| >= 2).
- **De-seasonalisation.** Each calendar month's mean cloud cover is removed
  before testing, so a cloudy-winter pattern cannot masquerade as a weather
  effect (addresses Jacobsen-Marquering's seasonal-confound critique).
- **Costed sleeve.** One-way cost x change-in-exposure x NAV; shorts would pay
  borrow but the sleeve here is long-or-flat, so no borrow is charged. Gross and
  net both reported; the breakeven cost is sub-basis-point.

## Data sources

- **`^GSPC` daily close.** S&P 500 daily total-return-adjusted close via
  yfinance, cached at `_cache/gspc_daily.parquet` (cache-only by default; the
  network is touched only on an explicit `fetch=True`). Used as the NYSE proxy.
- **NYC sky-cover climatology.** Hardcoded month-by-month mean and standard
  deviation of daily sky cover (octas) in `data.py`, from the NOAA/NWS New York
  Central Park (KNYC) U.S. Climate Normals (1991--2020). The trading-day cloud
  series is a *deterministic climatological reconstruction* drawn from these
  normals (seeded by date), NOT a hand-collected station log -- so the real-tape
  result is an **upper bound** on a live, station-fed strategy, and this is named
  on the Signal axis.

## Related desk studies

- **Mood / calendar siblings:** SAD (seasonal-affective) effect, the lunar-cycle
  effect, and daylight-saving-time return dips -- small behavioural signals in
  the same family that rarely survive costs.
- **[Study 158 -- Super-Bowl](../../158-super-bowl/)** and
  **[Study 223 -- Same-Month-Seasonality](../../223-same-month-seasonality/)**:
  other behavioural/calendar effects tested against the honest baseline and the
  cost gauntlet.
