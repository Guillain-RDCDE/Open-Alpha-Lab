# References & literature map — Study 758 (TSA-Throughput)

## The claim under test

- **TSA checkpoint volumes as a real-time travel-demand nowcast.** The Transportation
  Security Administration publishes the **number of travellers screened at U.S. airport
  checkpoints every day**, next-day, for free — TSA, *TSA checkpoint travel numbers*
  (`https://www.tsa.gov/travel/passenger-volumes`). During 2020–2021 this became the single
  most-watched high-frequency gauge of the travel recovery: it fell ~95% in April 2020 (to
  ~0.09M/day) and its climb back was quoted daily as a proxy for airline and hotel demand.
- **The trading folklore.** From "TSA is the real-time read on travel" follows the market-
  timing leap we test: because throughput is published *before* airline traffic reports (ASMs/
  RPMs), hotel RevPAR, or quarterly earnings, an **acceleration** in TSA volumes is read as an
  **early, tradable tailwind** for the travel basket — airlines (`JETS`) and hotels
  (`MAR`, `HLT`). This is the alt-data / "nowcasting" thesis popularised for card-spend,
  satellite-parking-lot and web-traffic data and applied to TSA in macro-tactical commentary
  and fintwit "reopening trade" threads. We test the strongest form: *does a TSA-momentum
  uptick lead the travel trade cleanly enough, outside the COVID regime, to trade?*
- **The data series.** TSA, *Daily checkpoint travel numbers* (2019–present), aggregated here
  to a **monthly average of daily throughput** (millions). It is the standard, widely-quoted
  high-frequency travel gauge; only a post-2019 history exists (TSA began publishing the daily
  comparison in 2019), which is itself a limitation of the nowcast — no pre-COVID business
  cycle to test against.

## Why TSA data isn't fetched live here — and what we do

- **TSA site not reachable / no tidy CSV.** TSA's passenger-volume page renders the daily
  numbers without a stable free CSV endpoint and is not reachable from this build's network
  sandbox. Following the desk convention for small, public, high-frequency alt-data series —
  **Study 385 (Jobless-Claims)** hardcodes a FRED `IC4WSA` snapshot, **Study 358 (Watch-Index)**
  and **Study 708 (Eurovision-Effect)** hardcode a cited alt-data table — we **hardcode a
  monthly snapshot** of the average daily throughput (millions), rounded, as-of 2026-06-30, as
  a **LABELLED PROXY** (never under a real-tape banner). The COVID collapse-and-reopen is
  included faithfully.
- **Equities.** `JETS`, `MAR`, `HLT` and `SPY` daily adjusted close via **yfinance** (no key),
  month-end sampled, total-return adjusted — labelled as such. The travel basket is an
  equal-weight ½ airlines · ½ hotels total-return index.

## Why "leading / real-time" is the crux — coincident vs lagging confusion

- **Reference-cycle dating and lead/lag.** Burns & Mitchell (1946), *Measuring Business Cycles*
  (NBER) — the original classification of series into leading, coincident and lagging at
  cyclical turns. A series can co-move strongly with a boom yet **lag** the equity market that
  discounts it first. We run an explicit **lead/lag cross-correlation** to locate where TSA
  momentum actually sits relative to travel-basket returns.
- **The stock market as a discounting mechanism.** Equity prices lead the real economy and
  react to *news* about the recovery well before physical activity confirms it (Samuelson's
  quip that the market "predicted nine of the last five recessions"). So a physical-activity
  series that lines up with *contemporaneous* travel-stock strength need not **lead** it — it
  may merely echo a re-rating the market already made (e.g. the Nov-2020 vaccine rally, months
  before TSA throughput recovered). This is the confound the study isolates.
- **Alternative data and alpha decay.** Monteiro & Zaman and others on "nowcasting with
  alternative data" document that high-frequency alt-data (card spend, foot traffic, TSA) is
  genuinely informative about *contemporaneous fundamentals* yet often carries little
  *tradable* equity alpha once the market already prices the trend. Katona, Painter, Patatoukas
  & Zeng (2018), *On the Capital Market Consequences of Alternative Data* — satellite
  parking-lot data predicts retail fundamentals but the price impact is largely arbitraged by
  sophisticated traders, so the marginal user earns little. The same caution applies to a
  free, universally-watched series like TSA throughput.
- **Predictive regressions and small-sample / out-of-sample caution.** Welch & Goyal (2008),
  *A Comprehensive Look at the Empirical Performance of Equity Premium Prediction* (Review of
  Financial Studies) — most predictors that look significant in-sample fail out-of-sample; the
  bar for a tradable signal is high, and a 7-year, one-regime sample is especially fragile.

## Why the inference is small-sample / placebo-based

- **Welch two-sample t.** Welch (1947), *The generalization of "Student's" problem when several
  different population variances are involved* (Biometrika) — unequal-variance test of the
  ACCELERATING-set forward mean against the unconditional mean.
- **Randomization / placebo null.** Because regime months are autocorrelated and the effective
  sample is small (90 months, one dominant COVID regime), we resample random same-size month
  sets and ask how often chance is as bullish as the ACCELERATING set (Fisher's randomization
  logic; Efron & Tibshirani, *An Introduction to the Bootstrap*, 1993).
- **Overlapping long-horizon returns.** The 12-month forward returns overlap heavily; classical
  OLS *t*-stats on them are inflated by ≈√12 (Hansen & Hodrick, 1980; Britten-Jones, Neuberger
  & Nyholm, 2011). We flag the one inflated statistic (the 12-month beta-control dummy)
  explicitly and lean on the placebo *p* for the honest read.
- **One coincident regime dominates.** The COVID-2020 collapse-and-reopen is one enormous
  coincident event; we report results with and without Jun-2020 → May-2022 so the verdict
  doesn't ride on a single regime.

## Method lineage (this study's engine)

- **Signal + inference.** [`strategy.tsa_momentum`](../tsa_throughput/strategy.py),
  [`strategy.summarize`](../tsa_throughput/strategy.py) (Welch *t* + placebo *p*),
  [`strategy.lead_lag`](../tsa_throughput/strategy.py) (the nowcast/leading test),
  [`strategy.beta_control`](../tsa_throughput/strategy.py) (is it beyond market beta?),
  [`strategy.timing_overlay`](../tsa_throughput/strategy.py) (long-travel-when-accelerating,
  one-month lag, one-way costs, borrow on shorts).
- **Deterministic synthetic control.**
  [`data.synthetic_tsa`](../tsa_throughput/data.py) plants a known TSA-momentum→returns link;
  `edge = 0` must not manufacture significance, a large `edge` must light up the test.

## Data sources used here

- **TSA checkpoint throughput** (hardcoded monthly snapshot, millions/day, LABELLED PROXY) +
  **yfinance JETS/MAR/HLT/SPY** daily adjusted close, 2019-01 → 2026-06, cached under
  `_cache/travel_prices.csv`. All headline numbers are pinned in [`docs/results.md`](results.md)
  and reproduced by [`examples/verify.py`](../examples/verify.py).

## Related desk studies

- **[Study 385 — Jobless-Claims-Momentum](../385-jobless-claims-momentum/)**: the sibling
  macro-nowcast done the same way (hardcoded snapshot + a market tape, lead/lag scan, timing
  overlay) — a famous "leading" labour series that turns out coincident-to-lagging.
- **[Study 358 — Watch-Index](../358-watch-index/)** and
  **[Study 708 — Eurovision-Effect](../708-eurovision-effect/)**: the LABELLED-PROXY
  hardcoded-alt-data pattern this study follows.
- **[Study 387 — Economic-Surprise-Index](../387-economic-surprise-index/)**: another
  macro-beats gauge asking whether a celebrated nowcast actually times equities.
