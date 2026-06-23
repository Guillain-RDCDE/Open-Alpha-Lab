# References & literature map — Study 385 (Jobless-Claims-Momentum)

## The claim under test

- **Claims as a leading indicator.** Initial unemployment-insurance claims are a component
  of The Conference Board's **Leading Economic Index (LEI)** — "average weekly initial
  claims for unemployment insurance" enters with a negative weight, the textbook statement
  that *rising* claims signal a *weakening* economy ahead. The Conference Board, *Description
  and Technical Notes: The Conference Board Leading Economic Index® (LEI) for the U.S.*
- **The market-timing folklore.** From this the trading lore follows: because claims "lead,"
  an uptick in the 4-week-MA of initial claims is read as an **early-warning for stocks** —
  get defensive when claims turn up. The idea is repeated across macro-tactical newsletters,
  CNBC/Bloomberg recession-watch segments, and "claims are the most important weekly number"
  commentary. We test the strongest form: *does a claims-momentum uptick lead equity
  downturns cleanly enough to trade?*
- **The data series.** U.S. Department of Labor, Employment & Training Administration,
  *Unemployment Insurance Weekly Claims Report*; **Initial Claims, 4-Week Moving Average,
  Seasonally Adjusted** (FRED series **`IC4WSA`**). The 4-week MA is the standard
  noise-smoothed gauge the nowcasting literature and the financial press actually quote.

## Why claims data isn't fetched live here — and what we do

- **FRED CSV endpoint firewalled.** The free `fred.stlouisfed.org/graph/fredgraph.csv?id=…`
  endpoint times out in this build's network sandbox (as does the DOL site). Following the
  desk convention for small, public, never-revised macro series — **Study 268 (Sahm-Rule)**
  hardcodes FRED `UNRATE`, **Study 158** hardcodes the Super Bowl table — we **hardcode a
  monthly snapshot** of `IC4WSA` (thousands), as-of 2026-06-22. It is the settled print, not
  the real-time vintage; that survivorship-of-revision caveat is named on the Signal axis.
- **Equities.** SPY daily adjusted close via **yfinance** (no key), month-end sampled,
  total-return adjusted — labelled as such.

## Why "leading" is the crux — coincident vs lagging confusion

- **Reference-cycle dating and lead/lead-lag.** Burns & Mitchell (1946), *Measuring Business
  Cycles* (NBER) — the original classification of series into leading, coincident and
  lagging at business-cycle turns. A series can co-move strongly with the cycle yet **lag**
  the equity market, which itself leads the real economy. We run an explicit **lead/lag
  cross-correlation** to locate where claims momentum actually sits relative to SPME returns.
- **The stock market as its own leading indicator.** Stock prices are themselves a Conference
  Board LEI component and famously lead the real economy (Samuelson's quip that the market
  "predicted nine of the last five recessions"). So a macro series that lines up with
  *contemporaneous* equity weakness need not **lead** the market — it may merely echo a turn
  the market already made. This is the confound the study isolates.
- **Predictive regressions and small-sample caution.** Welch & Goyal (2008), *A Comprehensive
  Look at the Empirical Performance of Equity Premium Prediction* (Review of Financial
  Studies) — most macro predictors that look significant in-sample fail out-of-sample; the
  bar for a tradable macro signal is high. Goyal & Welch's caution applies directly to a
  single famous "leading" series.

## Why the inference is small-sample / placebo-based

- **Welch two-sample t.** Welch (1947), *The generalization of "Student's" problem when
  several different population variances are involved* (Biometrika) — unequal-variance test
  of the RISING-set forward mean against the unconditional mean.
- **Randomization / placebo null.** Because regime months are autocorrelated and the
  effective sample is small, we resample random same-size month sets and ask how often chance
  is as bearish as the RISING set (Fisher's randomization logic; Efron & Tibshirani, *An
  Introduction to the Bootstrap*, 1993).
- **One coincident shock dominates.** The COVID-2020 claims spike (4-wk MA ~4,170k) is one
  enormous coincident event; we report results with and without 2020–2021 so the verdict
  doesn't ride on a single observation.

## Method lineage (this study's engine)

- **Signal + inference.** [`strategy.claims_momentum`](../jobless_claims_momentum/strategy.py),
  [`strategy.summarize`](../jobless_claims_momentum/strategy.py) (Welch *t* + placebo *p*),
  [`strategy.lead_lag`](../jobless_claims_momentum/strategy.py) (the early-warning test),
  [`strategy.timing_overlay`](../jobless_claims_momentum/strategy.py) (cash-on-rising-claims,
  one-month lag, one-way costs).
- **Deterministic synthetic control.**
  [`data.synthetic_claims`](../jobless_claims_momentum/data.py) plants a known claims→returns
  link; `edge = 0` must not manufacture significance, a large `edge` must light up the test.

## Data sources used here

- **FRED `IC4WSA`** (hardcoded monthly snapshot, thousands, SA) + **yfinance SPY** daily
  adjusted close, 1993-01 → 2026-06, cached under `_cache/spy_prices.csv`. All headline
  numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies

- **[Study 268 — Sahm-Rule](../268-sahm-rule/)**: the labour-market recession trigger done
  the same way (hardcoded FRED snapshot + SPY drawdowns); a sibling test of whether a famous
  labour signal is an early-warning you can trade.
- **[Study 266 — Misery-Index](../266-misery-index/)** and
  **[Study 267 — M2-Growth](../267-m2-growth/)**: companion macro-nowcasting teardowns asking
  whether a celebrated macro gauge actually times equities.
