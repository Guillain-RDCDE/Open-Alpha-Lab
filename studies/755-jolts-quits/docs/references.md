# References & literature map — Study 755 (JOLTS-Quits)

## The claim under test

- **Quits as worker confidence.** The JOLTS **quits rate** measures voluntary separations
  as a share of employment; the standard interpretation is that workers quit only when
  confident of a better outside option, so the quits rate is a real-time gauge of
  labour-market heat and worker sentiment. U.S. Bureau of Labor Statistics, *Job Openings
  and Labor Turnover Survey (JOLTS) — Concepts and Methodology* and the monthly *JOLTS News
  Release*; series **`JTSQUR`** (Quits: Total Nonfarm, Rate, Seasonally Adjusted).
- **The Fed watches quits.** Policymakers treat quits as a leading-ish read on wage
  pressure and labour tightness. See e.g. the frequent citation of the quits rate in FOMC
  communications and the "prime-age quits" discussion in Fed research; the *Beveridge
  curve* / quits framing in Barnichon & Figura and related labour-flow literature.
- **The market-timing folklore.** From "quits = confidence" the trading lore follows: a
  **falling** quits rate signals fading confidence and a softening cycle, so lighten equity
  risk — especially cyclicals — when quits roll over. The idea recurs in macro-tactical
  commentary and "quits rate is the number to watch" recession-watch segments. We test the
  strongest form: *does a quits-rate downturn lead equity/cyclical weakness cleanly enough,
  and early enough, to trade — given JOLTS's own publication delay?*

## Why the JOLTS release lag is central here

- **The ~6-week publication delay.** JOLTS is released roughly the first week of the month
  for the reference month **two months prior** (e.g. the March reading is published in early
  May). BLS, *JOLTS release schedule*. So the earliest an honest trader can act on a
  reference-month-t print is the close of month **t+2** — a **2-month execution lag**,
  applied once, with no look-ahead. This is *more* conservative than the 1-month lag used
  for the weekly-claims study (385) and it is the distinguishing operational constraint of
  this series: any residual lead is already stale before it can be traded.
- **Data provenance.** The quits tape is a hardcoded monthly snapshot of FRED `JTSQUR`
  (as-of 2026-07-13) — the settled print, **not** the real-time vintage. JOLTS is revised,
  and that survivorship-of-revision caveat is named on the Signal axis. Following the desk
  convention for small, public macro series (Study 385 hardcodes `IC4WSA`, Study 268
  hardcodes `UNRATE`), we freeze the snapshot so the study is reproducible in place.
- **Equities.** SPY daily adjusted close via **yfinance** (no key), month-end sampled,
  total-return adjusted; and a **cyclical-minus-defensive** long-short from **XLY**
  (consumer discretionary) minus **XLP** (consumer staples), monthly-rebalanced,
  total-return legs — the "risk appetite" tape the claim names. All labelled as such.

## Why "leading" is the crux — coincident vs lagging confusion

- **Reference-cycle dating and lead/lag.** Burns & Mitchell (1946), *Measuring Business
  Cycles* (NBER) — the original classification of series into leading, coincident and
  lagging at business-cycle turns. A series can co-move strongly with the cycle yet **lag**
  the equity market, which itself leads the real economy. We run an explicit **lead/lag
  cross-correlation** to locate where quits momentum actually sits relative to SPY returns.
- **The stock market as its own leading indicator.** Stock prices are a Conference Board LEI
  component and famously lead the real economy (Samuelson's quip that the market "predicted
  nine of the last five recessions"). So a labour series that lines up with *contemporaneous*
  equity weakness need not **lead** the market — it may merely echo a turn the market already
  made. This is the confound the study isolates.
- **Labour-market flows lag output.** Quits and other labour flows are classically
  *lagging-to-coincident* relative to the cycle peak (hiring/quitting decisions respond to
  realised demand). Blanchard & Diamond (1990), *The Cyclical Behavior of the Gross Flows of
  U.S. Workers* (Brookings) documents the cyclicality and timing of worker flows, including
  quits, relative to the cycle.
- **Predictive regressions and small-sample caution.** Welch & Goyal (2008), *A
  Comprehensive Look at the Empirical Performance of Equity Premium Prediction* (Review of
  Financial Studies) — most macro predictors that look significant in-sample fail
  out-of-sample; the bar for a tradable macro signal is high, and applies directly to a
  single famous "leading" labour series over a 25-year JOLTS sample.

## Why the inference is small-sample / placebo-based

- **Welch two-sample t.** Welch (1947), *The generalization of "Student's" problem when
  several different population variances are involved* (Biometrika) — unequal-variance test
  of the FALLING-set forward mean against the unconditional mean.
- **Randomization / placebo null.** Because regime months are autocorrelated and the
  effective sample is small (306 months, JOLTS begins Dec 2000), we resample random
  same-size month sets and ask how often chance is as bearish as the FALLING set (Fisher's
  randomization logic; Efron & Tibshirani, *An Introduction to the Bootstrap*, 1993).
- **One coincident shock dominates.** The 2020–2021 COVID collapse-and-rebound in quits is
  one enormous coincident event; we report results with and without 2020–2021 so the verdict
  doesn't ride on a single episode.

## Method lineage (this study's engine)

- **Signal + inference.** [`strategy.quits_momentum`](../jolts_quits/strategy.py),
  [`strategy.summarize`](../jolts_quits/strategy.py) (Welch *t* + placebo *p*),
  [`strategy.lead_lag`](../jolts_quits/strategy.py) (the leading-gauge test),
  [`strategy.timing_overlay`](../jolts_quits/strategy.py) (cash-on-falling-quits, 2-month
  release lag, one-way costs).
- **Deterministic synthetic control.**
  [`data.synthetic_quits`](../jolts_quits/data.py) plants a known quits→returns link;
  `edge = 0` must not manufacture bearish significance, a large `edge` must light up the test.

## Data sources used here

- **FRED `JTSQUR`** (hardcoded monthly snapshot, percent, SA) + **yfinance SPY / XLY / XLP**
  daily adjusted close, 2000-12 → 2026-05, cached under `_cache/*.csv`. All headline numbers
  are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies

- **[Study 385 — Jobless-Claims-Momentum](../385-jobless-claims-momentum/)**: *rising*
  jobless claims as an early-warning, same hardcoded-snapshot + SPY method — a sibling test
  of whether a famous labour signal is a tradable lead (it isn't).
- **[Study 384 — ISM-PMI-Regime](../384-ism-pmi-regime/)** and
  **[Study 387 — Economic-Surprise-Index](../387-economic-surprise-index/)**: companion
  macro-nowcasting teardowns asking whether a celebrated macro gauge actually times equities.
