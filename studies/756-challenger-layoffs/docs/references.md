# References & literature map — Study 756 (Challenger-Layoffs)

## The claim under test

- **The Challenger Job Cuts Report as a leading labour signal.** Challenger, Gray &
  Christmas, Inc. publishes a monthly **Job Cuts Report** tallying the number of layoffs
  **announced** by U.S. employers, released on/around the first Thursday of the following
  month — ahead of the BLS employment situation. Because it captures *announcements*
  (forward-looking intentions) rather than realized separations, it is widely framed in
  the financial press as an **early read on labour-market and macro weakness**. Challenger,
  Gray & Christmas, *Challenger Report* (monthly press releases), https://www.challengergray.com/.
- **The market-timing folklore.** From this the trading lore follows: a **spike** in
  announced job cuts is read as an early-warning for equities — get defensive when the
  Challenger number jumps. The idea recurs in recession-watch commentary (CNBC, Bloomberg,
  Reuters coverage of each monthly print) and macro-tactical newsletters. We test the
  strongest form: *does a Challenger cut-spike lead equity downturns cleanly enough to trade?*
- **What the data actually is.** Announced job cuts are *intentions*, not the household- or
  establishment-survey counts; they lead realized layoffs but are noisy, lumpy (a single
  large employer swings a month), and concentrated in downturns. Challenger's series is
  **proprietary** — there is no free FRED or yfinance feed — so we hardcode a **labelled,
  approximate** monthly reconstruction from the published headline totals (the desk's proxy
  convention, below).

## Why the job-cut series is a labelled proxy here — and what we do

- **Proprietary series, no free feed.** Unlike FRED macro series, the Challenger totals are
  not redistributable via an open API. Following the desk convention for small, public,
  clearly-cited series with no machine feed — **Study 358 (Watch-Index)** hardcodes a
  labelled auction-price proxy, **Study 708 (Eurovision)** hardcodes a labelled points
  table — we **hardcode an approximate monthly snapshot** of announced job cuts (thousands),
  built from Challenger's published press-release headlines and the well-documented monthly
  spikes (post-9/11 2001, the 2008–09 crisis, the 2015 energy bust, the record COVID-2020
  surge, the 2023 tech wave, the 2025 federal cuts). It is an **approximation, not the
  revised vintage**, and that limitation is named on the Signal axis.
- **Equities.** SPY daily adjusted close via **yfinance** (no key), month-end sampled,
  total-return adjusted — labelled as such.

## Why "leading" is the crux — coincident vs lagging confusion

- **Reference-cycle dating and lead/lag.** Burns & Mitchell (1946), *Measuring Business
  Cycles* (NBER) — the original classification of series into leading, coincident and
  lagging at business-cycle turns. A series can co-move strongly with the cycle yet **lag**
  the equity market, which itself leads the real economy. We run an explicit **lead/lag
  cross-correlation** to locate where the cut spike actually sits relative to SPY returns.
- **The stock market as its own leading indicator.** Stock prices are a Conference Board LEI
  component and famously lead the real economy (Samuelson's quip that the market "predicted
  nine of the last five recessions"). So a labour series that lines up with *contemporaneous*
  equity weakness need not **lead** the market — it may merely echo a turn the market already
  made. This is the confound the study isolates.
- **Announced vs realized layoffs.** Davis, Faberman & Haltiwanger (2006), *The Flow
  Approach to Labor Markets* (Journal of Economic Perspectives) — layoff/separation flows are
  highly cyclical and cluster at downturns; announced cuts lead *realized* separations, but
  that is a different question from leading *asset prices*, which discount ahead of both.
- **Predictive regressions and small-sample caution.** Welch & Goyal (2008), *A Comprehensive
  Look at the Empirical Performance of Equity Premium Prediction* (Review of Financial
  Studies) — most macro predictors that look significant in-sample fail out-of-sample; the
  bar for a tradable macro signal is high, and a single famous "leading" labour series is no
  exception.

## Why the inference is HAC + placebo-based

- **Welch two-sample t.** Welch (1947), *The generalization of "Student's" problem when
  several different population variances are involved* (Biometrika) — unequal-variance test
  of the SPIKE-set forward mean against the unconditional mean.
- **Newey-West HAC standard errors.** Newey & West (1987), *A Simple, Positive
  Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix*
  (Econometrica) — overlapping multi-month forward returns induce MA autocorrelation, so the
  spike-dummy coefficient is judged with a Bartlett-kernel HAC *t* (lag truncation = horizon).
- **Randomization / placebo null.** Because regime months are autocorrelated and the effective
  sample is small, we resample random same-size month sets and ask how often chance is as
  bearish as the SPIKE set (Fisher's randomization logic; Efron & Tibshirani, *An Introduction
  to the Bootstrap*, 1993).
- **One coincident shock dominates.** The COVID-2020 cut spike (~671k announced in April 2020)
  is one enormous coincident event; we report results with and without 2020–2021 so the
  verdict doesn't ride on a single observation.

## Method lineage (this study's engine)

- **Signal + inference.** [`strategy.cut_spike`](../challenger_layoffs/strategy.py),
  [`strategy.summarize`](../challenger_layoffs/strategy.py) (Welch *t* + HAC *t* + placebo *p*),
  [`strategy.hac_spike_t`](../challenger_layoffs/strategy.py) (Newey-West),
  [`strategy.lead_lag`](../challenger_layoffs/strategy.py) (the early-warning test),
  [`strategy.timing_overlay`](../challenger_layoffs/strategy.py) (cash-on-spike, one-month
  lag, one-way costs).
- **Deterministic synthetic control.**
  [`data.synthetic_cuts`](../challenger_layoffs/data.py) plants a known cuts→returns link;
  `edge = 0` must not manufacture significance, a large `edge` must light up the test.

## Data sources used here

- **Challenger, Gray & Christmas monthly job cuts** (hardcoded, labelled, approximate
  snapshot, thousands announced) + **yfinance SPY** daily adjusted close, 2000-01 → 2026-06,
  cached under `_cache/spy_prices.csv`. All headline numbers are pinned in
  [`docs/results.md`](results.md) and reproduced by [`examples/verify.py`](../examples/verify.py).

## Related desk studies

- **[Study 385 — Jobless-Claims-Momentum](../385-jobless-claims-momentum/)**: the sibling
  macro-nowcasting teardown (rising initial claims as an early-warning) done the same way —
  hardcoded labour snapshot + SPY, Welch *t*, placebo null, lead/lag scan, timing overlay.
- **[Study 749 — Layoff-Drift](../749-layoff-drift/)**: the *cross-sectional* companion —
  single-firm mass-layoff **announcement drift** (market-model CARs on the announcers),
  where this study asks the *aggregate/macro* question instead.
- **[Study 268 — Sahm-Rule](../268-sahm-rule/)** and **[Study 384 — ISM-PMI-Regime](../384-ism-pmi-regime/)**:
  companion labour/activity gauges asking whether a celebrated macro series actually times equities.
