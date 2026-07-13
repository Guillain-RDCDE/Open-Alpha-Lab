# References & literature map — Study 759 (Redbook-Retail)

## The claim under test

- **The Redbook Index as a consumer nowcast.** The **Johnson Redbook Index** (Redbook Research
  Inc.) is a weekly measure of **same-store (comparable) sales growth** at a large sample of
  U.S. general-merchandise retailers, reported as a **year-over-year percentage** and released
  each **Tuesday**. It is marketed as the most *timely* read on U.S. retail demand — weekly,
  same-store, minimal lag — and is quoted across Reuters, Investing.com and financial-TV
  "state of the consumer" segments. Redbook Research, *Redbook Retail Sales Index —
  Methodology* (subscription series).
- **The market-timing folklore.** From "Redbook is the real-time consumer" follows the trading
  lore: when same-store growth **accelerates**, the shopper is strengthening, so tilt into the
  **retail sector** — the SPDR S&P Retail ETF (**XRT**) — ahead of the crowd. We test the
  strongest form: *does a Redbook-momentum (acceleration) uptick lead retail stocks cleanly
  enough to trade?*
- **The target.** **XRT** — SPDR S&P Retail ETF (State Street), an equal-weighted basket of
  U.S. retail names; listed **2006-06**, which bounds the sample. SPY (SPDR S&P 500) is the
  broad-market benchmark for the retail-vs-market relative test.

## Why the Redbook series isn't fetched live here — and what we do

- **Proprietary, off FRED.** Unlike jobless claims or the unemployment rate, the weekly Redbook
  Index is a **paid, proprietary feed**; its long history is paywalled and it is **not on
  FRED/ALFRED**. Following the desk convention for small, clearly-cited alt-data series —
  **Study 358 (Watch-Index)** and **Study 708 (Eurovision-Effect)** hardcode a small approximate
  series — we hardcode a **monthly, approximate reconstruction** of the headline YoY same-store
  number (as-of 2026-07-12). It is a **LABELLED PROXY**, faithful in *shape* to the public
  record (the 2008–09 slide into negative growth, the soft 2015–16 patch, the strong 2018–19
  run, the COVID-2020 collapse, the 2021–22 *nominal* reopening/inflation surge into double
  digits, the 2023 deceleration) but **approximate in monthly level**. It is never presented
  under a real-tape banner; the proxy caveat is named on the Signal axis.
- **Equities.** XRT and SPY daily adjusted close via **yfinance** (no key), month-end sampled,
  total-return adjusted — labelled as such.

## Why "leading" is the crux — coincident vs lagging, and nominal contamination

- **The stock market as its own leading indicator.** Stock prices are a Conference Board LEI
  component and famously lead the real economy (Samuelson's quip that the market "predicted
  nine of the last five recessions"). A sector ETF like XRT reprices the consumer in **real
  time**, so a *monthly* same-store gauge that co-moves with it need not **lead** it — it may
  merely echo a turn the market already made. This is the confound the study isolates with an
  explicit **lead/lag cross-correlation**.
- **Reference-cycle classification.** Burns & Mitchell (1946), *Measuring Business Cycles*
  (NBER) — the original leading/coincident/lagging taxonomy at cycle turns; a series can
  co-move with the cycle yet lag the equity market.
- **Nominal vs real sales.** Redbook YoY is a **nominal** growth rate: the 2021–22 prints hit
  double digits largely on **inflation**, precisely when XRT *fell*. A gauge that cannot
  distinguish "more units sold" from "same units, higher prices" is treacherous to trade
  equities on — see the level-regime result, where *strong nominal* same-store months precede
  *weaker* forward returns.
- **Predictive regressions and small-sample caution.** Welch & Goyal (2008), *A Comprehensive
  Look at the Empirical Performance of Equity Premium Prediction* (Review of Financial Studies)
  — most macro predictors that look significant in-sample fail out-of-sample; the bar for a
  tradable single-series nowcast is high.

## Why the inference is small-sample / placebo-based

- **Welch two-sample t.** Welch (1947), *The generalization of "Student's" problem when several
  different population variances are involved* (Biometrika) — unequal-variance test of the
  ACCELERATING-set forward mean against the unconditional mean.
- **Randomization / placebo null.** Because regime months are autocorrelated and the effective
  sample is modest (241 months), we resample random same-size month sets and ask how often
  chance is as bullish as the ACCEL set (Fisher's randomization logic; Efron & Tibshirani,
  *An Introduction to the Bootstrap*, 1993).
- **One coincident/inflation shock dominates.** The 2021–22 double-digit surge is one enormous
  nominal event; we report robustness with the COVID/inflation window dropped so the verdict
  doesn't ride on it.

## Method lineage (this study's engine)

- **Signal + inference.** [`strategy.redbook_momentum`](../redbook_retail/strategy.py),
  [`strategy.summarize`](../redbook_retail/strategy.py) (Welch *t* + placebo *p*, absolute &
  relative), [`strategy.lead_lag`](../redbook_retail/strategy.py) (the nowcast/identification
  test), [`strategy.regime_summary`](../redbook_retail/strategy.py) (level-regime split),
  [`strategy.timing_overlay`](../redbook_retail/strategy.py) (own-when-accelerating, one-month
  lag, one-way costs).
- **Deterministic synthetic control.**
  [`data.synthetic_redbook`](../redbook_retail/data.py) plants a known Redbook→returns link;
  `edge = 0` must not manufacture significance, a large `edge` must light up the test.

## Data sources used here

- **Redbook same-store YoY** (hardcoded monthly LABELLED PROXY, %) + **yfinance XRT & SPY**
  daily adjusted close, 2006-06 → 2026-06, cached under `_cache/xrt_spy_prices.csv`. All
  headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies

- **[Study 385 — Jobless-Claims-Momentum](../385-jobless-claims-momentum/)**: the weekly claims
  number done the same way (hardcoded snapshot + ETF, momentum + lead/lag + timing overlay) — a
  sibling test of whether a famous "leading" macro series leads the tape or echoes it.
- **[Study 384 — ISM-PMI-Regime](../384-ism-pmi-regime/)**: the manufacturing cycle as an
  own-when-above-threshold regime signal — companion macro-nowcasting teardown.
