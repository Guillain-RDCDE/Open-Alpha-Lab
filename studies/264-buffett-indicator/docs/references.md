# References & literature map — Study 264 (Buffett-Indicator)

## The claim under test

**Buffett, W. & Loomis, C. (2001).** "Warren Buffett on the Stock Market."
*Fortune*, Dec. 10, 2001. The origin of the indicator. Buffett: "the ratio of
total market value of all publicly traded securities as a percentage of the
country's GNP… is probably the best single measure of where valuations stand at
any given moment." He sketched the rule of thumb: near 70–80% of GDP is a good
time to buy; approaching 200% is "playing with fire." This study tests whether
the *level* of market-cap-to-GDP forecasts the next year's equity return.

## Valuation predicts long-horizon returns — the supporting literature

- **Campbell, J. Y. & Shiller, R. J. (1988, 1998).** "Stock Prices, Earnings,
  and Expected Dividends" (*Journal of Finance*) and "Valuation Ratios and the
  Long-Run Stock Market Outlook" (*Journal of Portfolio Management*). The
  foundational case that price/dividend and price/earnings ratios mean-revert and
  predict multi-year returns. CAPE (PE10) is the close cousin of the Buffett
  Indicator — see [Study 120 — Excess-CAPE-Yield](../../120-excess-cape-yield/).

- **Cochrane, J. H. (2008, 2011).** "The Dog That Did Not Bark" (*RFS*) and
  "Discount Rates" (AFA Presidential Address, *Journal of Finance*). Establishes
  that almost all variation in valuation ratios reflects time-varying *expected
  returns*, not expected cash-flow growth — the theoretical engine behind any
  valuation-timing claim, and the reason the predictability is strongest at long
  horizons, weak at one year.

- **Goyal, A. & Welch, I. (2008).** "A Comprehensive Look at the Empirical
  Performance of Equity Premium Prediction." *Review of Financial Studies*,
  21(4), 1455–1508. The cold shower: a battery of valuation predictors
  (dividend yield, earnings yield, book/market…) that look significant in-sample
  fail to beat the prevailing-mean forecast **out-of-sample**. The single most
  important caution for this study — and exactly what the real-time
  expanding-median timing test demonstrates here.

- **Campbell, J. Y. & Thompson, S. B. (2008).** "Predicting Excess Stock Returns
  Out of Sample: Can Anything Beat the Historical Average?" *RFS*, 21(4). The
  partial rebuttal: with sensible economic restrictions, valuation predictors
  recover a *little* out-of-sample power — but the gains are small and fragile.

## Why the Buffett Indicator specifically is fragile

- **Numerator/denominator mismatch.** Market cap is a forward-looking,
  globally-earning quantity (US-listed firms earn abroad; foreign and private
  firms are excluded from GDP's "domestic" denominator); GDP is a backward,
  domestic flow. The ratio has **drifted structurally upward** for decades
  (rising profit share, globalisation, intangibles, lower rates) — so a fixed
  "200% is expensive" threshold is a moving target. This is why our honest test
  uses an **expanding** median rather than a hindsight level.

- **Non-stationarity / look-ahead.** Any threshold ("buy below 80%, sell above
  150%") is calibrated on the realized history; applied in real time it kept
  investors out of the market for most of 1996–2025. The in-sample tercile sort
  flatters the indicator; the expanding-window timing rule is the honest version
  (Goyal–Welch in miniature).

- **One-year horizon.** Valuation predictability is a *long-horizon* phenomenon
  (5–10 years). At the one-year horizon tested here, the signal-to-noise ratio is
  tiny — ~17% annual equity vol swamps a ~3 pp/100pp slope.

## Method lineage

- **Predictive regression with HAC/Newey-West standard errors.**
  Newey, W. K. & West, K. D. (1987), *Econometrica* 55(3). Overlapping/serially
  correlated forecast errors require HAC inference; we use the automatic-lag
  Bartlett kernel. The |t| ≥ 2 bar (and the higher ~3.0 hurdle Harvey-Liu-Zhu
  (2016) propose for valuation factors) is the inference standard.
- **Out-of-sample / walk-forward evaluation.** The expanding-median timing rule
  is the point-in-time analogue of Goyal–Welch's out-of-sample R²: it asks what a
  real investor, knowing only the past, would have done.

## Data sources

- **Buffett Indicator (market cap / GDP).** Year-end levels hardcoded in
  `data.py`. Numerator: Wilshire 5000 total US market value (FRED `WILL5000PR` /
  the corporate-equity market-value lineage). Denominator: nominal GDP (FRED
  `GDP`). Cross-checked against GuruFocus and CurrentMarketValuation.com
  Buffett-Indicator histories.
- **S&P 500 price returns.** `^GSPC` daily closes (split-only, **price index, no
  dividends**) from the repo-level `_cache/^GSPC_split_only.parquet`,
  resampled to year-end Dec/Dec calendar-year returns.

## Related desk studies

- **[Study 120 — Excess-CAPE-Yield](../../120-excess-cape-yield/)**: Shiller's
  CAPE-based expected-return model — the closest valuation cousin, and the one
  desk study graded `Real` on the Signal axis.
- **[Study 158 — Super-Bowl](../../158-super-bowl/)**: the small-n folklore
  teardown methodology (correct-baseline, positive-control) reused here.
