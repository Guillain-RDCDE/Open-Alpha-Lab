# References & literature map — Study 427 (Rate of Change)

## The claim under test

- **The folklore.** "Rate of Change is the oldest and purest momentum indicator. Go long when
  ROC crosses above zero (price higher than *N* days ago = uptrend), step aside to cash when it
  crosses below — you ride the trend and dodge the crashes." ROC predates RSI and MACD and ships
  on every charting platform.
- **The canonical exposition.** ROC and its cousins are catalogued in the technician's bibles:
  Robert D. Edwards & John Magee, *Technical Analysis of Stock Trends* (1948, and later editions);
  Martin J. Pring, *Technical Analysis Explained* (rate-of-change / momentum chapter); and John J.
  Murphy, *Technical Analysis of the Financial Markets* (1999), which defines
  `ROC = 100 × (P_t / P_{t−N})` and the zero-line crossover rule we test.

## What's real underneath — the momentum literature

- **Cross-sectional momentum is real.** Narasimhan Jegadeesh & Sheridan Titman,
  *Returns to Buying Winners and Selling Losers* (1993, Journal of Finance) — the foundational
  evidence that *relative* past returns predict future returns. This is the legitimate effect the
  ROC folklore borrows its credibility from.
- **Time-series momentum is also real — across many markets.** Tobias Moskowitz, Yao Hua Ooi &
  Lasse Heje Pedersen, *Time Series Momentum* (2012, Journal of Financial Economics) show a
  past-12-month sign rule pays *diversified across dozens of futures* — not as a single-asset
  long/flat overlay on one equity index with a secular drift, which is the form ROC-on-SPY takes.
- **Trend-following = a moving-average family.** Valeriy Zakamulin, *Market Timing with Moving
  Averages* (2017) proves that ROC, SMA crosses, EMA crosses and MACD are **algebraically close
  cousins** — weighted averages of past returns — which is exactly why our rivals race ends in a
  dead heat. The choice of indicator is largely cosmetic.

## Why the *difference* t is the only honest Signal test

- **Beta is not alpha.** A long/flat rule that is invested most of the time inherits the equity
  risk premium; its standalone Sharpe/*t* mostly measures that premium. The desk's inference bar
  (METHODOLOGY → *The inference bar*) requires the **incremental** statistic — here a Newey-West
  HAC *t* on the daily difference ROC − buy-and-hold. Whitney K. Newey & Kenneth D. West,
  *A Simple, Positive Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance
  Matrix* (1987, Econometrica) is the estimator.
- **Data-snooping over indicators.** Ryan Sullivan, Allan Timmermann & Halbert White,
  *Data-Snooping, Technical Trading Rule Performance, and the Bootstrap* (1999, Journal of
  Finance) and Halbert White, *A Reality Check for Data Snooping* (2000, Econometrica) show that
  picking the best of many technical rules manufactures significance — motivating our
  permutation placebo and the rivals race rather than cherry-picking a window.

## Timing, costs, and the buy-and-hold benchmark

- **Total return, not price-only.** SPY closes are `auto_adjust=True` (dividends + splits folded
  in), so the buy-and-hold benchmark is the honest one — a price-only index would understate the
  thing ROC has to beat. (House rule: *price-only vs total-return labeled everywhere*.)
- **One execution lag, costs one-way × NAV.** Signal at close *t*, return of *t+1* (one `shift`);
  cost = one-way bps × |Δposition| × NAV. The Sharpe race is **excess-vs-excess** because the rule
  sits in cash part of the time (METHODOLOGY → *House rules*).

## Method lineage (the study's engine)

- **The indicator + timing rules.** [`strategy.roc`](../rate_of_change/strategy.py),
  [`strategy.roc_signal`](../rate_of_change/strategy.py), and the rivals
  [`sma_cross_signal` / `macd_signal` / `rsi_signal`](../rate_of_change/strategy.py).
- **HAC inference + the decisive difference.** [`strategy.hac_tstat`](../rate_of_change/strategy.py)
  and [`strategy.diff_vs_hold`](../rate_of_change/strategy.py) — the Newey-West *t* on ROC minus
  buy-and-hold.
- **Permutation placebo.** [`strategy.permutation_pvalue`](../rate_of_change/strategy.py) — 5,000
  circular shifts of the position vector, the timing-information null.
- **Deterministic synthetic control.** [`data.synthetic_panel`](../rate_of_change/data.py) plants
  a regime-switching trend with a tunable `edge`; with the edge off the difference-*t* must NOT
  manufacture significance, with it on ROC must beat hold — the offline core runs with no network.

## Data sources used here

- **yfinance** daily total-return closes for **SPY**, 1993-01-29 → 2026-05-29 (pinned to the last
  complete month), cached under `_cache/bars_SPY_1d.parquet`, fingerprint `f3fa058adfc8`. All
  headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies

- [`../104-bollinger-reversion`](../../104-bollinger-reversion) and [`../178-cci`](../../178-cci) —
  sibling single-indicator teardowns that land the same place: a gross edge that is mostly
  market drift, not indicator skill.
- [`../363-pead-drift`](../../363-pead-drift) — the *counter*-example: a folk effect that **does**
  clear the t≥2 bar on the decisive test, for contrast with ROC's null.
- The **research-method demos** (data-mining-roulette, multiple-testing) frame why a high
  standalone *t* on a mostly-long rule is not evidence — ROC is the clean cautionary case.
