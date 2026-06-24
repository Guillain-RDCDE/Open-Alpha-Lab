# References & literature map — Study 441 (Camarilla Pivots)

## The claim under test

- **The folklore.** "Compute eight lines from yesterday's high, low and close. **L3** and **H3**
  are reversal levels the market respects — fade L3 long and H3 short back toward the central
  pivot; when price blows through **L4/H4** it breaks out and runs." The Camarilla equation is
  attributed to **Nick Stott / Nick Scott** (bond trader, late 1980s) and was popularised
  through day-trading education sites (e.g. the "Camarilla pivot" pages on Investopedia,
  TradingView's built-in *Pivot Points Standard / Camarilla* indicator, and countless retail
  forums). The fixed multipliers — `1.1/12, 1.1/6, 1.1/4, 1.1/2` — are lore, not derived.
- **The exact recipe we test.** With prior-session range `R = high − low` and close `C`:
  `H3 = C + R·1.1/4`, `L3 = C − R·1.1/4`, `H4/L4 = C ± R·1.1/2`, and the classic central pivot
  `P = (high + low + close)/3`. See [`data.camarilla_levels`](../camarilla_pivots/data.py).

## Why the placebo is the whole test

- **Support/resistance under scrutiny.** Osler (2000, *Support for Resistance: Technical
  Analysis and Intraday Exchange Rates*, FRBNY Economic Policy Review) and Osler (2003,
  *Currency Orders and Exchange Rate Dynamics*, Journal of Finance) found that *published*
  support/resistance levels (round numbers, prior extremes) do show mild predictive clustering
  in FX — but the effect is order-flow driven and concentrated at psychologically salient
  prices, not at arbitrary arithmetic levels. The key methodological point: a level only counts
  as "real" if it beats a **randomly placed** level. We make that the headline comparison.
- **Pivot points more broadly.** Person (*A Complete Guide to Technical Trading Tactics*, 2004)
  and the broad pivot-point literature popularised floor-trader pivots; controlled tests
  (e.g. academic and practitioner backtests of pivot bounces) generally find no robust edge once
  the line is compared to a control and costs are charged — consistent with our result.
- **Geometry banks free mean-reversion.** A volatile price that wanders into any horizontal line
  will, on average, drift back toward the range center afterward — a property of *crossing a
  line at all*, not of a particular line. This is why the one-sample "does it revert?" test is
  contaminated and the **real − random-control difference** is decisive.

## Method lineage (the desk's shared engine)

- **First-touch event study + one-sample t.** [`strategy.collect_events`](../camarilla_pivots/strategy.py)
  and [`strategy.ttest_vs_zero`](../camarilla_pivots/strategy.py) — first intraday touch of
  L3/H3, signed forward reversion toward the pivot, one-bar execution lag.
- **HAC (Newey-West) t.** [`strategy.hac_t`](../camarilla_pivots/strategy.py) on the daily-mean
  series — robustness to the within-day / cross-name clustering of events (Newey & West, 1987,
  *A Simple, Positive Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance
  Matrix*, Econometrica).
- **Random-control placebo + bootstrap on the difference.**
  [`strategy.collect_random_controls`](../camarilla_pivots/strategy.py) and
  [`strategy.bootstrap_diff_p`](../camarilla_pivots/strategy.py) — the honest "does the line beat
  a random line?" test (Efron & Tibshirani, *An Introduction to the Bootstrap*, 1993).
- **Deterministic synthetic control.** [`data.synthetic_panel`](../camarilla_pivots/data.py)
  plants a genuine respect-the-level pull (knob `respect`); with `respect=0` the test must NOT
  manufacture significance and must lose to its control — the offline core runs with no network.

## Data sources used here

- **yfinance** 5-minute intraday bars (price-only, `auto_adjust=False`) for SPY, QQQ, AAPL, MSFT,
  NVDA, 2026-03-30 → 2026-06-23, cached under `_cache/bars_*_5m.parquet`. Yahoo caps 5m history
  at ~60 calendar days and 1m at ~7 days — the loud short-span caveat. All headline numbers are
  pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies

- [`../104-bollinger-reversion`](../104-bollinger-reversion) — the same mean-reversion-at-a-line
  question with Bollinger bands; same "does the band beat a fair coin?" discipline.
- [`../178-cci`](../178-cci) and the wider technical-indicator zoo — most drawn-line / oscillator
  systems land NONE × MIRAGE under the same protocol.
- The **research-method demos** (data-mining-roulette, multiple-testing) frame why a touch that
  *sometimes* bounces is not a signal — the random-control placebo is the cleanest cure.
