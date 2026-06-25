# References & literature map — Study 469 (Relative Vigor Index)

## The claim under test

- **The folklore.** The Relative Vigor Index reads the bar **body** (close − open) relative to its
  **range** (high − low): in an up-trend a market closes above its open, in a down-trend below. A
  4-bar symmetric-weighted smoother is applied to numerator and denominator; the *N*-bar sums give
  the **RVI**, and the same smoother applied to the RVI gives a **signal line**. The trading lore
  is the classic oscillator-crossover: **RVI crossing above its signal line is a buy** (vigor
  turning up), the cross below a sell. This is the retail/technician staple built into TradingView,
  MetaTrader, Thinkorswim and every charting suite.
- **The source.** **John F. Ehlers** introduced the Relative Vigor Index in *Cybernetic Analysis
  for Stocks and Futures* (Wiley, 2004), and in his *Stocks & Commodities* articles ("Relative
  Vigor Index", c. 2002). Ehlers framed it explicitly as a refined, noise-damped momentum
  oscillator — the symmetric (1,2,2,1)/6 weighting is his "value" smoother designed to reduce lag
  while suppressing per-bar noise. The modern popular write-ups (Investopedia, StockCharts'
  ChartSchool, the TradingView docs) restate the crossover rule.
- **Lineage.** The RVI sits in the family of price-relative-to-bar oscillators alongside the
  Stochastic (%K vs %D crossover, Lane), Chande's momentum oscillators, and the host of
  signal-line-crossover triggers (MACD, etc.) — all of which the desk has tested against the
  random-entry baseline and which land in the same place.

## Why the high one-sample t is not evidence

- **Drift / beta.** US equity indices have a positive unconditional daily mean; a one-sample *t*
  of a long-only entry rule against **zero** measures that drift, not the rule. See Fama & French
  on the equity premium; the desk's standing rule is *excess-vs-excess* and *signal-vs-baseline*,
  never *signal-vs-zero*. Here the one-sample *t* reaches +7.39 at 20 days while the cross-vs-random
  Welch *t* is −0.37 — the entire apparent significance is the tide.
- **Data snooping on chart tools.** Lo, Mamaysky & Wang (2000, *Foundations of Technical Analysis*,
  Journal of Finance) formalize testing chart patterns against a properly matched null; Sullivan,
  Timmermann & White (1999, *Data-Snooping, Technical Trading Rule Performance, and the Bootstrap*,
  Journal of Finance) and White (2000, *A Reality Check for Data Snooping*, Econometrica) show how
  indicator rules manufacture significance unless raced against a fair benchmark.
- **HAC inference.** Newey & West (1987) standard errors for the one-sample mean; Welch (1947)
  two-sample *t* for the cross-vs-random difference.

## Method lineage (the desk's shared engine)

- **Causal RVI + signal line.** [`strategy.rvi`](../relative_vigor_index/strategy.py),
  [`strategy._swma4`](../relative_vigor_index/strategy.py) — Ehlers' 4-bar symmetric smoother and
  the *N*-bar sum, all strictly causal (no future bars).
- **Cross-up entries + forward-return + HAC t + random baseline.**
  [`strategy.cross_up_entries`](../relative_vigor_index/strategy.py),
  [`strategy.forward_returns`](../relative_vigor_index/strategy.py),
  [`strategy.hac_t`](../relative_vigor_index/strategy.py),
  [`strategy.run_experiment`](../relative_vigor_index/strategy.py).
- **Timing placebo.** [`strategy.phase_scramble_placebo`](../relative_vigor_index/strategy.py) —
  circularly roll the RVI/signal series vs price, destroying the cross's timing while keeping its
  marginal — the direct test of "does the cross's timing matter?".
- **Deterministic synthetic control.** [`data.synthetic_panel`](../relative_vigor_index/data.py)
  plants a real persistent-regime momentum (knob `edge`); with `edge = 0` the detector must NOT
  manufacture significance — the offline core runs with no network.

## Data sources used here

- **yfinance** daily adjusted (total-return) closes for SPY, QQQ, IWM, DIA, GLD,
  2005-01-03 → 2026-05-29 (As-of 2026-05-31, partial June dropped), cached as parquet under
  `_cache/`. All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies

- [`../450-andrews-pitchfork`](../450-andrews-pitchfork) — the same "the geometry forecasts"
  folklore tested with the random-entry baseline and a geometry placebo; same None × Mirage landing.
- [`../../studies/132-stochastic-oscillator`](../../studies) and the broader technical-indicator zoo
  — the %K/%D crossover is the RVI cross's sibling; most land None × Mirage because an oscillator
  fitted to past price re-describes the trend rather than forecasting it.
- The **research-method demos** (data-mining-roulette, look-ahead, curve-fitting) frame why a
  signal-vs-zero *t* is not enough; the RVI cross is a clean live example of beta masquerading as a
  momentum trigger.
