# References & literature map — Study 429 (Mass Index, Donald Dorsey)

## The claim's source

- **Donald Dorsey**, *"The Mass Index,"* **Technical Analysis of Stocks & Commodities** (June
  1992) — the original publication. Dorsey's thesis is that **range expansion**, not price
  direction, foretells reversals: he sums a 25-day window of the ratio of a 9-day EMA of the
  high-low range to a 9-day EMA *of that EMA*, and calls the **"reversal bulge"** when the index
  rises **above 27** and then falls back **below 26.5** — supposedly the signature of a trend
  about to flip.
- **Investopedia — "Mass Index"** and **StockCharts / TradingView indicator docs** — the modern
  restatements of the folk rule (the 27 / 26.5 bulge thresholds, "use it with a trend filter and
  fade the prevailing trend when the bulge fires"). These are the steelmanned framings we test.

## Range, volatility, and "reversal" indicators

- **Wilder, J. Welles (1978), *New Concepts in Technical Trading Systems*** — the origin of
  range-based indicators (ATR, DMI); the Mass Index belongs to this family of *range*-driven tools,
  which measure the **magnitude** of moves, not their **direction**.
- **Bollinger, J. (2001), *Bollinger on Bollinger Bands*** — the canonical "the squeeze precedes a
  big move" claim; like the Mass Index it conflates a *volatility* signal (a move is coming) with a
  *directional* signal (which way) — the exact category error this study isolates.
- **Mandelbrot, B. (1963), "The Variation of Certain Speculative Prices,"** *Journal of Business*,
  and the GARCH literature (**Engle 1982**, **Bollerslev 1986**) — **volatility clusters**: wide
  ranges follow wide ranges. The Mass Index bulge is, mechanically, a volatility-clustering
  detector; the open question is whether clustering carries any *forward-return* information (it
  doesn't, here).

## Why a high one-sample *t* still needs a base rate + placebo

- **Welch, B. (1947), "The generalization of Student's problem…"** — the Welch *t* we use for the
  bulge sample vs the non-bulge (base-rate) sample. A one-sample *t* against zero can be "passed"
  by mere market drift; the *t* vs base is the honest contrast.
- **Efron, B. & Tibshirani, R. (1993), *An Introduction to the Bootstrap*** & Fisher's
  randomization logic — the **label-shuffle placebo** (draw the same number of random trigger dates,
  recompute the mean forward return): the right null for a **rare event** (only 36 bulges in 21
  years).
- **Harvey, Liu & Zhu (2016), "…and the Cross-Section of Expected Returns,"** *RFS* — multiple-
  testing discipline: a single sub-2 *t* on a hand-picked horizon, on a 36-event sample, is exactly
  what *noise* produces; the sign-flip across the panel confirms it.

## Shared method (the desk's protocol)

- **Newey, W. & West, K. (1987), "A Simple, Positive Semi-Definite, Heteroskedasticity and
  Autocorrelation Consistent Covariance Matrix,"** *Econometrica* — the HAC *t* on the daily
  (fade − buy-hold) return difference.
- **Sharpe, W. (1994), "The Sharpe Ratio,"** *Journal of Portfolio Management* — excess-vs-excess
  risk-adjusted comparison for the part-time-in-cash timing race.
- House method: [`../../../METHODOLOGY.md`](../../../METHODOLOGY.md) — the seven beats, the inference
  bar (REAL needs robust *t* ≥ 2 on the real tape; a synthetic control proves the machinery, never
  the market), and the verdict rubric.

## Method lineage (this study's engine)

- **Mass Index + bulge trigger.** [`strategy.mass_index`](../mass_index/strategy.py) and
  [`strategy.bulge_signal`](../mass_index/strategy.py) — Dorsey's exact recipe and the 27→26.5
  completing-bar trigger.
- **Event study + label-shuffle placebo.** [`strategy.event_study`](../mass_index/strategy.py) and
  [`strategy.placebo_pvalue`](../mass_index/strategy.py) — forward returns after a bulge vs the base
  rate, raw and trend-signed, with the rare-event null.
- **Deterministic synthetic control.** [`data.synthetic_panel`](../mass_index/data.py) plants a
  real post-bulge reversal (knob `edge`); with `edge = 0` the inference must NOT manufacture
  significance — the offline core runs with no network.

## Data sources used here

- **yfinance** daily auto-adjusted OHLCV for SPY (+ QQQ, IWM, DIA, GLD for the panel),
  2005-01-03 → 2026-06-23, cached under `_cache/bars_*_1d.parquet`. All headline numbers are pinned
  in [`docs/results.md`](results.md) and reproduced by [`examples/verify.py`](../examples/verify.py).

## Related desk studies

- [`../430-klinger-oscillator`](../430-klinger-oscillator) — sibling volume oscillator built in the
  same lot; another "indicator that promises foresight but lags" lands NONE × MIRAGE.
- [`../423-force-index`](../423-force-index) — sibling price×volume timing rule; the same
  excess-vs-excess race idiom and "flags reversals? — busted" myth axis.
- [`../178-cci`](../178-cci) — Commodity Channel Index: another range/extreme oscillator that fails
  the follow-the-extreme test.
- [`../363-pead-drift`](../363-pead-drift) — the gold-standard real-tape event study whose shape,
  base-rate contrast, and synthetic-control discipline this one follows.
- [`../343-data-mining-roulette`](../343-data-mining-roulette) — why a single sub-2 *t* on a chosen
  horizon, on 36 events, is what noise looks like.
