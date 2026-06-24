# References & literature map — Study 406 (Harami)

## The claim under test

- **The folk recipe.** The harami ("pregnant" in Japanese) is a two-candle reversal: a large
  real body followed by a small real body that sits *inside* it. A **bullish harami** (large
  down day, small up day inside) after a downtrend is read as a bottom — buy; a **bearish
  harami** (large up day, small down day inside) after an uptrend is read as a top — sell. The
  pattern is canonical in Steve Nison's *Japanese Candlestick Charting Techniques* (1991), the
  book that introduced candlesticks to Western markets, and is taught on essentially every
  charting platform (Investopedia, StockCharts, TradingView). We steelman it as: *the harami
  inside bar carries directional reversal information on daily equity bars — long after
  bullish, short after bearish — that exceeds the unconditional drift, net of costs.*

## Why the steelman is *almost* coherent — the real effect it leans on

- **The inside bar as a volatility/conviction contraction.** A small body inside a large one
  genuinely reflects a one-day collapse in directional conviction — a real microstructure fact.
  The leap is from "conviction paused" to "trend reverses," which is what the data must show.
- **Short-horizon reversal.** Jegadeesh (1990), *"Evidence of Predictable Behavior of Security
  Returns"* (Journal of Finance), and Lehmann (1990), *"Fads, Martingales, and Market
  Efficiency"* (QJE), document one-week/one-month reversal at the individual-stock level — the
  effect a daily reversal pattern might proxy. But these are weak and largely arbitraged away
  in liquid large-caps over the modern sample.
- **The market's upward drift.** Over 2005–2026 US large-caps drifted up structurally. Any
  long-biased rule inherits that drift — which is precisely the confound that makes a one-legged
  "reversal" look real. Dimson, Marsh & Staunton (*Triumph of the Optimists*, 2002) document
  the long-run equity drift the bullish leg rides.

## The failure mode exposed

- **One-legged = beta, not a reversal.** The bullish leg works; the bearish leg is
  *wrong-signed* (shorting the "top" loses money as the stock keeps rising). A true reversal
  pattern must work both ways. The bullish leg's *excess over the unconditional drift* clears
  *t* = 2 only at the 1-day horizon — most of the multi-day gain is the market drift, not a flip.
- **Candlestick patterns under formal testing.** Marshall, Young & Rose (2006), *"Candlestick
  Technical Trading Strategies: Can They Create Value for Investors?"* (Journal of Banking &
  Finance), find candlestick patterns (harami included) add no value on DJIA stocks. Horton
  (2009), *"Stars, crows, and doji"* (Journal of Economics and Finance), and Zhu, Atri & Yegen
  (2016) reach similar null/weak conclusions. Lu, Shiu & Liu (2012) find weak, market-dependent
  results that vanish out of sample.
- **Data-snooping & out-of-sample fragility.** Nison calibrated candlestick lore on Japanese
  rice and 20th-century markets. Brock, Lakonishok & LeBaron (1992) and Sullivan, Timmermann &
  White (1999), *"Data-Snooping, Technical Trading Rule Performance, and the Bootstrap"*
  (Journal of Finance), show how much apparent technical-rule success disappears once selection
  is corrected — the spirit of our coin-signed label-shuffle placebo. Park & Irwin (2007),
  *"What Do We Know About the Profitability of Technical Analysis?"* (Journal of Economic
  Surveys), survey the broad pattern: results hinge on asset class, period, and costs.

## Method lineage (the desk's shared engine)

- **HAC / Newey-West t-stat.** Newey & West (1987), *"A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix"* (Econometrica) —
  [`strategy.hac_t`](../harami_pattern/strategy.py) and
  [`quantlab.analytics`](../../../quantlab/analytics.py).
- **Label-shuffle / bootstrap placebo.** Politis & Romano (1994), *"The Stationary Bootstrap"*
  (JASA); White (2000) Reality Check spirit — [`strategy.placebo_pvalue`](../harami_pattern/strategy.py).
- **Reproducibility stamp.** [`quantlab/repro.py`](../../../quantlab/repro.py) — the as-of
  freeze and content fingerprint each headline run carries.

## Data sources used here

- **Yahoo! Finance daily bars** (via `yfinance`, `auto_adjust=True`), 2005-01-03 → 2026-06-18
  across 30 liquid tapes (29 large-caps + SPY). The offline reproducible core and the synthetic
  control run on the deterministic [`data.synthetic_panel`](../harami_pattern/data.py)
  generator, never the network. Each headline is pinned with an as-of date and a panel content
  fingerprint (see [`docs/results.md`](results.md)).

## Related desk studies

- **[Study 402 — Engulfing-Pattern](../../402-engulfing-pattern/)**: the *opposite* geometry
  (the second body swallows the first) — also a busted reversal, same harness.
- **[Study 403 — Hammer / Hanging-Man](../../403-hammer-hanging-man/)**: a single-candle
  reversal in the same candlestick family.
- **[Study 404 — Shooting-Star](../../404-shooting-star/)** and
  **[Study 405 — Doji-Reversal](../../405-doji-reversal/)**: the rest of the candle-reversal
  flight, identical event-study treatment.
- **[Study 178 — CCI](../../178-cci/)**: another mean-reversion / overbought-oversold rule
  tested against a coin — same "does a technical rule add information?" question.
