# References & literature map — Study 485 (STARC Bands)

## The claim under test

- **The folklore.** STARC bands (Stoller Average Range Channel) wrap a short SMA of the close
  in an ATR envelope: `upper = SMA + k·ATR`, `lower = SMA − k·ATR`. Because the envelope is
  scaled by *volatility* (the ATR), a close *outside* a band is read as a statistically
  over-extended move that should snap back toward the SMA — so a close **below the lower band**
  is a high-probability **buy** (reversion), the mirror of a close above the upper band being a
  sell. This is a retail/technician staple built into TradingView, MetaTrader, Thinkorswim and
  every charting suite.
- **The source.** **Manning Stoller** introduced the STARC bands in the late 1980s (the name is
  his — *St*oller *A*verage *R*ange *C*hannel). The construction is described in the standard
  technical-analysis references: Steven Achelis, *Technical Analysis from A to Z*, and the
  Fidelity / Investopedia / StockCharts ChartSchool write-ups restate the SMA-±-k·ATR rule and
  the "bands contain price, edges revert" reading.
- **Cousins.** STARC bands are one of a family of **volatility envelopes**: Keltner channels
  (EMA ± k·ATR), Bollinger bands (SMA ± k·σ), Donchian channels (rolling high/low), and Wilder's
  ATR itself. They differ only in the *center* (SMA vs EMA) and the *width metric* (ATR vs
  standard deviation vs range) and inherit the same drift/volatility-clustering confound tested
  here.

## Why this is a "theory" / mechanical-proxy study

A STARC band is fully mechanical once `(sma_n, atr_n, k)` are fixed — there is no eyeballing.
We encode the **tightest mechanical rule a proponent would accept** with the parameters Stoller
popularised (a short SMA, an ATR over a few weeks, k = 2) and state the design discipline:

- **Causal bands.** The SMA and Wilder ATR at bar *t* use only closes/ranges through *t* (the
  ATR is a Wilder EMA of the true range; the SMA a trailing mean). No future bars touch a band.
- **Objective entry.** The first close below the lower band; entry at the **next close** (one
  documented lag).
- **The honest baseline.** The only meaningful comparison on an upward-drifting index is the
  **random-entry** control (same instrument, epoch and hold), because *any* dip-buy inherits the
  drift. We add a **shuffled-ATR placebo** that permutes the band half-widths while keeping the
  SMA and the price marginal — the direct test of "does the volatility scaling matter?"

## Why the high one-sample t is not evidence

- **Drift / beta.** US equity indices have a positive unconditional daily mean; a one-sample
  *t* of a long-only entry rule against **zero** measures that drift, not the rule. See Fama &
  French on the equity premium; the desk's standing rule is *signal-vs-baseline*, never
  *signal-vs-zero*. STARC pierces also cluster in **high-volatility drawdowns** (ATR spikes), and
  long-horizon returns after volatility shocks inherit both the recovery drift and a Jensen lift —
  none of it the band's doing.
- **Data snooping on chart tools.** Lo, Mamaysky & Wang (2000, *Foundations of Technical
  Analysis*, JF) formalize testing chart patterns against a properly matched null; Sullivan,
  Timmermann & White (1999, *Data-Snooping, Technical Trading Rule Performance, and the
  Bootstrap*, JF) and White (2000, *A Reality Check for Data Snooping*, Econometrica) show how
  envelope/threshold rules manufacture significance unless raced against a fair benchmark.
- **HAC inference.** Newey & West (1987) standard errors for the one-sample mean; Welch (1947)
  two-sample *t* for the pierce-vs-random difference. Wilder (1978, *New Concepts in Technical
  Trading Systems*) is the primary source for the ATR.

## Method lineage (the desk's shared engine)

- **Causal SMA + Wilder ATR + bands.** [`strategy.starc_bands`](../starc_bands/strategy.py),
  [`strategy.atr`](../starc_bands/strategy.py), [`strategy.true_range`](../starc_bands/strategy.py).
- **Forward-return + HAC t + random baseline.** [`strategy.forward_returns`](../starc_bands/strategy.py),
  [`strategy.hac_t`](../starc_bands/strategy.py), [`strategy.run_experiment`](../starc_bands/strategy.py).
- **Geometry placebo.** [`strategy.shuffled_atr_placebo`](../starc_bands/strategy.py) — permute
  the ATR series, keep the SMA and price marginal.
- **Deterministic synthetic control.** [`data.synthetic_panel`](../starc_bands/data.py) plants a
  real lower-band reversion (knob `edge`); with `edge = 0` the detector must NOT manufacture
  significance — the offline core runs with no network.

## Data sources used here

- **yfinance** daily adjusted (total-return) closes for SPY, QQQ, IWM, DIA, GLD,
  2005-01-03 → 2026-05-29 (As-of 2026-05-31, partial June dropped), cached as parquet under
  `_cache/`. All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced
  by [`examples/verify.py`](../examples/verify.py).

## Related desk studies

- [`../../104-bollinger-reversion`](../../104-bollinger-reversion) — the sibling σ-band reversion
  rule tested with the same random-entry idiom; STARC swaps σ for ATR and lands the same place.
- [`../../450-andrews-pitchfork`](../../450-andrews-pitchfork) — the channel-respects-price
  folklore with a geometry placebo; the direct template for this study.
- [`../../178-cci`](../../178-cci) and the broader technical-indicator zoo — most land
  None/Weak × Mirage/Fragile for the same reason: an envelope fitted to past price re-describes
  the trend and the apparent edge is drift.
