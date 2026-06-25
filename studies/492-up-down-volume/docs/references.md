# References & literature map — Study 492 (Up-Down-Volume)

## The claim under test

- **The folklore.** Aggregate the day's trading volume into **up-volume** (volume of advancing
  issues) and **down-volume** (volume of declining issues). When **down-volume overwhelms
  up-volume** market-wide — a *selling climax* — panic is said to be exhausting itself and a
  bounce follows; the mirror case, a *buying climax* (up-volume swamping down-volume, a blow-off),
  precedes weakness. The up/down-volume ratio is the volume half of the **Arms index (TRIN)** and
  a staple of classic tape-reading. The claim is that this breadth ratio *forecasts* the index.
- **The sources.**
  - **Richard W. Arms Jr.** introduced the **Arms index / TRIN** (1967, *Profits in Volume*),
    which combines the advance/decline *count* ratio with the up/down *volume* ratio; extreme
    TRIN readings are read as climaxes. The up/down-volume ratio here is the volume leg of that.
  - **Joseph Granville** popularised volume-confirms-price ideas and **On-Balance Volume** (1963,
    *Granville's New Key to Stock Market Profits*) — the lineage behind "volume breadth leads."
  - **Richard Wyckoff** (1910s–30s) named the **selling climax / buying climax** as turning-point
    tape phenomena, the qualitative ancestor of the mechanical rule tested here.
  - Modern restatements: Investopedia ("Up/Down Volume Ratio", "Arms Index"), StockCharts
    ChartSchool (TRIN, advance/decline volume), and John Murphy, *Technical Analysis of the
    Financial Markets* (breadth chapter).

## Why this is a "breadth-proxy" study

True up/down volume aggregates **every listed issue** on an exchange. We do not have that feed
offline, so — following the desk's design for breadth claims — we build the **tightest
reproducible proxy**: up/down volume across a basket of liquid SPDR sector ETFs (XLK XLF XLE XLV
XLI XLY XLP XLU XLB) plus SPY. We state the cap explicitly:

- **Proxy, not the tape.** Nine sector funds cannot reproduce the advance/decline volume of
  thousands of stocks; a richer breadth feed could in principle sharpen the signal. This is the
  honest, fully-cached version, and it bounds (does not overstate) the method.
- **The honest baseline.** The only meaningful comparison on an upward-drifting index is the
  **random-entry** control (same instrument, epoch and hold), because *any* dip-buy inherits the
  drift. We add a **shuffled-volume timing placebo** that permutes the up/down-volume share in
  time (marginal preserved, alignment destroyed) — the direct test of "does the up/down *timing*
  carry information?"

## Why the high one-sample t is not evidence

- **Drift / beta.** US equity indices have a positive unconditional daily mean; a one-sample *t*
  of a long-only entry rule against **zero** measures that drift, not the rule. See Fama & French
  on the equity premium; the desk's standing rule is *signal-vs-baseline*, never *signal-vs-zero*.
- **Mean reversion at the entry, not breadth.** A selling climax fires after a market drop, where
  short-horizon reversal already lives (Lo & MacKinlay 1988, *Stock Market Prices Do Not Follow
  Random Walks*, RFS); the random-day baseline soaks up most of that, which is exactly why the
  climax-minus-random *t* stays below 2.
- **Data snooping on technical rules.** Lo, Mamaysky & Wang (2000, *Foundations of Technical
  Analysis*, JF) formalise testing chart/indicator rules against a properly matched null;
  Sullivan, Timmermann & White (1999, *Data-Snooping, Technical Trading Rule Performance, and the
  Bootstrap*, JF) and White (2000, *A Reality Check for Data Snooping*, Econometrica) show how
  fitted rules manufacture significance unless raced against a fair benchmark.
- **HAC inference.** Newey & West (1987) standard errors for the one-sample mean; Welch (1947)
  two-sample *t* for the climax-vs-random difference.

## Method lineage (the desk's shared engine)

- **Breadth indicator + climax entries.** [`strategy.up_down_volume`](../up_down_volume/strategy.py),
  [`strategy.climax_entries`](../up_down_volume/strategy.py) — the up-volume share with a
  past-only rolling-quantile threshold (no look-ahead).
- **Forward-return + HAC t + random baseline.** [`strategy.forward_returns`](../up_down_volume/strategy.py),
  [`strategy.hac_t`](../up_down_volume/strategy.py), [`strategy.run_experiment`](../up_down_volume/strategy.py).
- **Timing placebo.** [`strategy.shuffled_volume_placebo`](../up_down_volume/strategy.py) —
  permute the up/down-volume share in time, keep the marginal.
- **Deterministic synthetic control.** [`data.synthetic_panel`](../up_down_volume/data.py) plants
  a real selling-climax bounce (knob `edge`); with `edge = 0` the detector must NOT manufacture
  significance — the offline core runs with no network.

## Data sources used here

- **yfinance** daily adjusted (total-return) OHLC for SPY/QQQ/IWM/DIA/GLD and daily **OHLCV** for
  the breadth basket (SPY + 9 SPDR sector ETFs), 2005-01-03 → 2026-05-29 (As-of 2026-05-31),
  cached as parquet under `_cache/`. All headline numbers are pinned in
  [`docs/results.md`](results.md) and reproduced by [`examples/verify.py`](../examples/verify.py).

## Related desk studies

- [`../../450-andrews-pitchfork`](../../450-andrews-pitchfork) — the desk template; same
  random-entry + placebo + synthetic-control idiom.
- [`../../104-bollinger-reversion`](../../104-bollinger-reversion) — the "buy the panic / band
  reverts" folklore tested with the random-entry baseline.
- [`../../109-obv-divergence`](../../109-obv-divergence) and the broader volume-indicator
  family — volume signals re-describe price and mostly land None × Mirage.
- The **research-method demos** (data-mining-roulette, look-ahead, curve-fitting) frame why a
  signal-vs-zero *t* — or a significant permutation placebo that still fails the drift baseline —
  is not a tradable edge.
