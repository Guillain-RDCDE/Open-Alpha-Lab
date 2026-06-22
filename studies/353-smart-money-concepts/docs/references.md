# References & literature map — Study 353 (Smart-Money-Concepts: ICT order blocks & FVGs)

## The claim under test

- **The viral pitch — "ICT / Smart Money Concepts" (SMC).** A sprawling YouTube/TikTok genre
  built on Michael J. Huddleston's *Inner Circle Trader* (ICT) material. The core thesis:
  large institutions ("smart money") leave detectable **footprints** on the chart, and retail
  traders can front-run them by reading two structures in particular:
  - **Fair-value gaps (FVG / "imbalance").** A three-candle pattern where candle 1 and candle
    3 do not overlap, leaving a price "void". SMC says price is magnetically drawn back to
    "fill" or "rebalance" the gap, so the gap is a high-probability target/level.
  - **Order blocks (OB).** The last opposite-colour candle before an impulsive ("displacement")
    move; SMC says price returns to the OB zone and reverses there, so forward returns from an
    OB beat returns from an arbitrary level.
- **The falsifiable reductions.** (a) Do FVGs fill *more / faster* than a random same-size price
  zone? (b) Do forward returns from an OB zone *beat* returns from random levels? The null for
  both is a **random walk**, which also produces gaps and bounces — so a high raw fill-rate or a
  positive raw bounce proves nothing on its own.

## Why the null is a random walk (the spine of the teardown)

- **Random walks have gaps and "support" too.** Roberts (1959), *Stock-Market "Patterns" and
  Financial Analysis*, and Harry Roberts' classic exercise: chartists shown random-walk series
  confidently identify "head-and-shoulders", "support", "breakouts". A pattern a memory-less
  process reproduces is not evidence of an actor. Our synthetic positive control is the modern
  version: a driftless random walk yields ~the same FVG count, fill-rate, and OB "edge".
- **Efficient markets / random walk.** Fama (1970), *Efficient Capital Markets*; Malkiel (1973),
  *A Random Walk Down Wall Street*. If price is close to a martingale, the conditional forward
  return from any chart-defined level is ~0 — exactly what the OB book shows.
- **Mean-reversion of gaps is mechanical.** A same-size void placed *near* a diffusing price is
  very likely to be touched within a few bars purely because the price range expands like √t
  (Brownian first-passage). The 84% "fill-rate" is a first-passage probability, not a magnet.
  First-passage / barrier-hitting for Brownian motion: Karlin & Taylor (1975), *A First Course
  in Stochastic Processes*, ch. 7.

## Technical-analysis evidence base

- **The academic verdict on chart patterns is weak.** Lo, Mamaysky & Wang (2000), *Foundations
  of Technical Analysis* — some patterns carry marginal information but tiny, fragile, and
  swamped by costs. Park & Irwin (2007), *What Do We Know About the Profitability of Technical
  Analysis?* — a survey: early positive results largely vanish out-of-sample and after
  data-snooping corrections. Bailey, Borwein, López de Prado & Zhu (2014), *Pseudo-Mathematics
  and Financial Charlatanism* — backtest overfitting manufactures "edges" from noise.
- **Selection / multiple-testing.** Harvey, Liu & Zhu (2016), *…and the Cross-Section of
  Expected Returns* — with enough freely-chosen rules, spurious significance is the default; a
  raw t needs a much higher bar. Here the OB rule has free parameters (impulse size, horizon)
  and still fails *raw*, which is the cleanest possible negative.

## Method lineage (the desk's shared engine)

- **Two-proportion test.** The FVG Signal axis is a fill-*rate*: a binary outcome, tested as a
  two-proportion z of the real fill-rate against the placebo random-zone fill-rate
  (`strategy.two_prop_z`). Both arms share the tape, so the difference isolates "FVG-ness".
- **Welch t on forward returns.** The OB Signal axis is a forward return; we test OB-book mean
  minus random-entry mean with an unequal-variance Welch t (`strategy.welch_t`) and a one-sample
  t vs 0 (`strategy._summ`) — an autocorrelation-light daily, non-overlapping-ish setting.
- **Matched controls, two ways.** A *within-tape placebo* (same-width zone at a random bar) and
  a *random-walk synthetic* (`data.synthetic_ohlc`) — the offline core runs with no network and
  fixed seeds, and the synthetic is the positive control proving the detector fires on noise.
- **Costs on NAV.** One-way bps × NAV, charged on both legs of the round-trip
  (`strategy.net_of_costs`) — the Tradability axis, though the gross signal is already absent.

## Data sources used here

- **yfinance** SPY daily OHLC, 1993-01-29 → 2026-06-18 (8,404 bars), cached under `_cache/`.
  SPY is the tape ICT/SMC content is most often demonstrated on; the geometry argument is
  instrument-agnostic. All headline numbers are pinned in [`docs/results.md`](results.md) and
  reproduced by [`examples/verify.py`](../examples/verify.py).

## Related desk studies

- **[Study 301 — Triple-RSI](../../301-triple-rsi/)** and
  **[Study 351 — BTC 5-minute Polymarket momentum](../../351-btc-5m-polymarket-momentum/)**:
  viral retail strategies whose impressive headline statistic survives only until you compare it
  to the right null. Here the null is a random walk; the "footprint" does not clear it.
- The broader technical-pattern zoo on the bench (head-and-shoulders, double-tops, three
  soldiers, NR7, fractals): the same lesson — a shape a random walk draws is not a signal.
