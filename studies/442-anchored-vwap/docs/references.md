# References & literature map — Study 442 (Anchored VWAP)

## The claim under test

- **The folklore.** "Anchor a volume-weighted average price to a *significant* event — the
  session open, a high-volume reversal bar, an earnings gap — and the line becomes a **price
  magnet**. Price gets pulled back to it, respects it as support/resistance, and you can fade
  rejections and buy bounces off it." It is now a staple of intraday charting and a button in
  every modern platform.
- **The popular source.** Brian Shannon, *Maximum Trading Gains with Anchored VWAP: The
  Perfect Combination of Price, Time & Volume* (2022) is the canonical retail treatment that
  named and popularised "anchored VWAP" (AVWAP) as a discretionary level tool; the idea spread
  widely on trading social media as the "AVWAP magnet."
- **What VWAP actually is.** The volume-weighted average price is, by construction, the running
  average transaction price — it sits *inside* the day's price range and is therefore crossed
  constantly. "Price returns to the VWAP" is true by definition for any average; the testable
  claim is that the *level itself* forecasts the next move more than a line drawn at random.

## What we measure, and why a random-level control is the test

- **Anchored VWAP (the level).** Running volume-weighted typical price $(H+L+C)/3$ from the
  09:30 anchor — a rule-based, non-cherry-picked anchor (the discretionary versions hand-pick
  the anchor, which is the part most exposed to selection bias).
- **Post-touch reversion (the reaction).** Oriented so positive = the level is respected (a
  bounce). One-bar execution lag (act on the next bar after the touch is confirmed) — the single
  documented lag, applied once (the desk's no-look-ahead convention).
- **The random-level control / placebo.** The crux. Because an average is trivially "respected,"
  the only honest signal is whether the AVWAP beats a **horizontal level placed at random** in
  the same session range. This is the Fisher randomization logic (Efron & Tibshirani, *An
  Introduction to the Bootstrap*, 1993) applied to a level claim: a level matters only if it
  beats a line drawn at random. The same control busts other "price respects level X" folklore
  (round numbers, prior-day high, moving averages).

## Why intraday level edges die — costs and microstructure

- **The spread is the constraint.** Intraday reactions are sub-basis-point; the bid-ask spread,
  paid twice per round trip, dwarfs them. Frazzini, Israel & Moskowitz (2018, *Trading costs*)
  on the gap between paper and net returns motivates the net-vs-gross discipline; Lesmond,
  Schill & Zhou (2004, *The illusory nature of momentum profits*) is the canonical demonstration
  that a real-but-tiny effect can be entirely a trading-cost mirage.
- **VWAP's legitimate role.** VWAP is a genuinely useful *execution benchmark* — Berkowitz,
  Logue & Noser (1988, *The total cost of transactions on the NYSE*, JF) — and the basis for
  VWAP/TWAP order-slicing. The failure documented here is the *forecasting* (magnet) claim, not
  the benchmark.

## Why a *t* on an event stream needs the HAC correction

- **Autocorrelated touches.** Crossings cluster in tape time (a choppy stretch fires many in
  minutes), so a naive *t* overstates significance. We report a **Newey-West (HAC)** *t* (Newey &
  West, 1987, *A simple, positive semi-definite, heteroskedasticity and autocorrelation
  consistent covariance matrix*, Econometrica) alongside the one-sample *t*.
- **Short-span humility.** yfinance caps 5-minute history at ~60 days; a 59-session window is
  high-power per day but silent on regime variation. Harvey, Liu & Zhu (2016, *…and the
  Cross-Section of Expected Returns*, RFS) on multiple-testing inflation is the reason we hold a
  flat result to the same **t ≥ 2** bar rather than mining horizons/anchors for a hit.

## Method lineage (the desk's shared engine)

- **Anchored VWAP + touch detector.** [`strategy.session_avwap`](../anchored_vwap/strategy.py)
  and [`strategy.avwap_reactions`](../anchored_vwap/strategy.py) — the running anchored level and
  the reversion-oriented post-touch reaction.
- **Random-level control + placebo.**
  [`strategy.random_level_reactions`](../anchored_vwap/strategy.py) and
  [`strategy.placebo_pvalue`](../anchored_vwap/strategy.py) — the honest "is the line special?"
  null.
- **HAC inference.** [`strategy.hac_t`](../anchored_vwap/strategy.py) — Newey-West *t* on the
  autocorrelated event stream.
- **Deterministic synthetic control.** [`data.synthetic_panel`](../anchored_vwap/data.py) plants
  a known AVWAP magnet; with the knob at zero the inference must NOT manufacture significance —
  the offline core runs with no network.

## Data sources used here

- **yfinance** 5-minute OHLCV bars for SPY, QQQ, AAPL, MSFT, NVDA, 2026-03-30 → 2026-06-23
  (59 complete RTH sessions, partial final session dropped), cached under `_cache/bars_*_5m.parquet`.
  All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies

- **104-bollinger-reversion** and **178-cci** — fellow technical-indicator teardowns where a
  popular "price respects this line/band" rule is held to the same bar.
- **377-bid-ask-bounce** and **376-moc-imbalance** — the intraday-microstructure neighbours; the
  spread-eats-the-edge lesson is the same.
- The **research-method demos** (data-mining-roulette, multiple-testing) frame why a flat *t*
  on the most-liquid tape, plus a random-level placebo, is the honest "not supported," not a
  failure to look hard enough.
