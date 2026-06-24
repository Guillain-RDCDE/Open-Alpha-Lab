# References & literature map — Study 440 (Floor-Trader Pivot Points)

## The claim under test

- **The folklore.** Compute the **pivot** `P = (prior High + Low + Close) / 3` each morning, plus
  resistances `R1, R2, R3` above and supports `S1, S2, S3` below; intraday, price *respects*
  these levels — bouncing off support, stalling at resistance — so you fade the touch. The
  recipe predates screens (floor traders used it for quick mental support/resistance) and is now
  built into essentially every charting platform.
- **The popular sources.** John L. Person, *A Complete Guide to Technical Trading Tactics* (2004),
  popularised pivot trading for retail. The pivot formula and its R/S extensions are documented in
  Investopedia's "Pivot Points" entry and in StockCharts' ChartSchool ("Pivot Points"), and the
  "S1 always holds / R1 caps the rally" lore is ubiquitous in day-trading education. There is no
  peer-reviewed foundation; it is a charting convention, not an academic anomaly.

## What we measure, and why the random-line control is the whole test

- **Touch → bounce hit-rate.** Any horizontal line inside a session's range gets touched and
  "bounced off" some fraction of the time, by the geometry of a wiggly price path. The only
  meaningful statistic is the pivots' bounce rate **in excess of** a line placed at a uniform-
  random price on the *same* tape (random side), run through the identical machinery — the control
  absorbs intraday volatility, drift, tolerance and horizon, isolating the pivot *arithmetic*.
- **Support/resistance as a testable object.** The academic treatment of round-number / level
  support is Osler (2000, *Support for Resistance: Technical Analysis and Intraday Exchange
  Rates*, FRBNY Economic Policy Review) and Osler (2003, *Currency Orders and Exchange-Rate
  Dynamics*, J. Finance), which find *some* clustering of orders at round numbers in FX — a far
  stronger prior than floor-trader pivots, and still fragile. Our null result for pivots is
  consistent with the broader skepticism.
- **No look-ahead / one execution lag.** Pivots for session *D* use only session *D−1*'s H/L/C;
  the fade is entered **one bar after** the touch and held a fixed horizon — the standard
  event-study convention.

## Why technical levels usually fail the audit

- **The randomness benchmark.** Lo, Mamaysky & Wang (2000, *Foundations of Technical Analysis*,
  J. Finance) formalised testing chart patterns against a properly specified null; most "levels"
  add nothing once the right baseline is used. Our permutation placebo is that benchmark for
  pivots.
- **Costs kill fast intraday rules.** An intraday fade round-trips in minutes; the bid-ask spread
  paid twice dominates a basis-point gross edge. The net-vs-gross discipline follows the broader
  cost literature (e.g. Frazzini, Israel & Moskowitz, 2018, *Trading Costs*).
- **Adaptive / efficient markets.** Lo (2004, *The Adaptive Markets Hypothesis*) explains why a
  widely-shared, costless recipe like a published pivot formula should be arbitraged away — our
  result is what that predicts.

## Method lineage (the desk's shared engine)

- **Touch / bounce detection + per-level breakout.**
  [`strategy.pivot_touch_events`](../pivot_points/strategy.py) and
  [`strategy.bounce_after`](../pivot_points/strategy.py).
- **Random-line control + permutation placebo.**
  [`strategy.control_touch_events`](../pivot_points/strategy.py) and
  [`strategy.permutation_placebo`](../pivot_points/strategy.py) — the honest "could an arbitrary
  line look this good?" null.
- **HAC (Newey-West) one-sample t.** [`strategy.hac_t`](../pivot_points/strategy.py) on the fade
  return (touches cluster within sessions, so a plain *t* is not trusted).
- **Deterministic synthetic control.** [`data.synthetic_panel`](../pivot_points/data.py) plants a
  genuine bounce away from the pivots; with the edge at zero the test must NOT beat the control,
  and a planted bounce must light up — the offline core runs with no network.

## Data sources used here

- **yfinance** 5-minute OHLC bars for SPY, QQQ, AAPL, MSFT, IWM, 2026-03-30 → 2026-06-23 (the
  ~60-day 5-minute cap), cached under `_cache/bars_*_5m.parquet`. All headline numbers are pinned
  in [`docs/results.md`](results.md) and reproduced by [`examples/verify.py`](../examples/verify.py).

## Related desk studies

- [`../104-bollinger-reversion`](../104-bollinger-reversion) — another "price returns to the band"
  level claim, tested against a random-entry control (same spirit: a level is only special if it
  beats random).
- [`../116-power-hour`](../116-power-hour) — intraday continuation on Yahoo intraday bars; the
  template for the short-span intraday caveat used here.
- [`../376-moc-imbalance`](../376-moc-imbalance) and
  [`../377-bid-ask-bounce`](../377-bid-ask-bounce) — intraday microstructure studies where the
  spread is the binding constraint, exactly as here.
- The **research-method demos** (data-mining-roulette, multiple-testing) frame why a single name
  brushing significance in one direction is noise, not signal.
