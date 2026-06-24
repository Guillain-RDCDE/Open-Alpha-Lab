# References & literature map — Study 448 (Point & Figure price targets)

## The claim under test

- **The folklore.** "A Point & Figure chart strips out time and plots only price as columns
  of Xs and Os. On a **double-top buy** (a new X-column one box above the prior X-top), count
  the width of the base, multiply by the box size and the 3-box reversal, project it from the
  breakout — that **price target** will be reached." The target (the **horizontal count**) is
  P&F's signature, falsifiable deliverable.
- **The seminal sources.** Victor de Villiers & Owen Taylor codified the method in *The Point
  and Figure Method of Anticipating Stock Price Movements* (1933). A.W. Cohen, *How to Use the
  Three-Point Reversal Method of Point and Figure Stock Market Trading* (1947, the basis of the
  Chartcraft / Investors Intelligence service) established the **3-box reversal** and the count
  targets used here. Thomas Dorsey, *Point & Figure Charting* (1995) and Jeremy du Plessis,
  *The Definitive Guide to Point and Figure* (2005, 2012) are the modern standard references for
  the box/reversal mechanics and both the **horizontal** and **vertical** count rules.
- **Why it is testable (unlike most TA).** P&F emits a concrete *number* (a target price) and an
  implicit stop (the breakout column's far edge), so each signal can be graded objectively —
  HIT or STOP. This is what lets us pose H1 (targets get hit) as a falsifiable hypothesis rather
  than a vibe.

## The right null — a same-distance target, not a coin flip

- **The geometry trap.** A *closer* target is mechanically easier to hit, so a high hit-rate
  alone proves nothing. We control for it by placing the **same target/stop distances** on
  **random entry days** of the same instrument (the same-distance baseline). The count adds
  value only if it beats that — a continuation/momentum statement, not a geometry one.
- **Continuation / momentum lineage.** That a breakout tends to keep going is the time-series
  momentum effect: Jegadeesh & Titman (1993, *Returns to Buying Winners and Selling Losers*, JF)
  and Moskowitz, Ooi & Pedersen (2012, *Time Series Momentum*, JFE). The symmetric hit-rate edge
  we find (buy **and** sell targets beat baseline) is exactly the continuation those papers
  document — P&F's count is a folk encoding of momentum.

## Why the hit-rate is real but the P&L is a mirage

- **Drift vs edge.** The buy/sell P&L asymmetry (longs profit, shorts bleed by the same amount)
  is the fingerprint of a directional drift bet, i.e. the **equity risk premium**, not a count
  edge. Disentangling a tradable anomaly from beta is the discipline urged by Fama (1998,
  *Market efficiency, long-term returns, and behavioral finance*, JFE) and is the core of beat 6
  on this desk. A genuinely tradable count would pay on both sides; this one does not.
- **Costs / borrow.** Per-signal round-trip costs and short-leg borrow follow the net-vs-gross
  discipline of Frazzini, Israel & Moskowitz (2018, *Trading Costs*). Here costs are trivial
  (few signals/year) — the failure is the drift dependence, not cost erosion.
- **Anomaly decay / data-snooping.** Sullivan, Timmermann & White (1999, *Data-snooping,
  technical trading rule performance, and the bootstrap*, JF) and Lo, Mamaysky & Wang (2000,
  *Foundations of Technical Analysis*, JF) on grading chart patterns; McLean & Pontiff (2016)
  on post-publication decay. The same-distance baseline + placebo are our snooping guard.

## Method lineage (the desk's shared engine)

- **P&F column engine + count target.** [`strategy.build_columns`](../point_and_figure/strategy.py),
  [`strategy.signals`](../point_and_figure/strategy.py) — 3-box-reversal columns, double-top/-bottom
  signals, horizontal-count target, all lag-clean (a column is only knowable once its box-crossing
  close prints).
- **Same-distance baseline + placebo.** [`strategy.baseline_hit_rate`](../point_and_figure/strategy.py)
  and [`strategy.placebo_pvalue`](../point_and_figure/strategy.py) — the honest geometry-controlled
  null.
- **HAC inference.** [`strategy.hac_t`](../point_and_figure/strategy.py) — Newey-West one-sample *t*
  on per-signal net P&L (the autocorrelation-robust statistic the REAL stamp requires).
- **Deterministic synthetic control.** [`data.synthetic_panel`](../point_and_figure/data.py) plants a
  known continuation; with the edge set to 0 the inference must NOT manufacture a hit-rate edge over
  the baseline. The offline core runs with no network.

## Data sources used here

- **yfinance** daily auto-adjusted OHLC for SPY, ^DJI, AAPL, GLD, 2000-01-03 → 2026-06-23 (the
  in-progress final bar dropped), cached under `_cache/bars_*_1d.parquet`. All headline numbers
  are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies

- [447-gann-angles](../../447-gann-angles/) — the Gann 1x1 fixed-slope line, the closest sibling
  (another mechanical chart geometry; same place a chart-folklore method lands honestly).
- [444-dow-theory](../../444-dow-theory/) — Industrials/Transports higher-high confirmation, a
  theory-kind teardown in the same family.
- [104-bollinger-reversion](../../104-bollinger-reversion/) — "price always returns to the bands,"
  another testable price-level claim that the drift baseline busts.
- The [research-method demos](../../343-data-mining-roulette/) frame why a same-distance baseline
  (not a coin flip) is the correct null for any price-target claim.
