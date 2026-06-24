# References & literature map — Study 416 (Rounding Bottom / saucer base)

## The claim under test

- **The folk recipe.** The "rounding bottom" (a.k.a. *saucer base* or *bowl*) is a staple of
  classical chart analysis: a long, smooth, U-shaped price base said to mark a transition from
  *distribution/capitulation* to quiet *accumulation*; the confirmed **breakout above the rim**
  (the prior resistance level) is sold as the buy signal that the markup phase has begun. We
  steelman it as the testable promise: *the forward return after a confirmed rounding-bottom
  breakout exceeds the stock's unconditional base rate (a random day in the same name).*

## Where the figure comes from

- **Edwards & Magee, *Technical Analysis of Stock Trends* (1948, and many later editions)** —
  the canonical source for reversal "patterns" including the rounding bottom / saucer. The
  pattern is defined visually (a gradual, symmetric bowl), which is precisely why a *mechanical*
  surrogate is needed to test it at all.
- **John J. Murphy, *Technical Analysis of the Financial Markets* (1999)** and **Martin Pring,
  *Technical Analysis Explained*** — restate the saucer/rounding-bottom and the
  accumulation-then-breakout narrative for modern audiences.
- **William O'Neil, *How to Make Money in Stocks* (CAN SLIM)** — the "cup" (and cup-with-handle)
  base is the saucer's close cousin; the breakout-from-a-base discipline is the same idea.

## Why the steelman almost works — and the trap it walks into

- **Equity drift premium.** Stocks drift up on average, so *any* long-only rule shows a positive
  forward return. Testing a bullish chart pattern against **zero** confounds the pattern with
  this premium. The correct benchmark is the **base rate** — the unconditional forward return of
  the same names — which isolates the pattern's marginal contribution. This is the central
  methodological point of the study.
- **The objective evidence on chart patterns is thin.** Lo, Mamaysky & Wang (2000,
  *Foundations of Technical Analysis*, Journal of Finance) built kernel-smoothing detectors for
  classic figures (head-and-shoulders, etc.) and found *some* informativeness but weak,
  inconsistent, and largely arbitraged once costs and conditioning are honest. Bulkowski's
  *Encyclopedia of Chart Patterns* tabulates pattern "success rates" but without a base-rate
  control or significance testing — exactly the gap this study fills.
- **Anchoring / apophenia.** Roberts (1959) and the random-walk literature show humans see
  "patterns" (including saucers and head-and-shoulders) in pure random walks; a detector that
  fires on a random walk with drift will *inherit the drift* and look bullish for free.

## How we test it (and why these checks)

- **Mechanical detector.** A least-squares **parabola** fit on the trailing window
  ([`strategy.detect_breakouts`](../rounding_bottom/strategy.py)) with positive curvature,
  good fit (R²), an **interior vertex** (a true bowl, not a slide), minimum depth, and a
  **confirmed rim breakout** (first close above the left-rim resistance). One execution lag
  (enter t+1 open).
- **Base-rate benchmark + Welch t.** [`strategy.base_rate_returns`](../rounding_bottom/strategy.py)
  and [`strategy.welch_t`](../rounding_bottom/strategy.py) — the breakout return is compared to
  the every-bar forward return, not to zero. **Newey-West HAC** *t*
  ([`strategy.hac_t`](../rounding_bottom/strategy.py)) handles the overlap in fixed-horizon
  windows (Newey & West, 1987).
- **Date-shuffle placebo.** [`strategy.permutation_placebo`](../rounding_bottom/strategy.py) —
  draw the same number of *random* entry dates per name and pool their forward returns; the
  honest "is the *shape* doing anything beyond picking that many dates in this drifting name?"
  null (Fisher randomization logic; Efron & Tibshirani, *An Introduction to the Bootstrap*, 1993).
- **Synthetic positive control.** [`data.synthetic_panel`](../rounding_bottom/data.py) plants
  the saucer *shape*; with `edge = 0` (shape, no continuation) the inference must NOT manufacture
  an edge, and with `edge > 0` it must recover the planted drift — the offline core runs with no
  network.

## Data sources used here

- **yfinance** daily adjusted OHLC for SPY + 29 long-listed US large-caps, 2004-01-02 →
  2026-06-23, cached under `_cache/rb_<TICKER>_1d.parquet`. All headline numbers are pinned in
  [`docs/results.md`](results.md) (panel fingerprint `69cb517da40d`) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies

- [`../178-cci`](../178-cci) and [`../104-bollinger-reversion`](../104-bollinger-reversion) —
  other mechanical-rule teardowns of classical technical signals, same base-rate / control idiom.
- [`../363-pead-drift`](../363-pead-drift) — the rare event-study that *does* clear the bar; the
  contrast (a fundamental surprise that drifts vs a chart shape that does not) is instructive.
- The **research-method demos** (data-mining-roulette, multiple-testing, look-ahead) frame why
  benchmarking a long-only pattern against **zero** instead of the base rate manufactures a
  false positive — the exact trap a rounding bottom walks into.
