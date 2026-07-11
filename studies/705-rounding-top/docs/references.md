# References & literature map — Study 705 (Rounding Top / dome distribution)

## The claim under test

- **The folk recipe.** The "rounding top" (a.k.a. *dome distribution*) is the bearish
  mirror of the rounding bottom / saucer: a long, smooth, inverted-U arc where price
  slowly rolls over — a moving average with a steadily **declining slope** — said to
  mark quiet **distribution** (smart money exiting into strength while the crowd is
  still buying). The confirmed **breakdown below the rim** (the support level set at
  the start of the arc) is sold as the signal that markdown has begun. We steelman it
  as the testable promise: *the forward return after a confirmed rounding-top
  breakdown, taken short, exceeds the base rate of shorting a random day in the same
  name.*

## Where the figure comes from

- **Edwards & Magee, *Technical Analysis of Stock Trends* (1948, and many later
  editions)** — the canonical source for reversal "patterns"; the rounding top is
  presented as the bearish counterpart to the rounding bottom / saucer, with the same
  visual, non-mechanical definition — which is precisely why a *mechanical* surrogate
  is needed to test it at all.
- **John J. Murphy, *Technical Analysis of the Financial Markets* (1999)** and
  **Martin Pring, *Technical Analysis Explained*** — restate the dome/rounding-top and
  the distribution-then-markdown narrative for modern audiences.
- **Thomas Bulkowski, *Encyclopedia of Chart Patterns*** — catalogs the rounding top
  among reversal patterns with tabulated "success rates," but without a base-rate
  control or significance testing — exactly the gap this study fills.

## Why the steelman almost works — and the trap it walks into

- **Equity drift premium, mirrored.** Stocks drift up on average, so *any* short-only
  rule shows a *negative* expected return before you even ask whether the pattern
  means anything — shorting is fighting the tape's own upward pull. Testing a bearish
  chart pattern against **zero** confounds the pattern with this drift-against-you
  cost. The correct benchmark is the **base rate of shorting a random day** in the
  same names, which isolates the pattern's marginal contribution. This is the central
  methodological point of the study (the same point Study 416 makes for the bullish
  saucer, from the other side).
- **The objective evidence on chart patterns is thin.** Lo, Mamaysky & Wang (2000,
  *Foundations of Technical Analysis*, Journal of Finance) built kernel-smoothing
  detectors for classic figures and found *some* informativeness but weak,
  inconsistent, and largely arbitraged once costs and conditioning are honest.
- **Anchoring / apophenia.** Roberts (1959) and the random-walk literature show humans
  see "patterns" (domes, head-and-shoulders, double tops) in pure random walks; a
  detector that fires on a random walk *with drift* inherits that drift, and for a
  short-side pattern the drift works *against* the believer's thesis by default — the
  fair null here is a genuinely *harder* bar to clear than the bullish mirror's.
- **A subtler trap in the plant itself.** Building the synthetic positive control for
  this study surfaced a real methodological lesson, documented in
  [`rounding_top/data.py`](../rounding_top/data.py): a naively-anchored planted dome
  can leak its own tail-end decline into the "clean" forward window whenever the
  detector's sliding 90-day lookback references a still-elevated point *inside* the
  dome rather than the true support — a look-ahead-flavoured artefact that inflated
  the null's *t* past 2 in most of 20 seeds before the fix (a wide flat pad at the
  support level, anchored to the exact bar the live detector references). Machinery
  proofs need the same scrutiny as the live detector.

## How we test it (and why these checks)

- **Mechanical detector.** A least-squares **parabola** fit on the trailing window
  ([`strategy.detect_breakdowns`](../rounding_top/strategy.py)) with *negative*
  curvature, good fit (R²), an **interior vertex** (a true dome, not a slide), minimum
  height above support, and a **confirmed breakdown** (first close below the left-rim
  support). One execution lag (short at t+1 open).
- **Base-rate benchmark + Welch t.** [`strategy.base_rate_returns`](../rounding_top/strategy.py)
  and [`strategy.welch_t`](../rounding_top/strategy.py) — the breakdown short return is
  compared to the every-bar SHORT forward return, not to zero. **Newey-West HAC** *t*
  ([`strategy.hac_t`](../rounding_top/strategy.py)) handles the overlap in fixed-horizon
  windows (Newey & West, 1987).
- **Date-shuffle placebo.** [`strategy.permutation_placebo`](../rounding_top/strategy.py) —
  draw the same number of *random* entry dates per name, short them the same way, and
  pool their forward returns; the honest "is the *shape* doing anything beyond picking
  that many dates to short in this drifting name?" null (Fisher randomization logic;
  Efron & Tibshirani, *An Introduction to the Bootstrap*, 1993).
- **Costs and borrow.** One-way costs × NAV on the round trip *and* an annualized
  stock-borrow rate pro-rated over the holding horizon — the house-rule cost a short
  position pays that a long position (Study 416) does not.
- **Synthetic positive control.** [`data.synthetic_panel`](../rounding_top/data.py) plants
  the dome *shape*; with `edge = 0` (shape, no continuation) the inference must NOT
  manufacture a decline edge, and with `edge < 0` it must recover the planted decline —
  the offline core runs with no network. Verified clean across 20 seeds (|t| ≥ 2 in
  1/20) before being trusted as this study's machinery proof.

## Data sources used here

- **yfinance** daily adjusted OHLC for SPY + 29 long-listed US large-caps, 2004-01-02 →
  2026-06-30, cached under `_cache/rt_<TICKER>_1d.parquet`. All headline numbers are
  pinned in [`docs/results.md`](results.md) (panel fingerprint `9da1b7ce7758`) and
  reproduced by [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is NOT)

- [416-rounding-bottom](../416-rounding-bottom/) — the **bullish saucer mirror** of
  this exact study: same parabola-fit machinery, opposite curvature (positive), a
  confirmed *breakout* above the rim, long-only. Read together they are a matched
  pair; both land on the same verdict (`None x Mirage`) from opposite directions —
  the chart-pattern trap runs in both signs.
- [189-double-top](../189-double-top/) — a **different geometry entirely**: two local
  peaks of similar height separated by a trough (an "M"), confirmed on a neckline
  break, detected via `scipy.signal.find_peaks`. A rounding top is one *smooth arc*,
  not two discrete peaks; this study's parabola detector would not flag a double-top
  and vice versa. Also lands on `None`, via a different detector and a Bonferroni
  multiple-testing correction across six pattern/horizon tests.
- [465-diamond-top](../465-diamond-top/) — a **broadening-then-narrowing** rhombus
  reversal (a volatility-contraction geometry), not a single smooth roll-over. Shares
  the bearish-reversal claim but a structurally different shape and detector.
- [706-diamond-bottom](../706-diamond-bottom/) — the diamond's bullish mirror, exactly
  as this study is 416's bearish mirror. Neither diamond study tests a slow,
  monotone-curvature dome.

None of the siblings test a **single smooth inverted-U roll-over confirmed by a
support breakdown** — the rounding-top claim is this study's own axis, and it is the
short-side twin of 416, not a restatement of 189, 465 or 706.
