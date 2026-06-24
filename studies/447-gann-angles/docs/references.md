# References & literature map — Study 447 (Gann Angles, the 1x1 line)

## The claim under test

- **The folklore.** "Markets move in geometric proportion to time. Draw a **1x1 angle** —
  one unit of price per one unit of time — from a significant pivot low, and price will
  *respect* it: while price rides **above** the 1x1 the trend is strong (the line is
  support); once price falls **below** it the trend has turned (the line is resistance).
  The 1x1 is the spine of the **Gann fan** (the 2x1, 1x2, 3x1 … are steeper/shallower
  lines off the same pivot)." It is among the most mystical, course-sold tools in technical
  analysis.
- **The source.** William Delbert (W.D.) Gann, *45 Years in Wall Street* (1949) and *Truth
  of the Stock Tape* (1923); the modern packaging is the "Gann fan / Gann angles" overlay
  in every charting platform. Gann tied the 1x1 to a broader (and far less falsifiable)
  doctrine of "squaring price and time" and anniversary-date cycles.
- **Why the 1x1 is the testable piece.** Most Gann lore is irreducibly subjective (which
  pivot? which scale? which cycle?). The **1x1 angle is the one falsifiable kernel**: a
  fixed-slope line from a mechanically-defined pivot. We encode exactly that and let the
  tape rule on it; the subjective dressing (manual pivots, custom scales) is the believer's
  escape hatch, which is precisely why we fix the pivot rule in advance.

## What we measure, and why this is the fair encoding

- **The pivot.** A confirmed centred-window swing low (the lowest low in a 21-bar window
  centred on the bar), usable only after the window closes — no look-ahead. This is the
  standard mechanical analogue of "a significant low" a practitioner would anchor on.
- **The 1x1 slope.** An **arithmetic** price increment per bar (the literal 45° line a
  charting package draws), calibrated to the instrument's own scale as full price range ÷
  number of bars — i.e. a genuine rise-over-run "one unit per one unit." This is the only
  free choice and it is fixed once, structurally, not fitted per day.
- **The trade & the race.** Long when the close is above the line, flat otherwise, entered
  one bar later (one documented execution lag), costed at 2 bps one-way × turnover. The
  comparison is against **buy-and-hold of the same total-return series**, so the contest is
  apples-to-apples (a long-only timer inherits most of buy-and-hold by construction — see
  the placebo).

## Why a negative/near-zero result still needs a placebo

- **The exposure confound.** A rule in the market ~70% of the time tracks buy-and-hold
  whether or not its *timing* has information. The honest null is a **random regime of the
  same on-fraction and average run length** (a matched two-state Markov switch); we ask how
  often such a random rule beats the real angle. This is the desk-standard "is it the
  signal, or just the exposure?" test (Fisher's randomization logic; Efron & Tibshirani,
  *An Introduction to the Bootstrap*, 1993).
- **Autocorrelation-robust inference.** Daily spreads are serially correlated, so the
  significance test is a **Newey-West (HAC)** one-sample *t* (Newey & West, 1987,
  *Econometrica*) with the standard Andrews-style bandwidth — the same statistic the desk
  uses across studies.

## The academic backdrop — why "no" is the prior

- **Weak-form efficiency.** Fama (1970, *Efficient Capital Markets*, JF) and the broad
  technical-analysis literature give fixed-geometry chart rules a very low prior of
  out-of-sample power. The most rigorous TA survey, Lo, Mamaysky & Wang (2000, *Foundations
  of Technical Analysis*, JF), finds only weak, mostly informational content in chart
  patterns — and nothing supporting fixed-slope angles.
- **Data-snooping discipline.** Sullivan, Timmermann & White (1999, *Data-Snooping,
  Technical Trading Rule Performance, and the Bootstrap*, JF) and White (2000, *A Reality
  Check for Data Snooping*, Econometrica) show how easily a universe of chart rules
  manufactures spurious winners; the Gann fan's many angles are a textbook multiple-testing
  hazard, which is why we pre-commit to the single 1x1 spine and a fixed pivot rule.

## Method lineage (the desk's shared engine)

- **HAC one-sample t.** [`strategy.hac_t`](../gann_angles/strategy.py) — Newey-West t of the
  active-minus-benchmark daily spread vs 0 (the inference-bar number).
- **Same-shape random-regime placebo.** [`strategy.placebo_pvalue`](../gann_angles/strategy.py)
  — a matched two-state Markov switch; the "angle vs random timing" null.
- **Deterministic synthetic control.** [`data.synthetic_panel`](../gann_angles/data.py)
  plants a known line effect via alternating up/down legs; at edge 0 the detector must NOT
  manufacture significance, at edge > 0 it must light up — the offline core runs with no
  network.

## Data sources used here

- **yfinance** daily auto-adjusted (total-return) OHLC for `SPY`, `^DJI`, `AAPL`, `GLD`,
  2000-01-03 → 2025-12-30 (GLD from 2004-11-18), cached under `_cache/bars_*_1d.parquet`.
  All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies

- [`../444-dow-theory`](../444-dow-theory) — the Industrials/Transports confirmation rule:
  another mechanical encoding of a charting *theory*, raced the same way (HAC t on the
  active-vs-benchmark spread).
- [`../104-bollinger-reversion`](../104-bollinger-reversion) — band-touch reversion: a
  fixed-rule technical signal pinned against random-entry and breakout controls.
- [`../178-cci`](../178-cci) — a classic technical oscillator under the same protocol.
- The **research-method demos** ([`../348-curve-fitting`](../348-curve-fitting),
  [`../346-multiple-testing`](../346-multiple-testing) and siblings) frame why the Gann fan's
  many angles + free pivots are a snooping trap — and why we pre-commit to one falsifiable
  line.
