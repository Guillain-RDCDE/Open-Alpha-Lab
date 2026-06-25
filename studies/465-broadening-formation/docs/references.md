# References & literature map — Study 465 (Broadening Formation / megaphone top)

## The claim under test

- **The folklore.** A *broadening formation* (a.k.a. **megaphone**, broadening top, expanding
  triangle, "five-point reversal") is the pattern of **diverging** swing highs and swing lows:
  the highs make higher highs, the lows make lower lows, and the range fans out like a megaphone.
  The lore is that this expanding, increasingly volatile range marks an **over-excited,
  exhausted top** in which "smart money distributes to the crowd," so a break of the **lower
  boundary** signals a reversal **down** — short it. Restated on Investopedia, StockCharts'
  ChartSchool, ThePatternSite (Bulkowski), and every chart-pattern course.
- **The source.** The pattern enters the canon with **Richard W. Schabacker**, *Technical
  Analysis and Stock Market Profits* (1932), and is codified by **Robert D. Edwards & John
  Magee**, *Technical Analysis of Stock Trends* (1948) — the founding text of chart-pattern
  technical analysis — under "Broadening Formations" (orthodox broadening top, right-angled and
  ascending/descending variants). **Thomas Bulkowski**'s *Encyclopedia of Chart Patterns* (2000,
  2005) catalogues hit-rates and "broadening top/bottom" break statistics that the retail world
  cites as evidence.
- **Variants.** Right-angled broadening (one flat boundary), ascending/descending broadening
  wedges, and the "broadening bottom" are the same diverging-pivot geometry with sign/anchor
  tweaks; all inherit the drift confound and the small-sample fragility tested here.

## Why this is a "theory" / mechanical-proxy study

The broadening formation is *semi-subjective*: a discretionary chartist chooses which swings are
"the" megaphone boundaries. Following the desk's design for this kind, we encode the **tightest
mechanical rule a proponent would accept** and state the irreducible subjectivity explicitly:

- **Objective pivots.** Confirmed **fractals** (Bill Williams' fractal definition: a local
  extremum with *k* strictly-lower/higher bars on each side), only usable *k* bars later — a
  documented confirmation lag, no look-ahead.
- **Objective megaphone.** The last two confirmed swing highs must be *rising* and the last two
  confirmed swing lows *falling* (the two boundary lines diverge); no hand-picking of points.
- **The honest baseline.** The only meaningful comparison for a short on an upward-drifting index
  is the **random-entry short** control (same instrument, epoch and hold), because *any* short
  inherits the (negative) drift. We add a **shuffled-pivot placebo** that destroys the diverging
  geometry while keeping the price marginal — the direct test of "does the megaphone matter?"

Hand-drawn megaphones add *hindsight* (which swings to connect is a free parameter), which can
only inflate the in-sample story; the mechanical version is the charitable **upper bound**.

## Why the (negative) one-sample t is not evidence

- **Drift / beta.** US equity indices have a positive unconditional daily mean; a one-sample
  *t* of a **short** rule against **zero** measures that drift with a minus sign, not the rule.
  See Fama & French on the equity premium; the desk's standing rule is *signal-vs-baseline*,
  never *signal-vs-zero*. A short that "loses" need not contain any reversal information — it is
  just short the drift.
- **Data snooping on chart tools.** **Lo, Mamaysky & Wang (2000)**, *Foundations of Technical
  Analysis* (Journal of Finance), formalize testing chart patterns against a properly matched
  null and find most patterns add little once the benchmark is fair. **Sullivan, Timmermann &
  White (1999)**, *Data-Snooping, Technical Trading Rule Performance, and the Bootstrap* (JF),
  and **White (2000)**, *A Reality Check for Data Snooping* (Econometrica), show how
  visually-fitted rules manufacture significance unless raced against a fair benchmark — exactly
  the small-sample megaphone trap here (25 events, easy to over-read).
- **HAC inference.** Newey & West (1987) standard errors for the one-sample mean; Welch (1947)
  two-sample *t* for the short-minus-random difference.

## Method lineage (the desk's shared engine)

- **Confirmed-fractal pivots + rolling megaphone.** [`strategy.find_pivots`](../broadening_formation/strategy.py),
  [`strategy.build_megaphones`](../broadening_formation/strategy.py) — the mechanical geometry
  with the confirmation lag baked in.
- **Forward-return (short) + HAC t + random baseline.** [`strategy.forward_returns`](../broadening_formation/strategy.py),
  [`strategy.hac_t`](../broadening_formation/strategy.py), [`strategy.run_experiment`](../broadening_formation/strategy.py).
- **Geometry placebo.** [`strategy.shuffled_pivot_placebo`](../broadening_formation/strategy.py) —
  permute pivot prices, keep positions/kinds and marginals.
- **Deterministic synthetic control.** [`data.synthetic_panel`](../broadening_formation/data.py)
  plants a real expanding-range reversal (knob `edge`); with `edge = 0` the detector must NOT
  manufacture significance — the offline core runs with no network.

## Data sources used here

- **yfinance** daily adjusted (total-return) closes for SPY, QQQ, IWM, DIA, GLD,
  2005-01-03 → 2026-05-29 (As-of 2026-05-31, partial June dropped), cached as parquet under
  `_cache/`. All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced
  by [`examples/verify.py`](../examples/verify.py).

## Related desk studies

- [`../../450-andrews-pitchfork`](../../450-andrews-pitchfork) — the sibling "price respects the
  drawn channel" study; same confirmed-fractal engine, same random-entry idiom, same None×Mirage.
- [`../../104-bollinger-reversion`](../../104-bollinger-reversion) — "the band reverts price"
  folklore; the volatility-expansion cousin of the megaphone, tested the same way.
- [`../../178-cci`](../../178-cci) and the broader technical-pattern zoo — most land
  None × Mirage because an indicator fitted to past price re-describes the trend.
- The **research-method demos** (data-mining-roulette, look-ahead, curve-fitting,
  multiple-testing) frame why a handful of textbook megaphones and a signal-vs-zero *t* are not
  evidence; the broadening top is a clean live example of a rare, visually-seductive non-signal.
