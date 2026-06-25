# References & literature map — Study 461 (Descending Triangle)

## The claim under test

- **The folklore.** A *descending triangle* is a **flat horizontal support** (a run of swing
  lows at roughly the same price) underneath a **falling resistance** (a run of descending swing
  highs), the two converging toward an apex. It is taught as a **bearish continuation**: price
  coils, then **breaks down through the flat support and keeps falling**. The textbook trade is to
  **short the support break**, with a measured-move target the height of the triangle. This is a
  retail/technician staple built into TradingView, MetaTrader, Thinkorswim and StockCharts.
- **The source.** The triangle figures are codified in **Robert D. Edwards & John Magee,
  *Technical Analysis of Stock Trends*** (1948, the foundational chart-pattern text), which names
  the ascending/descending/symmetrical triangles and assigns the descending triangle a bearish
  bias. **Thomas N. Bulkowski**'s *Encyclopedia of Chart Patterns* (2000/2005) is the modern
  empirical catalogue, reporting break-out direction frequencies and "performance" by pattern;
  **John J. Murphy**, *Technical Analysis of the Financial Markets*, and StockCharts' ChartSchool
  restate the rule for a general audience.
- **The distinguishing geometry.** What makes a *descending* triangle (vs a plain horizontal
  support) is the **descending highs** — the falling ceiling that "squeezes" price into the
  support. Our placebo deliberately destroys exactly that constraint to ask whether it is
  load-bearing or whether the result is just a generic support break.

## Why this is a "theory" / mechanical-proxy study

The descending triangle is *semi-subjective*: a discretionary chartist chooses which swings count
as the flat lows and the descending highs, and what counts as "close enough" to horizontal.
Following the desk's design for this kind, we encode the **tightest mechanical rule a proponent
would accept** and state the irreducible subjectivity explicitly:

- **Objective pivots.** Confirmed **fractals** (Bill Williams' fractal definition: a local
  extremum with *k* strictly-lower/higher bars on each side), only usable *k* bars later — a
  documented confirmation lag, no look-ahead.
- **Objective triangle.** Over a rolling window of the most-recent confirmed pivots, require the
  swing **highs to descend** (last well below first, no higher high) and the swing **lows to be
  flat** (spread within a tolerance band) — no hand-picking which dots to connect.
- **The honest baseline.** The only meaningful comparison on a drifting tape is the
  **random-entry** control (same instrument, epoch and hold, also booked short), because *any*
  short on an upward-drifting index inherits a drift *headwind*. We add a **scrambled-highs
  placebo** that shuffles the swing-high prices — destroying the "descending highs" geometry while
  keeping the marginal — to test "is the triangle's defining shape doing anything beyond a plain
  support break?"

Hand-drawn triangles add *hindsight* (free parameters in the tolerance and which swings to use),
which can only inflate in-sample fit; the mechanical version is the charitable **upper bound**.

## Why a high signal-vs-zero t is not evidence

- **Drift / beta.** US equity indices have a positive unconditional daily mean; a SHORT inherits a
  *negative* drift, so a short-the-break rule's one-sample *t* against **zero** measures the tide
  (here a headwind), not the pattern. The desk's standing rule is *signal-vs-baseline*, never
  *signal-vs-zero*. See Fama & French on the equity premium.
- **Data snooping on chart tools.** Lo, Mamaysky & Wang (2000, *Foundations of Technical
  Analysis*, *Journal of Finance*) formalize testing chart patterns against a properly matched
  null and find most add little conditional information; Sullivan, Timmermann & White (1999,
  *Data-Snooping, Technical Trading Rule Performance, and the Bootstrap*, *JF*) and White (2000,
  *A Reality Check for Data Snooping*, *Econometrica*) show how pattern-fitted rules manufacture
  significance unless raced against a fair benchmark.
- **HAC inference.** Newey & West (1987) standard errors for the one-sample mean; Welch (1947)
  two-sample *t* for the break-vs-random difference.

## Method lineage (the desk's shared engine)

- **Confirmed-fractal pivots + rolling triangle test.**
  [`strategy.find_pivots`](../descending_triangle/strategy.py),
  [`strategy.build_support`](../descending_triangle/strategy.py) — the mechanical geometry with the
  confirmation lag baked in.
- **Forward-return (short P&L) + HAC t + random baseline.**
  [`strategy.forward_returns`](../descending_triangle/strategy.py),
  [`strategy.hac_t`](../descending_triangle/strategy.py),
  [`strategy.run_experiment`](../descending_triangle/strategy.py).
- **Geometry placebo.** [`strategy.scrambled_highs_placebo`](../descending_triangle/strategy.py) —
  permute swing-high prices, keep positions/support/marginal.
- **Deterministic synthetic control.**
  [`data.synthetic_panel`](../descending_triangle/data.py) plants a real break-down continuation
  (knob `edge`); with `edge = 0` the detector must NOT manufacture significance — the offline core
  runs with no network.

## Data sources used here

- **yfinance** daily adjusted (total-return) closes for SPY, QQQ, IWM, DIA, GLD,
  2005 → 2026, cached as parquet under `_cache/`. All headline numbers are pinned in
  [`docs/results.md`](results.md) and reproduced by [`examples/verify.py`](../examples/verify.py).

## Related desk studies

- [`../450-andrews-pitchfork`](../450-andrews-pitchfork) — the sibling "price respects the
  geometry" chart-tool teardown using the same confirmed-fractal + random-entry idiom.
- [`../410-head-and-shoulders`](../410-head-and-shoulders) and the other chart-figure studies —
  the same Edwards & Magee pattern family, mechanically encoded.
- The broader technical-indicator zoo (e.g. [`../178-cci`](../178-cci)) — most land None × Mirage
  because an indicator fitted to past price re-describes the trend rather than forecasting it.
- The **research-method demos** (data-mining-roulette, look-ahead, curve-fitting) frame why a
  signal-vs-zero *t* is not enough; the descending triangle is a clean live example of a pattern
  whose distinguishing geometry can be scrambled with no loss.
