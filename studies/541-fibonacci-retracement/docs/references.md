# References & literature map — Study 541 (Fibonacci-Retracement)

## The claim, at full strength

- **Fibonacci retracement** (technical-analysis folklore). The claim that after a price swing the
  pullback tends to **stop and reverse** at a retracement of a *Fibonacci ratio* of that swing —
  most famously **38.2%** and **61.8%** ("the golden ratio", the limit of consecutive Fibonacci
  ratios), plus the common **23.6%** and the (non-Fibonacci but conventional) **50%**. Popularised
  by charting texts and terminal defaults.
- **Frost, A. J. & Prechter, R. (1978/2005)**, *Elliott Wave Principle.* The canonical modern
  source tying Fibonacci ratios to swing structure (wave 2 retraces ~50–61.8% of wave 1, etc.) —
  the same ratios this study tests as pure retracement levels.
- **Murphy, J. (1999)**, *Technical Analysis of the Financial Markets.* The standard practitioner
  reference that codifies drawing Fibonacci retracements from a swing high to a swing low and
  treating 38.2 / 50 / 61.8% as support/resistance.
- **Bulkowski, T. (2005)**, *Encyclopedia of Chart Patterns.* Catalogues pattern/level "hit rates";
  the empirical spirit (does the level *work*?) this study applies to Fibonacci retracements.

## The skeptic's case (why a placebo is the right test)

- **Fama, E. (1970)**, *"Efficient Capital Markets."* The weak-form efficiency benchmark: past
  price alone should not forecast returns, so a level derived purely from past swing geometry
  should carry no edge — the null this study cannot reject.
- **Lo, A., Mamaysky, H. & Wang, J. (2000)**, *"Foundations of Technical Analysis."* A rigorous,
  automated evaluation of chart-based rules — the methodological ancestor of testing a charting
  claim mechanically against a null rather than by eyeballing charts.
- **Roberts, H. (1959)** / **Malkiel (1973), *A Random Walk Down Wall Street*.** The random-walk
  case that chart levels are patterns the eye imposes on noise — motivating the **placebo of
  arbitrary fractions**: if 38.2% is special, it must beat 44%.

## The method we build

- **ZigZag swing detector.** The standard tool for marking the "prior swing" a Fibonacci
  retracement is drawn on: confirm a pivot only after price reverses by a threshold, so the pivot
  is look-ahead-free at its confirmation bar.
- **The head-to-head design.** The decisive test is not "does the Fib arm earn?" alone but "does it
  beat a **placebo** of non-Fibonacci fractions *interleaved in the same depth band*?" — matching
  the depth range so the only difference is the exact fractions (Fibonacci vs arbitrary).
- **Welch (1947)** — the unequal-variance two-sample *t* for the Fib-minus-placebo edge.
- **Newey & West (1987)** — the HAC standard error for the pooled reversal returns.
- **Label-shuffle / coin placebo** (Fisher 1935; Good 2005) — same swing bars, random direction:
  does the reversal bet beat a coin placed at the identical swings?

## Neighbours on this bench (the dedup map)

- **[Study 445 — Elliott Wave](../../445-elliott-wave/)** — tests the *wave-count* prediction
  ("buy wave 3" after a Fibonacci-validated impulse 1-2). Study 541 strips out the wave labelling
  entirely and tests the **retracement level itself** as support/resistance, with an explicit
  arbitrary-fraction placebo — the level, not the wave.
- **[Study 440 — Pivot Points](../../440-pivot-points/)** / **[441 Camarilla](../../441-camarilla-pivots/)**
  / **[497 Woodie](../../497-woodie-pivots/)** — other computed intraday support/resistance levels.
  Study 541 is the *Fibonacci-of-the-prior-swing* level, and its distinctive move is the matched
  placebo of non-Fibonacci fractions.
- **[Study 93 — Round Numbers](../../93-round-numbers/)** — whether price clusters/reverses at
  round price levels. Same "is this level special?" question; Study 541 asks it of Fibonacci
  *retracement fractions* of a swing rather than round *price* levels, and answers with a
  fraction-placebo.
- **[Study 77 — Golden Mean](../../77-golden-mean/)** / **[203 Golden Butterfly](../../203-golden-butterfly/)**
  — other φ / "golden" folklore. Distinct constructs; 541 is the retracement-level reversal claim.

## Shared method

- House methodology: [`METHODOLOGY.md`](../../../METHODOLOGY.md) — the inference bar (a robust
  *t* ≥ 2 on the real tape plus a placebo null and seed-robustness), one documented execution lag,
  gross and net labelled, costs one-way × NAV, and the price-index survivorship note.
