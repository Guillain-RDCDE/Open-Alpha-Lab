# References & literature map — Study 410 (Cup & Handle)

## The claim under test

- **The folklore.** A stock that carves a **cup-with-handle** — a rounded, U-shaped base that
  recovers to its prior high (the cup), followed by a short, shallow pullback (the handle) — is
  poised to launch once it **breaks out above the rim ("the pivot")**. Buy the breakout and ride
  the run. It is the single most famous pattern in retail technical analysis.
- **The source.** William J. O'Neil coined and popularised the figure in *How to Make Money in
  Stocks* (McGraw-Hill, 1988, and later editions) as the centrepiece of his **CAN SLIM** system
  and the chartbook culture of *Investor's Business Daily*. O'Neil presented it as a recurring
  precursor to large advances in the great winning stocks.
- **The honest caveat O'Neil's own followers concede.** The cup-with-handle is **discretionary**:
  identifying the "right" cup depth, handle shape, and pivot is a judgement call, which makes it
  notoriously hard to test. That is precisely why this study fixes a transparent **mechanical**
  definition and reports the verdict for *that* rule, stating up front it is not the only one.

## What the evidence says

- **Pattern-recognition tests.** Andrew Lo, Harry Mamaysky & Jiang Wang, *Foundations of
  Technical Analysis* (2000, *Journal of Finance*) built kernel-smoothing detectors for classic
  figures (head-and-shoulders, tops/bottoms) and found *some* of them carry marginal information
  — but the effects are weak, regime-dependent, and shrink out of sample. Their method is the
  intellectual ancestor of any objective chart-figure test, including this one.
- **Survey verdict.** Park & Irwin, *What Do We Know About the Profitability of Technical
  Analysis?* (2007, *Journal of Economic Surveys*) review ~100 studies: early positive results
  largely evaporate once data-snooping, transaction costs, and out-of-sample testing are imposed.
  The cup-with-handle specifically has little independent academic support beyond O'Neil's own
  exposition.
- **Data-snooping discipline.** Ryan Sullivan, Allan Timmermann & Halbert White, *Data-Snooping,
  Technical Trading Rule Performance, and the Bootstrap* (1999, *Journal of Finance*) and White's
  *Reality Check* (2000, *Econometrica*) show that mining many chart rules guarantees some look
  significant by luck. Our **same-tape label-shuffle placebo** is the lightweight analogue: it
  asks whether random entry dates on the same drifting tape do as well (they do).

## The methodological trap this study illustrates

- **A *t* against zero is not enough on a drifting tape.** Any long-only "buy and hold N days"
  rule on an up-drifting stock earns a positive mean with a respectable *t* — because the **base
  rate** is positive. The correct null is the name's *own* buy-and-hold (excess return) plus a
  placebo of random dates, not zero. This study's headline 10-day *t* = 2.14 against zero
  **collapses** under that correction (placebo *p* = 0.34) — the canonical illustration of why the
  desk's inference bar pairs the *t* with a placebo.

## Method lineage (the desk's shared engine)

- **Objective figure detector.** [`strategy.swing_pivots`](../cup_and_handle/strategy.py) +
  [`strategy.detect_cups`](../cup_and_handle/strategy.py) — swing-pivot cup (depth band, rim
  tolerance, U-shape) + shallow handle + first close clearing the rim. No look-ahead in the signal.
- **Excess-over-base-rate + placebo.** [`strategy.run_experiment`](../cup_and_handle/strategy.py)
  subtracts each name's base rate and runs a same-tape random-date placebo — the honest arbiter.
- **HAC *t*.** [`strategy.hac_t`](../cup_and_handle/strategy.py) — Newey-West Bartlett-kernel *t*
  for autocorrelation among clustered breakouts.
- **Deterministic synthetic control.** [`data.synthetic_panel`](../cup_and_handle/data.py) plants
  clean cup-with-handle shapes and a known post-breakout drift (knob `edge`); with `edge = 0` the
  inference must NOT manufacture significance. Runs offline.

## Data sources used here

- **yfinance** daily auto-adjusted (split + dividend) OHLC for a fixed 30-name large-cap basket
  including **SPY**, 2005-01-03 → 2026-05-29 (as-of **2026-05-31**), cached under
  `_cache/cuph_*.parquet`. All headline numbers are pinned in [`docs/results.md`](results.md) and
  reproduced by [`examples/verify.py`](../examples/verify.py).

## Related desk studies

- **[104-bollinger-reversion](../../104-bollinger-reversion/)** and **[178-cci](../../178-cci/)** —
  sister mechanical-TA teardowns on daily bars; same "objective rule + placebo vs base rate" idiom.
- The **research-method demos** (data-mining-roulette, multiple-testing, look-ahead) frame why a
  *t* alone is not enough — Cup & Handle is the live example where a *t* > 2 against zero is
  **busted by the placebo**.
- **[363-pead-drift](../../363-pead-drift/)** — the contrast case: a folk effect that *does* clear
  the placebo and earns Signal = REAL.
