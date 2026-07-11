# References & literature map — Study 704 (Three Drives)

## The claim under test

- **The pattern.** The **Three Drives** pattern, part of the Larry Pesavento harmonic-trading
  lineage (Pesavento & Jouflas, *Trade What You See*, 2007) and taught alongside Elliott Wave and
  Gartley material: five labelled swing pivots — points 1 through 5, with an implicit start
  "point 0" before the first push — forming **three drives** (0→1, 2→3, 4→5) to a new high or
  low, separated by **two corrections** (1→2, 3→4). The believers' rule: each correction retraces
  roughly **61.8%** of the drive before it, and each drive extends the prior correction by roughly
  **1.27×** — "three symmetric, Fibonacci-spaced pushes" that exhaust the trend, so price is
  supposed to reverse hard the moment drive 3 completes at point 5.
- **No single canonical source.** Like Wolfe Waves (sibling 697), Three Drives circulates through
  trading-education material rather than one peer-reviewed text, and sources disagree on the
  precise bands (correction depth is variously cited as 0.618, or the wider 0.382–0.886
  Fibonacci retracement zone; extension as a tight ~1.27, or a wider 1.13–2.618 band). We encode
  the **widest mechanical version a proponent would accept** — the full retracement zone for the
  corrections, the full extension range for the drives — and say so plainly in
  `three_drives/strategy.py`.
- **The academic-adjacent anchor.** The pattern's grammar is the same Fibonacci retracement/
  extension vocabulary Elliott (1938) codified for impulse/corrective waves and Gartley (1935)
  codified for XABCD harmonics — see siblings 445-elliott-wave and 468-gartley-harmonic. Three
  Drives has no equivalent peer-reviewed anchor of its own; it is a derivative folk pattern built
  entirely from the same Fibonacci grid applied three times in a row.

## What we measure, and the honesty rails

- **Algorithmic 6-point detection.** A percentage **ZigZag** swing filter (4% headline reversal;
  3%/5%/8% swept for robustness) marks the pivots — the same swing-marking idiom as siblings
  445-elliott-wave / 697-wolfe-waves, so a candidate pattern is never a hand-drawn chart, it is a
  mechanical rule applied identically everywhere. A window of 6 consecutive alternating pivots
  (point 0 through point 5) is a candidate iff its four leg ratios land on the Fibonacci grid
  **and** each drive extends beyond the one before it (point 3 past point 1, point 5 past point
  3 — the "extending drives" signature every source draws).
- **No price target — the claim is pure reversal.** Unlike Wolfe Waves' EPA line or Gartley's
  D-point-plus-AB=CD symmetry, Three Drives makes no specific price-target claim; we test the one
  thing it does claim — fade the reversal at point 5, entered at the **next bar's close** (one
  documented execution lag) — against fixed 5/10/20/40-day horizons.
- **The base rate is a coin flip, not a random-day placebo.** Because there is no target/stop
  geometry to replay at a matched distance (the idiom siblings 697-wolfe-waves and
  448-point-and-figure use), the honest null here is simpler and just as sharp: a **random time,
  random direction** entry, matched in count per ticker — "does knowing three Fibonacci drives
  just finished beat blind timing and a coin-flip direction?"
- **Fibonacci-grid placebo.** Swap the specific 0.382–0.886 / 1.13–2.618 bands for a **random**
  ratio grid, keeping the entire ZigZag machinery and the "extending drives" ordering rule (the same
  idiom sibling 468-gartley-harmonic uses for its XABCD grid). If the *particular* Fibonacci
  numbers are what marks a genuine reversal, the real grid should beat almost every random grid's
  mean fade return.
- **Time-symmetry myth-check.** The folklore's own word is "symmetric" — we score every detection
  by how evenly spaced in time its three drives are (the coefficient of variation of the three
  drive-leg bar-counts) and Welch-*t* the fade returns of the more- vs less-symmetric half. This
  is the direct, falsifiable version of "symmetric" as a *predictive* claim, not just a geometric
  filter the detector already enforces on price.
- **Synthetic positive control.** A deterministic panel builds the three-drives geometry from
  exact anchor points (not accumulated per-bar noise, which would fragment a leg into spurious
  extra pivots) whose ratios sit *exactly* on the Fibonacci grid, with a **tunable planted
  reversal** after point 5. `edge = 0` must not manufacture significance (the geometry is real,
  the follow-through isn't); a real planted reversal must light up unmistakably — see
  `three_drives/data.py::synthetic_panel`.

## Data sources

- **Daily OHLC**, yfinance (no key), auto-adjusted (total-return), cached under `_cache/` — SPY,
  QQQ, DIA, IWM, ^GSPC, ^IXIC, ^DJI, GLD (the same broad-index/ETF basket family as siblings
  445-elliott-wave / 697-wolfe-waves — deep daily history, no exotic single names). As-of
  **2026-06-30** (the last complete calendar month at publication).
- All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Shared method (this desk)

- **Newey, W. K. & West, K. D. (1987).** "A simple, positive semi-definite, heteroskedasticity
  and autocorrelation consistent covariance matrix." *Econometrica* 55(3). The HAC *t* on every
  fade-return sample.
- **Wilson, E. B. (1927).** "Probable inference, the law of succession, and statistical
  inference." *JASA* 22(158). The interval on every win-rate.
- **Sullivan, R., Timmermann, A. & White, H. (1999).** "Data-snooping, technical trading rule
  performance, and the bootstrap." *Journal of Finance* 54(5). Why one positive point estimate
  from one of many parameter settings (ZigZag threshold, Fibonacci tolerance) is not evidence on
  its own — the motivation for the threshold sweep and the ratio-grid placebo.
- House protocol & inference bar: [`../../METHODOLOGY.md`](../../METHODOLOGY.md). The **t ≥ 2 on
  the real tape** rule for a `REAL` stamp; a synthetic control is a machinery proof, never market
  evidence.

## Related desk studies — the dedup map (what this study is NOT)

- [445-elliott-wave](../445-elliott-wave/) — a **different** five/three-wave count (1-2-3-4-5
  impulse then A-B-C correction), Fibonacci **retracement** rules on a single impulse-correction
  cycle, no reversal-at-the-end-of-three-pushes claim. Shares the ZigZag pivot vocabulary; the
  pattern shape and the claim are different.
- [697-wolfe-waves](../697-wolfe-waves/) — the closest cousin in *shape*: another algorithmically
  detected five-point structure on the same ZigZag machinery and basket. But Wolfe Waves is a
  **converging wedge** whose point-5 "throw-over" projects a specific **price-and-time EPA
  target** (a target-hit-vs-random-day-placebo test); Three Drives has **no target line at all**
  — it is three Fibonacci-*proportioned* pushes (not a converging channel) and the falsifiable
  claim is a plain reversal, tested against a coin-flip base rate, not a target-hit rate.
- [415-triple-top-bottom](../415-triple-top-bottom/) — a **price-level** pattern (three touches of
  roughly the *same* high/low, a breakout-of-the-neckline test), with no Fibonacci-ratio
  requirement at all and no requirement that the touches ascend/descend. Three Drives requires the
  opposite geometry — each drive **beyond** the last — and Fibonacci-exact spacing between them.
- [468-gartley-harmonic](../468-gartley-harmonic/) — the closest cousin in **method**: the same
  Fibonacci-retracement-grid-plus-random-grid-placebo idiom, applied to a **four-leg XABCD**
  (one drive, one correction, one drive, one reversal point D) rather than three full
  drive/correction cycles. Directly comparable ratio-grid-placebo results, different pattern
  shape and one more full drive/correction cycle.

None of the siblings test the specific claim this study does: an algorithmically-detected
**three-drive, Fibonacci-proportioned, extending-drives** structure whose point-5 completion is
supposed to mark a plain trend-exhaustion reversal.
