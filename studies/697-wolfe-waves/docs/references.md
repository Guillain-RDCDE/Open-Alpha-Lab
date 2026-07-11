# References & literature map — Study 697 (Wolfe Waves)

## The claim under test

- **The pattern.** Bill Wolfe's **Wolfe Wave**: a five-point reversal structure (points 1-2-3-4-5,
  alternating swing highs/lows) forming a converging wedge whose final leg (point 5) *overshoots*
  the trendline drawn through points 1 and 3 — the "throw-over" — right before price snaps back.
  The claimed payoff is the **EPA line** ("Estimated Price at Arrival"): extend the trendline
  through points 1 and 4 forward, and that is the price *and* the approximate time the market is
  drawn back to — proponents call it a "natural equilibrium" point the pattern's own geometry
  reveals in advance.
- **No single canonical source.** Unlike Elliott Wave (Elliott 1938) or Gann's angles (Gann's own
  writings), Wolfe Wave rules were popularised through trading-education material and courses
  rather than one peer-reviewed or even self-published canonical text; the pattern circulates on
  retail charting sites, and **every source states the rule set slightly differently** — how
  strictly wave 4 must stay "inside the channel", how the time target is actually projected, even
  whether the wedge should converge or merely run parallel. That disagreement is itself evidence
  of how much discretion the pattern leaves a human chartist. We encode the **one rule every
  source agrees on** (point 5 pierces the 1-3 trendline) plus the **channel/convergence checks**
  most sources add (wave 3 extends the wedge; wave 4 stays inside the channel and makes a lower
  high / higher low than wave 2) — see `wolfe_waves/strategy.py` for the exact, falsifiable
  version we test.

## What we measure, and the honesty rails

- **Algorithmic 5-point detection.** A percentage **ZigZag** swing filter (4% headline reversal;
  3%/5%/8% swept for robustness) marks the pivots — the same swing-marking idiom as sibling
  445-elliott-wave, so a candidate pattern is never a hand-drawn chart, it is a mechanical rule
  applied identically everywhere.
- **The EPA price target** is the line through points 1 and 4, evaluated the instant point 5's
  ZigZag pivot is *confirmed* (a further ≥4% reversal from point 5's own extreme — the
  look-ahead-free confirmation the whole desk uses). Entry is the **next bar's close** (one
  documented execution lag).
- **The target-hit test** walks forward on intraday High/Low and asks which comes first: the EPA
  target, or the invalidation stop (point 5's own extreme, i.e. the wedge failing outright). A bar
  touching both is scored conservatively as a loss. No hit inside 90 bars is a timeout, excluded
  from the hit-rate and reported separately — the same "hit target before stop" idiom sibling
  448-point-and-figure uses for its horizontal-count target.
- **The base rate.** A **same-distance, same-direction random-day placebo** on the identical tape
  (matched target/stop distances from every real trade, replayed from random days excluding a
  ±10-bar window around every real signal) — the honest "would a random day, walked the same
  distance, have looked this good?" null. `p` = share of placebo draws whose hit-rate is ≥ the
  observed one.
- **The EPA time target** — most sources describe a "time axis" projection but disagree on the
  mechanics. We fix **one** ex-ante convention (point 5's bar + the point-1-to-point-4 bar span)
  and test whether it predicts anything about *when* a winning trade actually resolves — a
  correlation and a mean absolute error against the actual bars-to-hit.
- **Secondary, geometry-free cut.** A fixed-horizon (5/10/20/40-day) directional forward return in
  the pattern's predicted direction, one-sample + Newey-West (1987) HAC *t*, plus a same-bars
  random-direction coin placebo — doesn't depend on the target/stop levels at all, so it can't be
  an artefact of how we picked those.
- **Synthetic positive control.** A deterministic panel builds the wedge geometry from exact,
  fixed anchor points (not accumulated per-bar noise, which would fragment a leg into spurious
  extra pivots) with a **tunable planted reversal** after point 5. `edge = 0` must not
  manufacture significance; a real planted reversal must light up unmistakably — see
  `wolfe_waves/data.py::synthetic_panel`.

## Data sources

- **Daily OHLC**, yfinance (no key), auto-adjusted (total-return), cached under `_cache/` —
  SPY, QQQ, DIA, IWM, ^GSPC, ^IXIC, ^DJI, GLD (the same broad-index/ETF basket family as sibling
  445-elliott-wave — deep daily history, no exotic single names). As-of **2026-06-30** (the last
  complete calendar month at publication).
- All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Shared method (this desk)

- **Newey, W. K. & West, K. D. (1987).** "A simple, positive semi-definite, heteroskedasticity
  and autocorrelation consistent covariance matrix." *Econometrica* 55(3). The HAC *t* on the
  secondary fixed-horizon cut.
- **Wilson, E. B. (1927).** "Probable inference, the law of succession, and statistical
  inference." *JASA* 22(158). The interval on every hit-rate.
- **Sullivan, R., Timmermann, A. & White, H. (1999).** "Data-snooping, technical trading rule
  performance, and the bootstrap." *Journal of Finance* 54(5). Why a positive point estimate
  from one of many parameter settings (ZigZag threshold, target/stop convention) is not evidence
  on its own — the motivation for the ZigZag-threshold sweep and the random-day placebo.
- House protocol & inference bar: [`../../METHODOLOGY.md`](../../METHODOLOGY.md). The **t ≥ 2 on
  the real tape** rule for a `REAL` stamp; a synthetic control is a machinery proof, never market
  evidence.

## Related desk studies — the dedup map (what this study is NOT)

- [445-elliott-wave](../445-elliott-wave/) — a **different** five/three-wave count (1-2-3-4-5
  impulse then A-B-C correction) with **Fibonacci retracement** rules and no target-price
  geometry; tests whether wave 3 extends after a wave-2 pullback. Shares the ZigZag pivot
  vocabulary; the pattern shape, the entry rule and the target claim are all different.
- [447-gann-angles](../447-gann-angles/) — a single fixed-slope **trend line** (the 1x1 angle)
  re-anchored at swing lows, tested as a long/flat trend-following filter. No 5-point structure,
  no price+time target, no wedge.
- [448-point-and-figure](../448-point-and-figure/) — the closest cousin in **method**: it also
  tests whether a chart pattern's price target gets hit before a stop, against a same-distance
  random-day placebo. But the pattern is a box-and-reversal **count** (double top/bottom breakout
  width × 3), not a 5-point wedge, and it found the count target genuinely gets hit more than
  chance (`Signal: Real`) even though the money is one-directional. Wolfe Waves' target-hit rate,
  by contrast, matches the random baseline almost exactly — the two studies land on opposite
  Signal verdicts using the same target-hit idiom, which is itself informative about which chart
  claims survive an honest test and which don't.
- [704-three-drives](../704-three-drives/) — another Wolfe-adjacent five-point Fibonacci
  structure (the "Three Drives" pattern: three symmetric price/time-ratio drives, not a
  converging wedge with a throw-over). No EPA line, no channel-containment rule; the falsifiable
  claim is Fibonacci time/price symmetry across the three drives, not a target-hit test.

None of the siblings test the specific claim this study does: an algorithmically-detected 5-point
**converging wedge** whose point-5 throw-over projects a **price-and-time EPA target** via the
1-4 trendline.
