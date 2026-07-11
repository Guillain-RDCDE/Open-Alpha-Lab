# References & literature map — Study 676 (Gator Oscillator)

## The claim under test

- **The folklore.** Bill Williams' **Gator Oscillator** (*Trading Chaos*, 1995; refined
  in *New Trading Dimensions*, 1998) plots two histograms — the absolute spreads
  between the Alligator's Jaw/Teeth and Teeth/Lips lines — colored green when the
  spread widens vs the prior bar and red when it narrows. The reading: red bars on both
  sides mean the alligator's mouth is shut and it is "sleeping" (no trend, stay out);
  the moment both bars flip green it is "waking up hungry," and the believer enters in
  the direction the three lines are already fanned, riding the trend the Gator claims
  it is about to "eat."
- **What the Gator actually is, mechanically.** It is not a second, independent
  indicator — it is the **rate of change of the Alligator's own spread**, a derived
  transform of the same three SMMAs. Any information it carries is already fully
  contained in the Alligator lines it is built from; the honest question this study
  narrows to is whether *watching the color change* adds a genuine, tradable timing
  edge over simply knowing the fan direction — not whether trends exist (they do; see
  421's teardown) but whether the Gator's specific "wake" signal times entry into them
  better than just being in the fan.
- **No peer-reviewed anchor.** Unlike the FOMC vol crush (637) or momentum (multiple
  studies on this desk), the Gator Oscillator has no academic literature behind it — it
  is retail-platform folklore (MetaTrader/MT4 default indicator, Williams' own trading
  books) with essentially zero third-party statistical scrutiny. The claim is tested
  here on its own terms, at full strength, exactly as stated by MT4 documentation and
  Williams' own books.

## What we measure, and the honesty rails

- **Same fan, same lines as sibling 421.** Jaw/Teeth/Lips = SMMA(13/8/5) of the median
  price `(H+L)/2`, forward-shifted 8/5/3 bars — identical construction to
  [421-williams-alligator](../421-williams-alligator/), so any difference in verdict
  between the two studies is attributable to the Gator's specific "wake" signal, not to
  a different Alligator.
- **A genuine "sleep" requirement, not a hair-trigger.** The naive "both bars flip
  red→green on consecutive bars" definition fires on ~24% of trading days (any local
  minimum of two correlated, noisy series) — nothing like the folklore's "the market
  consolidated, then woke up." We require **≥ 3 consecutive both-red bars** immediately
  before the flip (`MIN_SLEEP = 3`, fixed once and reused in the event study, the
  timer, and the synthetic control — never re-tuned to chase a result).
- **Directional sign comes from the concurrent Alligator fan**, exactly as a believer
  would trade it (long on a bullish fan, short on a bearish one); wakes that fire while
  the three lines aren't cleanly ordered are excluded from the *signed* test (154 of
  346) but retained in the magnitude ("trend-capture") test, which asks a direction-free
  question.
- **One documented execution lag.** The wake/fan state is known at the close of bar
  *t*; the event study enters the next bar's **open** (t+1); the timer's position
  signal is shifted once and acts on the return of t+1 onward.
- **HAC (Newey-West) *t*** on the signed forward return (autocorrelation-robust; wake
  events on the same name can be temporally close), a **Welch *t*** vs the
  unconditional base rate, and a **5,000-draw label-shuffle placebo** (event study) /
  **2,000-draw circular-block-permutation placebo** (timer).
- **Costs charged one-way × NAV per leg** (5 bps swept to 10), 50 bps/yr borrow on
  short legs, and a flat 4%/yr cash-leg proxy credited to the timer while flat (FRED
  unavailable offline — the same proxy sibling 421 uses) so a mostly-in-cash timer
  isn't unfairly penalised *or* flatteringly compared against a 100%-equity
  buy-and-hold without a matched cash leg.

## Why the "amazing Sharpe" on the timer is flagged, not celebrated

- The wake timer is in the market **0.9%** of the time (13 events on SPY across 26.5
  years). Being mostly in a low-volatility cash proxy mechanically shrinks realized
  volatility and inflates the Sharpe ratio — a well-known artifact, not evidence of
  skill (house rule: "normalise before you marvel"). The **Sharpe-difference HAC *t***
  and a **block-permutation placebo that reshuffles the same sparse exposure pattern**
  are the only honest arbiters here, and both say the advantage is not distinguishable
  from luck (*t* = −1.16 vs buy-and-hold, placebo *p* = 0.512).

## Data sources

- **Daily OHLCV**, 30 names (SPY + 29 liquid US large-caps), yfinance
  (`auto_adjust=True`), cached under `_cache/` as one parquet per ticker,
  2000-01-03 → 2026-06-30.
- **Synthetic positive control**: `gator_oscillator/data.py::synthetic_panel` /
  `synthetic_multi_panel` — a deterministic, seeded multi-week trend-persistence
  generator with a tunable `edge` knob, the same construction family as sibling 421.
- All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).
- Bill Williams, *Trading Chaos: Applying Expert Techniques to Maximize Your Profits*
  (1995) and *New Trading Dimensions* (1998) — the original description of the
  Alligator/Gator/Awesome Oscillator/Accelerator Oscillator family. MetaTrader 4/5
  built-in indicator documentation (the Gator Oscillator ships as a default MT4/5
  indicator) for the canonical color/formula definition used here.

## Related desk studies (the dedup map — what this study is NOT)

This is the **paired study** to the Alligator itself and Williams' other momentum
cousins — all four reuse the same Jaw/Teeth/Lips fan or the same underlying philosophy,
and none of them tests what this study tests: **does the Gator's specific "wake"
color-change signal, layered on top of the fan, add a real timing edge?**

- [421-williams-alligator](../../421-williams-alligator/) — the Alligator itself, run
  as a long/flat and long/short **continuous** timing rule (in the fan whenever it's
  fanned). Verdict there: Weak signal / Mirage tradability, beaten by a plain SMA(200).
  **This study**: not "is the fan tradable" (421 already answered that) but "does
  watching the Gator's histogram color, specifically the wake transition, time entries
  into that fan *better* than just being in it" — and races the two head-to-head
  (Sharpe-diff *t* = +0.65, not significant).
- [184-williams-fractals](../../184-williams-fractals/) — Williams' **swing-high/low**
  pivot marker, a completely different construction (5-bar local extrema), used for
  stop placement in his own system. Not a moving-average spread, not this study's axis.
- [420-awesome-oscillator](../../420-awesome-oscillator/) — Williams' **momentum**
  cousin (SMA(5) − SMA(34) of the *median price*, unrelated to the Jaw/Teeth/Lips fan
  entirely). A momentum oscillator, not a convergence/divergence-of-the-Alligator
  signal.
- [474-accelerator-oscillator](../../474-accelerator-oscillator/) — the **second
  derivative** of the Awesome Oscillator (AO minus its own SMA(5)), Williams' momentum
  acceleration signal. Also unrelated to the Alligator's three lines.

None of the siblings tests the Gator's own "wake" transition as a timing signal on top
of the Alligator fan — that is this study's own, narrow axis.
