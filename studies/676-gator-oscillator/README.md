# Study 676 — Gator Oscillator 🐊💤

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does waking predict where price goes? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | 346 wake events (both Gator bars flipping red→green after ≥3 sleeping bars) pooled across a 30-name basket. Signed by the concurrent Alligator fan, the forward return clears **\|*t*\| ≥ 2** at **no** horizon (1/5/10/20 days) — HAC *t* tops out at **+1.35**, Welch *t* vs the unconditional base rate at **−1.88**. The "trend-capture" fallback (bigger moves ahead, any direction) fares no better — the one horizon with any hint of significance (10-day, Welch *t* = **−2.41**) says forward moves are *smaller*, not bigger. |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | A real SPY timer's headline Sharpe (**+2.548** vs buy-and-hold's +0.412) is a cash-dominance artifact from sitting out **99.1%** of the time (13 wake events in 26.5 years). The Sharpe-**difference** HAC *t* vs buy-and-hold is only **−1.16**; a block-permutation placebo shows a random reshuffle of the same sparse exposure matches or beats it **51%** of the time. |
| **Does the gator's awakening catch the start of a trend?** | ![Busted](https://img.shields.io/badge/Catches_trend_starts%3F-Busted-8b949e?style=flat-square) | The synthetic control proves the *machinery* can catch a genuine planted multi-week trend (*t* = **+3.20**); on the real tape the signed wake-event return is flat at every horizon, and watching the histogram's color change adds nothing measurable over just knowing the Alligator's fan is bullish or bearish (*t* = **+0.65**, not significant). |

> **In one sentence:** Bill Williams' Gator Oscillator — literally the rate-of-change of his own Alligator's spread, plotted as a green/red histogram — fires a genuine "wake" event only 346 times across a 30-name basket over 26 years, and none of those wakes predict the forward direction or magnitude of the next move (\|*t*\| < 2 everywhere); a real SPY timer built on it *looks* spectacular on raw Sharpe only because it sits in cash 99% of the time, and that "edge" evaporates the moment you test it against buy-and-hold or against a random reshuffle of the same sparse trades.

## What we tested

We compute the canonical Bill Williams Alligator — Jaw/Teeth/Lips = SMMA(13/8/5) of the
median price, forward-shifted 8/5/3 bars, identical to sibling
[421-williams-alligator](../421-williams-alligator/) — then the Gator's two histograms
(upper = \|Jaw−Teeth\|, lower = \|Teeth−Lips\|, colored green when taller than the prior
bar). A **wake** fires the first bar both histograms flip red→green together, *after* at
least 3 consecutive both-red "sleeping" bars (a fixed, un-tuned threshold — the naive
single-bar-flip definition fires on ~24% of days and is not what the folklore means). We
pool wake events across SPY + a 29-name liquid large-cap basket, sign the forward return
by the concurrent Alligator fan, and test 1/5/10/20-day horizons with a HAC *t*, a
Welch *t* vs the unconditional base rate, and a 5,000-draw label-shuffle placebo — plus
an unsigned "trend-capture" magnitude test. On the third axis, a real SPY timer enters
the fan direction the bar after a wake, holds 10 sessions, and races NET Sharpe (5/10
bps one-way costs, borrow on shorts, a flat cash-leg proxy while flat) against
buy-and-hold **and** against 421's "always in the fan" rule — the decisive comparison
for whether the Gator's color-change timing adds anything to the Alligator it's built
from. A trend-persistence synthetic control (same construction family as 421) confirms
the machinery detects a planted trend and stays silent on a fair-coin null. **Dedup:**
siblings [421-williams-alligator](../421-williams-alligator/) (the Alligator fan itself,
run continuously), [184-williams-fractals](../184-williams-fractals/) (a different
5-bar pivot marker), [420-awesome-oscillator](../420-awesome-oscillator/) and
[474-accelerator-oscillator](../474-accelerator-oscillator/) (Williams' unrelated
momentum cousins) never test the Gator's own wake signal — this study is the paired
teardown. As-of **2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what the green/red bars mean, why "waking" needed a real definition, the wake-event scorecard, and why a spectacular-looking Sharpe turns out to be an illusion — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the wake detector, the pooled HAC/Welch event study across horizons, the magnitude test, the cash-dominance Sharpe artifact unpacked with a block-permutation placebo, the race against 421's Alligator, and the synthetic positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`gator_oscillator/`](gator_oscillator/). Indicator = two histograms built
from the same displaced-SMMA Alligator fan as sibling 421 (Williams' 13/8/5 fan). Basket
is a **survivors panel** (every name still trades in 2026), named on the Signal axis;
cash leg on the timer proxied at 4%/yr flat (FRED unavailable). **Not investment
advice** — research & education. See [LICENSE](../../LICENSE).*
