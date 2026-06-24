# Study 406 — Harami Pattern 🕯️

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the inside bar flip the trend? | ![Mixed](https://img.shields.io/badge/Signal-Mixed-dab617?style=flat-square) | Pooled, the signed return (long bullish / short bearish) clears the bar at multi-day horizons — **+0.10% at 5 days, HAC *t* = 3.39** — but it is **one-legged**: the bullish leg is strong (*t* = +10.2 at 10d) while the bearish leg is **wrong-signed** (*t* = −5.8). Real on the long side, busted on the short. |
| **Tradability** — could you trade it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | Net of a 5 bps round trip + short borrow, the two-sided rule is **≈ 0** (net −0.006% at 5 days). The thin gross edge does not survive realistic costs. |
| **A reliable reversal?** | ![Misattributed](https://img.shields.io/badge/Reliable_reversal%3F-Misattributed-8b949e?style=flat-square) | The bullish leg's *excess over the unconditional drift* clears *t* = 2 only at **1 day** (Welch *t* = 2.74), fading to noise by 10 days — the gain is **beta/drift, not a trend flip** — and the textbook trend/volume filters don't help. |

> **In one sentence:** the harami inside-bar "reversal" is mostly beta in a costume on liquid US large-caps — pooled it clears HAC *t* = 3.39 at 5 days, but that is the *bullish* leg riding a 21-year bull market (its edge over buy-and-hold clears *t* = 2 only at 1 day) while the *bearish* leg points the **wrong way** (shorting the "top" loses, *t* = −5.8); net of costs the two-sided rule is ≈ 0, and the harness's synthetic control proves it *would* see a real reversal if one existed.

## What we tested

We detect every **harami** candle — the precise real-body rule: opposite colours, the current
body strictly *smaller* and sitting fully *inside* the prior one (the geometric inverse of the
engulfing pattern) — across a fixed **30-name** basket (29 long-listed liquid US large-caps +
**SPY**, yfinance daily 2005→2026, cache-first). Confirming at the close, we **enter the next
open** (one execution lag) and measure the forward **1 / 3 / 5 / 10-day** return *signed by the
pattern direction* (long after bullish, short after bearish) against the unconditional drift,
with a Newey-West **HAC t**, a coin-signed label-shuffle placebo, a **leg split**, a Welch test
of the bullish leg vs the drift, realistic costs, and a trend/volume **myth-check**. A
deterministic synthetic control with a *planted* day-after reversal confirms the engine can bank
a real edge and that a random walk can't fake one. Survivorship (the basket is names still
trading in 2026) is named on the Signal axis — it works *for* the long leg, yet the pattern
still fails as a two-legged reversal.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what a harami inside bar is, why the bullish leg "works" but the bearish leg points the wrong way, and why most of the gain is just the market drifting up — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the real-body inside-bar detector, forward 1/3/5/10-day signed returns, HAC *t* vs zero + a coin-sign placebo, the leg split, the bullish-leg-vs-drift Welch test, costs, the trend/volume myth-check, and a synthetic faithful-engine / power control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`harami_pattern/`](harami_pattern/). Bars are total-return adjusted (`auto_adjust=True`); the signed return is long after bullish / short after bearish. Basket is **survivors** — named on the Signal axis. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
