# Study 413 — Bull Flag 🚩

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the breakout beat buy-and-hold? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | Forward return after a confirmed bull-flag breakout is **negative as an excess over each name's own base rate at every horizon** (−0.38% at 20 days, one-sample *t* = **−1.07**, HAC *t* = −1.07, win-rate **below 50%**). The same-tape **placebo p ≈ 0.60** (random dates win ~3-in-5), no strictness setting produces a placebo-clean positive, and survivorship tilts the test *for* the figure and it still loses. The breakout slightly **underperforms** just holding the name. |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | Nothing to deploy: the gross excess is below zero *before* costs (net −0.29% to −0.70%), and the SPY-only version rests on **4** events at *t* ≈ 0. You'd pay to underperform. |
| **"Resolves up specifically"?** | ![Busted](https://img.shields.io/badge/Resolves_up%3F-Busted-8b949e?style=flat-square) | The figure's whole premise is *direction* — and the **down**-break of the identical flag drifts **up MORE** than the textbook up-break (+0.71% vs −0.19% at 5 days; +0.68% vs −0.38% at 20 days). Break up or break down, the stock drifts the same: the post-figure return is a property of the **names**, not the **break direction**. |

> **In one sentence:** the bull flag is the textbook bullish-continuation figure, but on 21 years of SPY + 29 large-caps the confirmed breakout *underperforms* simply holding the name (excess −0.38% at 20 days, *t* = −1.07, placebo *p* ≈ 0.60, win-rate < 50%) — and the "failed" down-break of the same flag drifts up *more*, so the breakout direction carries no information at all.

## What we tested

We rebuild the bull flag with the closest **mechanical** definition we can write down — a steep **flagpole** (≥12% over 12 sessions), a tight flat-to-down **flag** that gives back no more than half the pole in a ≤10% range, and a confirmed close back above the flag's **high** (the breakout) — across **SPY + 29 long-listed US large-caps** on daily auto-adjusted OHLC. For each confirmed breakout we measure the forward **5 / 10 / 20 / 40-day** return as an **excess over each name's own base rate** (does the figure beat buy-and-hold for *that* name?), entering the close one day after the breakout (no look-ahead). The Signal axis tests the excess against zero with a one-sample / HAC *t* and a **same-tape label-shuffle placebo** (random entry dates on the same drifting names); the myth-check runs the **identical** detector requiring a *down*-break instead. Chart figures are partly subjective — we test one transparent definition and say so. A deterministic synthetic control with a *planted* post-breakout drift confirms the engine can bank a real edge (it finds none here).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what a bull flag is (pole + flag + breakout), why "raw return up" isn't the same as "beats holding the name," and the gotcha that the *failed* break does just as well — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the mechanical detector, forward 5/10/20/40-day excess over base rate, a one-sample/HAC *t* + same-tape placebo, the down-break symmetry test, a strictness sweep, SPY-only, and a synthetic faithful-engine / power control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`bull_flag/`](bull_flag/). Detector = steep flagpole + tight flat-to-down flag + confirmed close above the flag high; the down-break of the same flag is the symmetry myth-check. Basket is **survivors** — named on the Signal axis (the bias tilts *for* the figure and it still loses). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
