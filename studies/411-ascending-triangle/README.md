# Study 411 — Ascending Triangle 📐

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the breakout predict an up-move? | ![Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square) | Across **385** confirmed up-breaks (SPY + 29 large-caps, 21.4 yrs), the forward return *in excess of each name's base rate* clears a naive bar (**+1.47% at 20d, one-sample t = 3.81, HAC t = 4.05**) — but the **same-tape placebo p = 0.21** (random dates win ~1-in-5), strictness keeps *p* ~0.16–0.32, and a **zero-edge synthetic control reproduces t = 4.51 at placebo p = 0.12**. The number is real; its attribution to the triangle is not. |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | What you'd trade is the names' own up-drift / momentum into resistance — random dates keep pace. The SPY-only version rests on **11** events at *t* ≈ 1; costs are a footnote because there's no placebo-clean edge to tax. Nothing to deploy. |
| **"Breaks up specifically"?** | ![Busted](https://img.shields.io/badge/Breaks_up_specifically%3F-Busted-8b949e?style=flat-square) | The **down**-break of the identical figure also drifts **up** (+1.20% / +2.16% at 20/40d, *t* = 2.62 / 3.39). The pattern's whole premise is *direction*, and direction carries no information. |

> **In one sentence:** the ascending triangle's breakout *is* followed by a real-looking rise (excess over base rate at *t* ≈ 3.8) — but a same-tape placebo of random dates beats it one time in five, the *down*-break of the same figure drifts up just as much, and a zero-edge synthetic control reproduces the very same *t*, so the rise is the stocks' own momentum into resistance, not the triangle breaking upward.

## What we tested

Chart figures are partly subjective, so we wrote down the **closest mechanical definition** of the
ascending triangle and tested *that* (saying so loudly): a **flat resistance** (≥3 swing highs within
4% of a level) + a **rising-low trendline** (positive slope, last low ≥5% above the first) + a
**confirmed breakout** (first close above the rim). On daily auto-adjusted OHLC for **SPY + 29 US
large-caps** (2005→2026), we enter the next close after each breakout (one documented lag) and
measure the forward 5/10/20/40-day return as an **excess over each name's own buy-and-hold base
rate** — the honest version of *"does it beat buy-and-hold?"* The Signal axis pairs a one-sample/HAC
*t* with a **same-tape random-date placebo** (the arbiter that exposes drifting-tape false positives)
and a **down-break symmetry test** (same figure, opposite resolution — the sharpest within-study
control); a deterministic synthetic control with a *planted* post-breakout drift proves the detector
+ inference can bank a real edge when one exists, and that the naive *t* can fire on geometry alone.
The basket is **survivors** — named on the Signal axis (the bias tilts the test *for* the figure, and
it still fails).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what an ascending triangle is, why "it went up after the breakout" is mostly the baseline drift, and why the *down*-break going up too is the giveaway — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the objective detector, forward excess returns by horizon, the one-sample/HAC *t* vs the same-tape placebo, the down-break symmetry test, detector-strictness robustness, the SPY small-*n* trap, and a synthetic faithful-engine / power control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`ascending_triangle/`](ascending_triangle/). Detector is one transparent mechanical definition of the figure — not the only one a human would draw. Basket is **survivors** — named on the Signal axis. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
