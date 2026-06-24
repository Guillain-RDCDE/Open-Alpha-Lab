# Study 447 — Gann Angles 📐

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the 1x1 angle predict? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | Long-when-above the 1x1 line, the active-minus-benchmark daily spread has HAC **t = −1.01** (SPY), **−1.75** (^DJI), **−2.87** (AAPL), **−1.60** (GLD) — **all negative**, none near the **+2** bar. The angle *subtracts* return on every market tested. |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | Strategy Sharpe **0.58** vs buy-hold **0.51** on SPY (worse on the other three), after **353** costed switches. No horizon, cost level or instrument where timing the line beats simply holding. |
| **Predictive power?** | ![Busted](https://img.shields.io/badge/Predictive_power%3F-Busted-8b949e?style=flat-square) | A same-shape random on/off rule does just as well — placebo **p = 0.35 → 0.90**. The real angle is indistinguishable from a coin flip. The planted control proves the test *would* see a real line effect (t up to **+6.96**); markets don't have one. |

> **In one sentence:** W.D. Gann's mystical 1x1 angle — a fixed-slope line that supposedly turns into support/resistance — has **zero** predictive power on SPY, the Dow, Apple and gold; timing it underperforms buy-and-hold everywhere (HAC t negative on all four), it is statistically indistinguishable from a random on/off rule (placebo p up to 0.90), and a planted positive control confirms our detector would light up on a real line effect (t = +6.96) — so the silence on real tape is a genuine None, not a blind spot.

## What we tested

We make the subjective objective. For `SPY`, `^DJI`, `AAPL` and `GLD` (yfinance daily, 2000→2025) we find **confirmed swing lows** (centred 21-bar minima, usable only after they print — no look-ahead) and draw the literal Gann **1x1**: an arithmetic line rising a fixed price step per bar (calibrated to the chart's own scale so it sits at ~45°), re-anchored at each new pivot low. The trade is **long when the close is above the line, flat when below**, entered one bar later, costed at 2 bps × turnover. The Signal axis is a **HAC (Newey-West) one-sample t** on the daily active-minus-benchmark spread; the myth-check is a **same-shape random-regime placebo** (matched on-fraction and run length — the honest "is it the angle, or just being in cash part-time?" null); we also test the latched "trend holds until the angle breaks" variant. A deterministic synthetic market with a *planted* line effect proves the detector is unbiased at zero edge and powerful when an edge is real.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what a Gann 1x1 angle is, the line drawn on the S&P, why timing it loses to buy-and-hold, and why a random rule does just as well — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the 1x1 from mechanical pivots, HAC *t* on the active-minus-benchmark spread across four markets, the same-shape random-regime placebo, the latch test, and a planted-edge faithful-engine / power control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`gann_angles/`](gann_angles/). The 1x1 is an arithmetic fixed-slope line from confirmed centred-window pivot lows; the race is strategy vs buy-and-hold of the same total-return series. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
