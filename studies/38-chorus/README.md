# Study 38 — Chorus 🎺

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | The diversification mechanism is real and visible — components are −0.03 correlated and the momentum+reversal pair (gross Sharpe **0.66**) beats both its parts — but the naive 3-signal chorus is a flat **0.00** gross once an off-key voice is included. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Turnover **0.38**/day ⇒ **break-even 0.02 bp**; net-negative the instant any realistic cost is charged. |
| **Whole > parts?** | ![Mixed](https://img.shields.io/badge/Mixed-dab617?style=flat-square) | `CONFIRMED` for a curated decorrelated pair (0.66 > 0.36, 0.64); `NOT SUPPORTED` for the naive "throw them all in" blend — a decorrelated *loser* dilutes, it doesn't diversify. |

> **In one sentence:** combining several weak, decorrelated signals into one book really *can* out-sing every soloist — the momentum+reversal pair does — but breadth multiplies *information*, so the naive chorus is only as good as its worst sincere voice, and the desk's capstone lesson ("the edge is diversification, not prediction") comes with fine print: a decorrelated bet with negative expectancy subtracts.

## What we tested

Kakushadze & Serur's *151 Trading Strategies* §3.20 (**"combining alphas"**) and the Fundamental Law of
Active Management (Grinold-Kahn: Sharpe ≈ IC · √breadth) make the desk's recurring claim at full strength:
no single anomaly is impressive alone, but a portfolio of several **weak, decorrelated** signals has a
materially better Sharpe than any component. We build three simple, causal, dollar-neutral cross-sectional
signals on the **current S&P 500** (263 names, 2010–2026) — momentum ([Study 24](../24-stampede/)),
short-term reversal ([Study 33](../33-slingshot/)) and a low-volatility tilt — combine them both
equal-weight and inverse-vol, and ask whether the chorus beats every soloist, and whether it survives its
own turnover. This is the capstone of the desk's first 37 studies.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why three so-so signals can beat any one of them, and why "throw them all in" still isn't a free lunch |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | standalone-vs-combo Sharpe, the pairwise-correlation matrix, the breadth (√k) sweep, equal-wt vs risk-parity, the cost wall |

The fingerprinted real run is in [docs/results.md](docs/results.md); the beat-7 breadth-and-scheme
complement (breadth is the lever, the blend scheme is not) is worked in [docs/extension.md](docs/extension.md).

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine: [`quantlab/`](../../quantlab/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
