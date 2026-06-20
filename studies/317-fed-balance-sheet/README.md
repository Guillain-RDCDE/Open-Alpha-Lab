# Study 317 — Fed-Balance-Sheet 🏦

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | The slogan rests on QE beating QT. The QE−QT daily-return gap is just **+2.54 bps**, 95% CI **[−4.10, +9.15]**, bootstrap *p* = **0.47** — stocks rose in *both* regimes (QT HAC *t* = **+2.35**). |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | "Fight the Fed in QT" loses to buy-and-hold: **−2.8%/yr** sitting in cash, **−5.5%/yr** shorting (Sharpe 0.28 vs 0.55). Costs are irrelevant — there is simply nothing to harvest. |
| **"Don't fight the Fed"?** | ![Busted](https://img.shields.io/badge/Busted-8b949e?style=flat-square) | Over 33 years, fighting the Fed when it tightened would have **cost** you. The chart-overlay confuses "the Fed eases into crashes that then rebound" with "QE drives stocks". |

> **In one sentence:** stocks went up while the Fed was *both* easing and tightening, so "Don't fight the Fed" times nothing — and the literal QT-avoidance rule loses to just holding the index.

## What we tested

The macro-Twitter version of **"Don't fight the Fed"**: liquidity drives everything, so go
long stocks when the Federal Reserve is expanding its balance sheet (QE) and get out — or
short — when it is shrinking it (QT). We sort 33 years of daily SPY returns by the Fed's
**announced balance-sheet direction** (QE / QT / neither — a hand-built regime table standing
in for the network-blocked FRED `WALCL` series), test the decisive QE-vs-QT contrast with a
HAC *t* and a circular-block-bootstrap CI, and race a literal cash-in-QT / short-in-QT timing
rule against buy-and-hold on an excess-of-cash Sharpe. A deterministic synthetic tape with a
tunable QE>QT drift is the positive control that proves the harness can find the edge — when
one is actually there.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the slogan, why QT-up-too kills it, and the "Fed eases into the bottom" trap in plain English |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | regime means with HAC *t*, the QE−QT bootstrap CI, the excess-Sharpe race, costs, the synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`fed_balance_sheet/`](fed_balance_sheet/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
