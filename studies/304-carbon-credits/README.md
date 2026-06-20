# Study 304 — Carbon-Credits 🌍

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | Buy-and-hold KRBN compounded at **+16.2%/yr** — but the reward-over-cash HAC *t* is only **+1.41** (below the bar), and **100%+ of the compound came from a single year (2021, +108%)**. Literature-supported, this short one-cycle tape can't certify it. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | A 29%-vol, −36%-drawdown niche commodity whose excess-of-cash Sharpe (0.57) is indistinguishable from cash; the obvious trend-timing overlay makes it strictly **worse** (lower return, deeper drawdown) at every setting. |
| **Does timing beat buy-and-hold?** | ![Not supported](https://img.shields.io/badge/Not_supported-8b949e?style=flat-square) | A time-series momentum overlay loses to naive buy-and-hold gross at 63/126/252-day windows; bootstrap P(overlay wins) = **0.15 / 0.03 / 0.00**. Doing nothing beat trying to time it. |

> **In one sentence:** carbon allowances look like a one-way green escalator only because one explosive year (2021) carries the whole tape — over the full cycle a passive KRBN hold is just expensive, undistinguishable-from-cash commodity beta, and the "just time the trend" fix makes it worse, not better.

## What we tested

The marketed case for carbon credits is a buy-and-hold no-brainer: the regulatory cap shrinks every year, polluters must buy a dwindling supply, so an allowance ETF like **KRBN** (KraneShares Global Carbon Strategy ETF) should ride the energy transition upward — *"the cap only goes one way."* We take that literally on KRBN total-return daily bars since its 2020-07-30 inception, ask the only question that matters for a "reward" (does it beat **cash**, with an autocorrelation-robust *t*?), then steelman the obvious objection — *you should have timed it, not just held it* — by racing a time-series (absolute) momentum overlay against naive buy-and-hold, **excess-of-cash to excess-of-cash**, with a block-bootstrap CI on the Sharpe difference. A deterministic synthetic tape with tunable drift and persistence is the positive control.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the green-bet story, the one-year mirage, and why "just time it" backfires |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | excess-of-cash HAC *t*, the momentum overlay race, block-bootstrap CIs, synthetic controls, capacity |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`carbon_credits/`](carbon_credits/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
