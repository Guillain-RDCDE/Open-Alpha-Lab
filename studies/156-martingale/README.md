# Study 156 — Martingale

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | Martingale excess over buy-and-hold: **−23 bps/episode**, HAC *t* = **−0.31**; every ticker \|*t*\| < 1.5 across 592 real episodes (SPY, QQQ, AAPL, GE, 1993–2026). |
| **Tradability** — does it survive costs, capacity, tail risk? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Real ruin rate **6.2%**; P&L skew **−3.1**; capital requirement grows as 2^K − 1 (63x for 6 levels). Ruin events occur in every extended market history. |
| **Beats B&H?** | ![Busted](https://img.shields.io/badge/Busted-8b949e?style=flat-square) | The **93.8%** win-rate is exit-asymmetry, not edge. Buy-and-hold of the same capital earns **+337 bps/episode net** vs the martingale's **+314 bps** — the simpler strategy wins. |

> **In one sentence:** the martingale / averaging-down strategy picks up many small gains at the cost of a fat left tail — the 94% win-rate is real but uninformative, the ruin risk is real and non-diversifiable, and buy-and-hold of the same total capital beats it in expectation with no tail risk.

## What we tested

A staple of retail trading forums: enter long, add to the position (double) each time it falls 5% from your last entry, take profit when price recovers 5% above the initial entry — *"it lowers your average cost and makes recoveries pay off bigger."* We take that literally. The baseline is a **buy-and-hold of the same total capital** (up to 63× the initial stake for 6 levels) over each episode window — so the comparison is honest, not just against the initial buy. We run it on four real tapes (SPY since 1993, QQQ since 1999, AAPL and GE since 1993), giving **592 non-overlapping episodes** across ~30 years, expose the ruin-rate and P&L skew profile, and derive the ruin probability by Monte Carlo over 5,000 synthetic paths. A synthetic tape with tunable mean-reversion serves as the positive control (the engine finds edge when mean-reversion is planted; the real market simply has none at this scale).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the recipe, the win-rate trap in plain language, the ruin vs capital trade-off, why B&H wins |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | per-ticker HAC *t*, ruin probability sweep, P&L skew vs capital levels, synthetic positive control, gambler's ruin mathematics |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`martingale/`](martingale/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
