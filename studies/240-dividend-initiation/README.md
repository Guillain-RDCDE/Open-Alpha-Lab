# Study 240 — Dividend-Initiation

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## The claim

> Do first-time dividend payers signal a durable re-rating?

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![NONE](https://img.shields.io/badge/NONE-c0392b?style=flat-square) | Initiator spread vs EW basket: **+2.1%/yr**, HAC *t* = **+0.40** (8 events, 2003–2025). Too few events per year, too much noise. No inference possible. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Max 3 events/year in a 50-name universe → impossible to diversify; transaction costs and market-impact dominate; universe is survivorship-biased. |
| **Myth check** — does signalling theory hold here? | ![No Edge](https://img.shields.io/badge/No_Edge-8b949e?style=flat-square) | Short-window announcement returns are real (Asquith & Mullins 1983); long-horizon one-year returns are noise (n = 8, t = 0.40 vs EW). The re-rating happens at announcement, not over the forward year. |

> **In one sentence:** dividend initiation events are too rare in a large-cap universe to measure a one-year forward-return premium — we get 8 initiator years in 25 years of data, and the +2.1%/yr spread vs the equal-weight basket dissolves into noise at a t-stat of 0.40.

## What we tested

Signalling theory (Bhattacharya 1979, Miller & Rock 1985) predicts that paying a first dividend is a costly, credible signal of future earnings. We identify the **first calendar year with annual dividends ≥ $0.01/share** from yfinance for a **50-name large-cap universe (survivorship-biased — named, not hidden)**, and measure the forward calendar-year total return of these first-time payers vs (a) the equal-weight full basket and (b) an arm of persistent non-payers in the same universe. A synthetic panel with a tunable planted premium confirms the engine detects a real premium when one is planted.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the signalling claim, the eight initiation events, why short-window ≠ long-horizon, structural barriers |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | per-year returns, control arm composition problem, HAC t-stats, small-sample power analysis, synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`dividend_initiation/`](dividend_initiation/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
