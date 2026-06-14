# Study 146 — Country-Momentum

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | Absolute HAC *t* = +2.55 over 30 years, but the *active* return vs an equal-weight basket is only +1.1%/yr (*t* = +0.70). Sub-periods: momentum **underperforms** simple equal-weighting in both the 2010s and 2020s. |
| **Tradability** — does it survive costs, capacity, scale? | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) | The gross active return vs the equal-weight benchmark is near zero before any costs; realistic ETF bid-ask spreads erase it. The surviving Sharpe (+0.50) is equity beta, not rotation alpha. |
| **Survivorship?** | ![Named](https://img.shields.io/badge/Survivorship-Named-8b949e?style=flat-square) | Current 23-ETF universe projected back; in 1996 fewer tickers were available. Results are an upper bound; live performance would be worse. |

> **In one sentence:** country momentum earned a faint absolute return over 30 years — but it is mostly equity beta, the active return over simply holding all countries equally has a t-stat of 0.70, and the edge reversed sign in the 2010s and 2020s, when the equal-weight basket beat the momentum sort.

## What we tested

A staple of global macro quant books (Kakushadze & Serur 2018, §3.9; Asness, Liew & Stevens 1997): rank 23 single-country equity ETFs by their trailing 12-1 month total return each month, go **long the top-4**, equal-weight, rebalance monthly. We pit it against two controls — an **equal-weight all-country basket** (does the sort add anything over just holding the universe?) and a **random-rotation control** (does the 12-1 signal beat an uninformed picker of the same size?) — and sweep transaction costs at the monthly rebalance cadence. A deterministic synthetic panel with a tunable persistent cross-country drift serves as the positive control: the machinery harvests momentum when we plant it, and reads zero when we don't.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the recipe, the equity-beta trap, the 2010s decay, why costs finish the job |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | HAC *t*-stats, active return vs EW and random, sub-period breakdown, cost/Sharpe sweep, bootstrap CI, synthetic positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`country_momentum/`](country_momentum/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
