# Study 348 — Curve-Fitting 🎚️

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the optimised crossover have a real edge? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | On a random walk the IS-best pair reverts to noise out of sample (Sharpe 0.94 → −0.05, HAC *t* = −0.15). On real SPY its OOS Sharpe is pure beta (84% exposed to a bull run); its alpha over buy-and-hold is **−6.1%/yr**, HAC *t* = −2.70. |
| **Tradability** — is there an edge to harvest? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | The tuned winner *underperforms* simply holding the index, while paying turnover to chase a pair fitted to the past. There is nothing to trade. |
| **Does the in-sample winner survive out of sample?** | ![Busted](https://img.shields.io/badge/Busted-8b949e?style=flat-square) | The in-sample Sharpe is inflated by the grid search and does not transfer — and it inflates *further* the bigger the grid. A genuinely planted signal *does* survive (the synthetic control proves it), so the method is a detector, not a generator of edges. |

> **In one sentence:** tune a moving-average crossover until the backtest sparkles and the winning parameters carry nothing into the future — on a coin-flip tape the in-sample Sharpe of ~1 collapses to zero out of sample, and on real SPY the "survivor" is just the market's beta wearing a costume.

## What we tested

The most seductive way to build a strategy: sweep a rule's parameters, keep the configuration with the best backtest, and trust it to keep working. We make the trap concrete with a moving-average crossover — two integer windows, a 93-pair grid — and run one honest protocol: optimise on the in-sample half, then judge the crowned winner, untouched, on the out-of-sample half. We do it on a deterministic synthetic tape (a planted-trend *positive control* and a pure-random-walk *null*) and on real SPY total return, then strip out the market beta that flatters any long/flat timing rule. The subject isn't the crossover — it's whether *picking the best of many rules* on past data tells you anything at all.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why a tuned backtest looks brilliant, the in-sample → out-of-sample collapse, and the tell that it was always noise — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the IS−OOS shrinkage vs grid size, HAC *t* and block-bootstrap CI, the alpha-vs-beta strip, and the synthetic positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`curve_fitting/`](curve_fitting/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
