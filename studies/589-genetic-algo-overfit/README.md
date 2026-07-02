# Study 589 — Genetic-Algo-Overfit 🧬

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — did the GA find a real edge? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | On a tape we *built* to have zero timing edge, a genetic algorithm evolves a rule with an in-sample Sharpe of **0.99** that goes to **0.087** out-of-sample (shrinkage **0.90**). The Deflated Sharpe Ratio is **0.00** — the IS Sharpe is *below* the luck bar (expected max **3.49** for the GA's **2,371** trials); placebo *p* **0.073**; seed-robust null OOS *t* **−0.84**. A synthetic-only method demo — no real tape, so it can never earn `REAL`. |
| **Tradability** — is there anything to harvest? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | The champion's demeaned OOS timing Sharpe is an economically meaningless **+0.087**, which a mild 2 bps turnover cost erases to **+0.012**. By construction there is nothing there. |
| **Does GA search manufacture a false edge?** | ![Confirmed](https://img.shields.io/badge/Confirmed-8b949e?style=flat-square) | The IS Sharpe climbs with the search budget (0.76 → 0.99 → 0.98 as trials go 143 → 2,371 → 5,963) while the OOS Sharpe orbits zero; the DSR and placebo catch it cold, and the positive control proves the same GA *banks* a planted edge OOS (Sharpe 2.14 at strength 0.30). |

> **In one sentence:** hand a genetic algorithm a menu of features and let it evolve a trading rule on a pure random walk, and it *will* breed a gorgeous Sharpe-1 backtest — which evaporates out of sample (shrinkage 0.90), sits *below* the deflated-Sharpe luck bar its own 2,371 trials created, and looks like noise to a placebo, while the very same optimiser cleanly banks a genuinely planted edge.

## What we tested

The seductive pitch behind every "AI finds alpha" story: don't hand-design a strategy, **evolve**
one — a genetic algorithm searches millions of rule combinations and keeps the fittest. We make the
trap undeniable by running the GA on a tape we *know* is empty: a pure random walk where, by
construction, no timing rule can work. A deterministic GA (tournament selection, crossover, mutation,
elitism) evolves a long/flat rule — a weighted combination of seven technical features — to maximise
the **in-sample Sharpe**, then we run the frozen champion, untouched, out-of-sample and put two
numbers on the wreckage: the **IS − OOS shrinkage** and the **Deflated Sharpe Ratio** (Bailey &
López de Prado), which haircuts the Sharpe for the number of genomes the GA effectively tried. The
book is *demeaned* so a long-biased rule can't inherit any drift — the collapse is overfitting, not
beta. A **synthetic positive control** with a genuinely planted edge confirms the diagnostics punish
*searching*, not *having an edge*. *A pure method demo on a synthetic world — cousin of
[344 Backtest-Overfitting](../344-backtest-overfitting/) (a grid search) and
[348 Curve-Fitting](../348-curve-fitting/) (tune two windows); 589 swaps the grid for an
**evolutionary optimiser** over a multi-feature rule.*

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what a genetic algorithm is, how it "discovers" a brilliant backtest out of nothing, and the in-sample → out-of-sample death — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the GA (selection/crossover/mutation), IS−OOS shrinkage vs search budget, the Deflated Sharpe Ratio with the expected-max-Sharpe bar, the label-shuffle placebo, costs, and the seed-robust synthetic positive control |

The fingerprinted headline run (null tape fp `e350016fa3d2`, as-of 2026-06-30) is in
[docs/results.md](docs/results.md); the whole machinery runs offline and deterministic on the
synthetic world in [`genetic_algo_overfit/data.py`](genetic_algo_overfit/data.py).

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine: [`genetic_algo_overfit/`](genetic_algo_overfit/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
