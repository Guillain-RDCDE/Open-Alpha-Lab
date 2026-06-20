# Study 350 — Dartboard-Portfolio 🎯

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does random *selection* beat the index? | ![Mixed](https://img.shields.io/badge/Mixed-dab617?style=flat-square) | The dart-minus-index spread is statistically real on our tape (HAC *t* = −3.66) but here it's **negative**, and the dartboard is indistinguishable from the equal-weight index (−0.26%/yr, *t* = −2.17): it's a size tilt, not a stable effect. |
| **Tradability** — is there an edge to harvest? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | No stock-picking edge — only a known size/concentration tilt (a loser on a mega-cap tape) carried at high random-rebalance turnover. |
| **A blindfolded monkey beats the pros?** | ![Busted](https://img.shields.io/badge/Busted-8b949e?style=flat-square) | The monkey *tracks the equal-weight index*, not skill. On mega-caps in a mega-cap decade, 0 of 500 throws beat the index or the expert. |

> **In one sentence:** the famous "monkey beats the experts" is a size tilt in disguise — the dartboard is just a noisy equal-weight bet, so it beats a cap index only when small-caps lead, and on a mega-cap universe in a mega-cap decade it loses to both the index and the experts, every single throw.

## What we tested

Burton Malkiel's quip from *A Random Walk Down Wall Street*: *"a blindfolded monkey throwing darts… could select a portfolio that would do just as well as one carefully selected by experts"* — revived by Research Affiliates' finding that thousands of *random* portfolios beat the cap-weighted index. We run the experiment as a cross-sectional race: 500 blindfolded monkeys each pick 10 names at random from a US mega-cap universe and hold them equal-weight, raced against the cap-weighted index, a concentrated expert (the 10 largest names), and — the decisive control — an equal-weight index that already holds the size tilt the monkey is supposedly exploiting. (Distinct from [Study 171](../../171-naive-1-over-n/), which tests 1/N *allocation*; this is random *selection*.)

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the legend, why a monkey "wins," and the catch the headline never mentions in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the 500-monkey distribution, HAC *t* and block-bootstrap CI, the equal-index size-tilt control, costs, and the synthetic positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`dartboard_portfolio/`](dartboard_portfolio/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
