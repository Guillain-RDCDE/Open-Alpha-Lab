# Study 976 — The Family Tree 🌳

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the clustering step change the portfolio at all? | ![Real](https://img.shields.io/badge/Real-2ea44f?style=flat-square) | The tree is not decoration: HRP's weights differ from plain inverse-variance weights by **16.7%** of the book on the 40-name panel and 9.3% on the multi-asset one, and the clustering changes the ordering on 3 of 3 panels. What it buys is a different question: HRP holds an effective **26.5** positions against 3.3 for the optimiser, with a maximum weight of 8.4% against 31.6% and no shorts at all. |
| **Tradability** — does it beat the optimiser, and 1/N, out of sample? | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) | On the wide panel HRP realises **13.47%** annualised against **13.07%** for minimum variance (paired *t* = -1.01) and **16.20%** for 1/N (*t* = +11.20). Against the control that matters — inverse variance, which is HRP without the tree — the difference is -0.527% (*t* = +6.50). Turnover is 0.27 a rebalance against 1.54, which is where the practical case for it is strongest. |

> **In one sentence:** Hierarchical risk parity does what it says — no matrix inverse, no shorts, weights that move 18% as much as the optimiser's — and most of its out-of-sample volatility advantage comes from being a **risk-weighted long-only book**, not from the clustering: plain inverse variance is within 0.53% of it.

## What we tested

Marcos López de Prado's **hierarchical risk parity** (2016) builds a portfolio without
ever inverting a covariance matrix: measure the distance between assets as
`sqrt(0.5(1 − ρ))`, cluster them into a tree, reorder the matrix so that relatives sit
together, then split risk recursively down the branches. The claim is that this avoids the
instability a quadratic optimiser inherits from the matrix's smallest eigenvalues, and delivers
lower out-of-sample volatility with far steadier weights. We implement all three steps from
scratch — no SciPy dependency — and race HRP against **minimum variance**, **inverse variance**,
**equal risk contribution** and **1/N** on three panels: eleven sector ETFs, forty single names
and ten multi-asset sleeves, rolling, out of sample, with costs and one day of execution lag.

The control that decides the study is **inverse variance**: HRP with the hierarchy switched
off. Any advantage HRP has over the optimiser that inverse variance also has is an argument for
*risk weighting a long-only book*, not for clustering — a distinction the literature's
enthusiasm frequently loses. A Monte Carlo with planted block correlation (strong blocks, weak
blocks, and none at all) establishes what the tree is worth when the structure it looks for is
known to be there.
**Dedup:** distinct from **975-covariance-shrinkage** (a better matrix for the same optimiser),
**977-max-diversification** and **978-resampled-frontier** (other allocation rules),
**890-sector-risk-parity** and **896-risk-parity-trend** (risk parity applied to a specific
sleeve rather than compared as an estimator), and **171-naive-1-over-n** (the benchmark, which
appears here as one of the five competitors).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the family tree drawn from the correlation matrix, what recursive bisection does to the weights, and the control that separates hierarchy from risk weighting |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | linkage and quasi-diagonalisation from scratch, five weightings compared out of sample on three panels with paired tests, turnover and concentration, window sensitivity and a planted-block Monte Carlo |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`hrp/`](hrp/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
