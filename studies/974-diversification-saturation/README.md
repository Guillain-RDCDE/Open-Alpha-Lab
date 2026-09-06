# Study 974 — The Nth Asset 🧩

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does risk keep falling as asset classes are added? | ![Real](https://img.shields.io/badge/Real-2ea44f?style=flat-square) | A single randomly chosen asset from this universe ran **15.4%** annualised volatility; an equal-weight portfolio of all 12 runs **10.6%** — a **31%** reduction. The empirical curve tracks the textbook one (sigma^2 = sigma^2/k + rho·sigma^2·(k−1)/k) closely: average pairwise correlation is **0.35**, which puts the theoretical floor at **10.0%** — and the twelve-asset portfolio is already within 0.6% of it. |
| **Tradability** — is there a number a portfolio should stop at? | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) | Yes, and it is smaller than the industry implies. The third asset removes **9.0%** of the portfolio's volatility; the twelfth removes **0.6%**. At a 5% relative-improvement threshold the curve stops paying at **k = 3**; at 2% it stops at **k = 7**. And the count that matters is not the nominal one: 12 assets here are worth **1.4 effective bets**. Choosing well beats choosing many — the best 3-asset combination reached **6.2%** against **12.4%** for a random one of the same size. |

> **In one sentence:** Diversification is real and it saturates fast: most of the **31%** volatility reduction available in this twelve-asset universe arrives by the 3th holding, the rest of the universe is worth 14.3% more, and the twelve nominal assets amount to 1.4 genuinely independent bets.

## What we tested

"Add another asset class" is the reflex answer to every portfolio question, and the
industry has an infinite supply of them. This study prices the *k*-th one. From a universe of
twelve genuinely different exposures — US large and small cap, developed and emerging
international, long and intermediate Treasuries, investment-grade and high-yield credit, gold,
broad commodities and two property sleeves — we build equal-weight portfolios of every size from
one to twelve, drawing the constituents **at random** three hundred times per size so the answer
cannot be an artefact of the order somebody listed the assets in. Weights drift between monthly
rebalances and every rebalance is charged, because a twelve-asset book is not free to maintain.

The empirical curve is then set against the closed form `σ²/k + ρσ²(k−1)/k` evaluated with the
sample's own average variance and correlation, which predicts both the shape and the floor —
the average covariance, below which no amount of diversification can go. Two things turn that
into advice: a **stopping rule** (the first k whose successor buys less than a stated relative
improvement, swept from 10% to 1%), and **Meucci's effective number of bets**, which counts how
many independent exposures twelve correlated funds actually amount to. A greedy in-sample
ordering, clearly labelled as hindsight, bounds what choosing well rather than choosing many
could buy.
**Dedup:** distinct from **171-naive-1-over-n** (1/N versus optimisation at a *fixed* universe
size), **1007-how-many-stocks** (idiosyncratic risk inside a single equity market — the
Evans-Archer question), **975-covariance-shrinkage** (estimating the matrix better rather than
choosing how many assets), **68-all-weather** / **203-golden-butterfly** / **205-three-fund**
(specific named allocations) and **976-hierarchical-risk-parity** (weighting, not counting).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why the second asset is worth more than the tenth, the curve and the floor it cannot pass, and how many holdings the arithmetic actually supports |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | random-subset distributions per portfolio size, the closed-form comparison, marginal-benefit stopping rules under four thresholds, Meucci effective bets, a greedy in-sample ceiling and correlation-controlled synthetic panels |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`diversify_n/`](diversify_n/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
