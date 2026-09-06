# Study 978 — The Resampled Frontier 🎲

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does resampling change the portfolio in a way that survives out of sample? | ![Real](https://img.shields.io/badge/Real-2ea44f?style=flat-square) | Averaging a thousand optimisations does produce a different portfolio: it differs from the single-shot answer by **45%** of the book, holds **10** positions against 4, and caps its largest weight at 37% against 54%. Against a **known** true covariance and mean it also helps: the utility gap falls from 0.0183 to **0.0150** (0.0169 for shrinkage, 0.0234 for 1/N). The averaging is doing something real — the question is whether it is doing anything *distinctive*. |
| **Tradability** — does it beat the cheap fixes it competes with? | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) | Out of sample on the 10-sleeve panel, resampling returned **+6.79%/yr** at 6.91% volatility (Sharpe +0.98) against +6.16% / +0.85 for the plain optimiser (paired *t* on the return difference +0.97) and +5.90% / +0.93 for plain shrinkage (*t* = **+1.07**). Its weights sit **45%** from the shrunk portfolio's and 45% from the plain one — closer to the cheap fix than to the thing it is fixing, which is the whole story. It also costs 60 optimisations per rebalance. |

> **In one sentence:** Resampling works, in the sense that it produces a more diversified portfolio and a smaller utility gap against a known truth — but it lands **45%** away from what a default shrinkage produces in one pass, beats it by +0.05 of Sharpe with *t* = +1.07, and costs 60 optimisations to get there.

## What we tested

Richard Michaud's **resampled efficiency** answers estimation error by embracing it:
draw hundreds of bootstrap samples of the return history, run the same optimisation on each,
and average the weight vectors. The portfolios come out smoother, more diversified and less
prone to the corner solutions that make a mean-variance optimiser embarrassing. The method has
been defended and attacked for a quarter of a century, largely without anyone putting it next
to the boring alternative.

This study does. Michaud resampling — both the parametric Monte Carlo original and the
non-parametric row bootstrap — runs against **plain optimisation**, a **default shrinkage**
(Ledoit-Wolf covariance plus a 50% pull of the expected returns toward their average) and
**1/N**, on ten multi-asset sleeves and eleven sector ETFs, under two objectives (minimum
variance, which uses only the covariance, and maximum Sharpe, where expected-return error is
catastrophic and resampling is supposed to shine). Everything is long-only, rolling and out of
sample with costs, and every comparison is paired.

Two things settle it. First, the **weight distance**: how far the resampled portfolio sits from
the thing it fixes versus from the cheap fix. Second, a **known-truth simulation** — the only
setting where "produces a better portfolio" is falsifiable — scoring each method by the utility
it gives up against the portfolio the true parameters imply.
**Dedup:** distinct from **975-covariance-shrinkage** (shrinkage appears here only as the
competitor), **976-hierarchical-risk-parity** and **977-max-diversification** (different
allocation rules, no resampling), **968-bootstrap-choice** (bootstrapping for *inference*, not
for portfolio construction) and **171-naive-1-over-n** (the benchmark).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why an optimiser produces embarrassing portfolios, what averaging a thousand of them looks like, and the one-line alternative that lands in almost the same place |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | both resampling flavours, long-only projected optimisers checked by brute force, paired out-of-sample tests on two objectives and two panels, weight-distance geometry and a known-truth utility-gap experiment |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`resampled/`](resampled/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
