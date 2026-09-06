# Study 979 — The Prior Is the Portfolio 🗿

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the model do anything the prior did not already do? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | The zero-view identity is exact: with no views the posterior portfolio equals the prior to **3.2e-11** of the book, for every prior, every tau and every covariance tested. So the model contributes no information of its own — it is a blending rule. What it *does* contribute is a calibrated way to size a tilt: a 3%/yr view on one sleeve moves **4.3%** of the book at tau = 0.05, rising to 18.0% for a 10%/yr view. And the choice of prior matters more than the view: the same view under three defensible priors produces portfolios **19.3%** apart, against the 2.4% the view itself moved. |
| **Tradability** — is a view-driven tilt worth running out of sample? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | With a mechanical 12-1 momentum view sized at 3%/yr, the tilted portfolio returned **+7.24%/yr** at 8.82% volatility (Sharpe +0.82) against **+8.26%** / +0.94 for the untouched prior — paired *t* on the return difference **-1.15** across 68 rebalances. Plain mean-variance on the same data managed +7.28% / +1.09 with a largest weight of 52% against the tilted book's 18%, which is the comparison Black-Litterman was invented to win — and does. |

> **In one sentence:** Black-Litterman with no views returns the prior exactly (3e-11 of the book), so everything it produces is the prior plus the view — and since changing the prior moves the answer **19.3%** while a 3%/yr view moves it 2.4%, the model is mostly a disciplined way of holding a prior.

## What we tested

Black-Litterman is the model people reach for when mean-variance optimisation
embarrasses them, and its least-discussed property is an identity: **with no views, the
posterior portfolio is the prior, exactly**. Not approximately, not usually — exactly, for any
covariance matrix, any risk aversion and any value of the mysterious `tau`. Everything the model
produces is therefore the prior plus the views; the algebra contributes only the blend. This
study verifies that identity to machine precision (including on a deliberately singular
covariance matrix), and then does the two things the identity implies are worth doing.

First, **calibrate the view**: given a view that one sleeve will out-perform by *X* a year at
confidence `tau`, how much of the book actually moves? That converts two unitless parameters
nobody derives into a number a person can hold an opinion about. Second, **compare the view
against the prior**: the same view is run under three defensible priors (equal weight, inverse
volatility, risk parity) and the spread between the resulting portfolios is measured against the
distance the view itself moved. No market-capitalisation data is used anywhere — a price feed
does not carry it, and running three explicit priors and reporting the spread is more honest
than inventing one. Finally a mechanical 12-1 momentum view is attached and raced out of sample
against the untouched prior and against plain mean-variance.
**Dedup:** distinct from **978-resampled-frontier** and **975-covariance-shrinkage** (other
answers to estimation error, with no view mechanism), **976-hierarchical-risk-parity** and
**977-max-diversification** (allocation rules with no expected returns at all),
**902-multi-factor-composite** (combining signals rather than combining a signal with a prior)
and **507/518** (whether momentum works — here it is only a stand-in for "a view").

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the identity that means the model cannot invent information, what a 3%/yr view is actually worth in weights, and the choice that matters more than the view |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the full posterior algebra with the zero-view identity pinned on a singular matrix, view-strength surfaces in tau and size, prior-versus-view sensitivity, and an out-of-sample race under three priors |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`bl_prior/`](bl_prior/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
