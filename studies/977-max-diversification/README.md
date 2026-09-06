# Study 977 — Maximum Diversification 🧭

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the diversification ratio a distinct objective, or minimum variance in disguise? | ![Real](https://img.shields.io/badge/Real-2ea44f?style=flat-square) | It is a genuinely different objective. The most diversified portfolio's weights differ from the minimum-variance ones by **100%** of the book on the multi-asset panel and 35% on sectors, and it achieves an in-sample diversification ratio of **1.78** against 1.21 — by construction, but it is worth confirming the implementation does what the formula says. On a panel of equal-volatility assets the two objectives collapse into one, and the two weight vectors agree to 1.9e-16, which is the boundary of the distinction. |
| **Tradability** — does it survive out of sample and pay for its turnover? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | The objective is delivered but the advantage is not free money. Out of sample the realised diversification ratio slips to 1.79 (+1% of the in-sample value), realised volatility is **9.95%** against 7.23% for plain inverse volatility (paired *t* = -2.96) and 9.58% for 1/N, and it turns over 0.32 a rebalance against 0.08. It beats the free competitor on **0 of 2** panels. |

> **In one sentence:** Maximum diversification is a real and distinct objective — it holds an effective 4.2 positions against the optimiser's 1.5 and delivers the higher diversification ratio it promises — but most of what it achieves out of sample is available from inverse-volatility weighting, which requires no correlation matrix and no optimiser at all.

## What we tested

Choueifaty and Coignard's **diversification ratio** is the weighted average
volatility of a portfolio's holdings divided by the volatility of the portfolio itself; the
*most diversified portfolio* maximises it, and an asset-management business was built on the
idea. We implement it in closed form — maximising the ratio is minimum variance on the
*correlation* matrix, unscaled by volatility, with a projected-gradient pass for the long-only
constraint — and race it against **minimum variance**, **inverse volatility**, **1/N** and
**equal risk contribution** on eleven sector ETFs and ten multi-asset sleeves, rolling and out
of sample with costs.

Two things decide the study. First, is it actually a different portfolio? We measure how much
of the book differs from minimum variance, and pin the **degenerate case** — equal volatilities
— where the two objectives are provably the same problem, which is both a sanity check on the
implementation and the honest boundary of the claim. Second, does it beat the *free* competitor?
Inverse-volatility weighting captures much of the same intuition without a correlation matrix
or an optimiser, and if the MDP cannot beat it, the correlation machinery is not earning its
keep. Alongside realised volatility we report the diversification ratio **out of sample**,
because a quantity that is maximised in-sample and slips afterwards is telling you it was
partly estimation error.
**Dedup:** distinct from **975-covariance-shrinkage** (a better matrix for the same optimiser),
**976-hierarchical-risk-parity** (a different construction avoiding the inverse),
**974-diversification-saturation** (how many assets, not how to weight them),
**171-naive-1-over-n** and **890-sector-risk-parity** (specific weighting schemes on specific
sleeves).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what 'diversification' means once you write it as a ratio, the portfolio it produces, and the free alternative that gets most of the way there |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the closed form and its brute-force check, the degenerate equal-volatility case, five weightings out of sample with paired tests, in-sample versus delivered diversification ratios and window sensitivity |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`max_div/`](max_div/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
