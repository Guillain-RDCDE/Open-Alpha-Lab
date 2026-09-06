# Study 987 — Gold's Loud Cousin 🥈

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is silver statistically just a levered gold position? | ![Busted](https://img.shields.io/badge/Busted-c0392b?style=flat-square) | Regressed on gold over 4,801 sessions, silver's beta is **1.45** (±0.03) with an R² of 62% — so about 38% of silver's variance is *not* gold. That leftover is not noise. Its annualised volatility is 20.3%, and it loads on outside factors with a largest |*t*| of **10.58** (on industrial). Nor is the beta a constant: over rolling one-year windows it ranges 0.94 to 2.15, a spread of **86% of its own mean**, and it is 1.29 on gold's up days against 1.57 on its down days. 'Silver is levered gold' is a reasonable first approximation and a poor second one. |
| **Tradability** — does a levered gold position beat holding silver? | ![Useful](https://img.shields.io/badge/Useful-2ea44f?style=flat-square) | Holding a 1.45× daily-rebalanced gold position instead of silver, financed at cash plus 0.5%, tracked it with a correlation of 0.79 and an annualised tracking error of **20.3%** — which is not tracking, it is a different asset. Over 19 years silver compounded at +7.7% against the replica's +11.9% (Sharpe 0.39 vs 0.56). Part of that gap is pure arithmetic: levering a 18%-vol asset 1.45× costs about **1.1% a year** in volatility drag, before financing and before costs. |

> **In one sentence:** Silver's beta to gold is 1.45 with an R² of 62%, but it wanders across a 86% range and leaves a 20%-vol residual with structure in it — so silver is levered gold in the same way a dog is a levered cat.

## What we tested

"Silver is gold with the volume turned up" is the oldest line in the
precious-metals trade, and unlike most market folklore it is arithmetically checkable. If it is
exactly true, silver is **redundant**: hold gold, size it up, done.

This study checks it three ways, because the three get conflated constantly. **Is the beta
stable?** A full-sample coefficient is meaningless if the true loading wanders between 1 and 2.5
by decade, so rolling betas are measured at four windows, split by gold's up and down days, and
split by gold's volatility. **Is the residual noise, or a second asset?** Gold is projected out
with a strictly trailing beta and the leftover is hunted for structure against the dollar, real
rates, industrials and copper — the place the story was always most likely to break, since
roughly half of silver demand is industrial and none of gold's is. **Does the replication
work?** The statistical question and the portfolio question are different: a levered gold
position rebalanced daily does not earn β times gold's return, it earns β times gold's *daily*
return compounded, which costs `β(β−1)σ²/2` a year — arithmetic, not opinion, and larger than
most alphas argued about in this space. The replica is run as a financed strategy against
holding silver, and the gold/silver ratio trade gets the same treatment.
**Dedup:** distinct from **156-gold-as-a-hedge** and **488-real-rates-and-gold** (gold's own
drivers), **291-gold-miners-leverage** (miners against their metal, which appears here only as a
sanity check), **619-commodity-momentum** (a cross-sectional commodity signal) and
**774-levered-etf-decay** (the decay of a *product*; this study computes the same arithmetic for
a position an investor constructs themselves, and asks whether it replicates a different asset
at all).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | how much of silver is really gold, what the rest is made of, and why holding twice the gold is not the same trade |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | rolling and regime-split betas, residual factor loadings, the closed-form volatility drag of leverage, the replication run as a financed strategy, the ratio trade, and a three-world synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`loudcousin/`](loudcousin/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
