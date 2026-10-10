# Study 1022 — Behind the Curve 🐢

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the Taylor-rule gap forecast stocks or bonds? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | On 163 quarters (1969Q1–2009Q3) the next-quarter equity excess return loads **+0.14% per point** of real-time Taylor gap — the *wrong* sign — with Newey-West *t* = +0.43, Stambaugh-corrected *t* = +0.47 and a simulation-null **p = 0.69**; the unemployment-gap version agrees (*t* = +0.55), and 4- and 8-quarter horizons are no better. The bond leg comes closest: AAA yields rose 0.037 pp a quarter per point of "too easy" (*t* = −2.04), but against a null built for a persistent regressor p = **0.06** (one-sided 0.04), and what there is lives before 1987. None of the twelve specifications clears the pre-registered bar — on final-vintage data that flatter the claim (Orphanides 2001). |
| **Tradability** — does "stocks when the Fed is behind, bills when ahead" pay? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | One lag, 10 bp one-way: net excess Sharpe **0.12 vs 0.28** for buy-and-hold (ΔSR −0.17, block-bootstrap 95% CI [−0.41, +0.03]); 1.77% a year over bills against 5.12%, timing alpha −1.86% (*t* = −1.22). It trades about once a year, so costs are not what kills it — the gross edge is already negative, and a real-time-demeaned variant fares no better. |

> **In one sentence:** Being behind the curve bought stocks no tailwind — the real-time Taylor gap's one-quarter equity slope is +0.14% per point (simulation p = 0.69), only the bond headwind comes close (p = 0.06), and the timing rule's net Sharpe was 0.12 against 0.28 for buy-and-hold, on final-vintage data that flatter it.

## What we tested

The macro-desk staple: when the Fed sits **behind the curve** — its policy rate below what the
[Taylor (1993)](docs/references.md) rule `i* = 2 + π + ½(π − 2) + ½·gap`
prescribes — easy money is a tailwind for stocks and a headwind for bonds; when it is **ahead**
(too tight), sell stocks. We build the gap only from what an investor could have computed —
a one-sided (past-only) HP output gap and an Okun-style unemployment gap, every macro print lagged
a quarter, the 3-month T-bill as the policy-rate proxy — on statsmodels' **final-vintage** US
macro, Fama-French total returns and Moody's AAA yields (all SHA-pinned, offline). Predictive
regressions use Newey-West, the Stambaugh (1999) correction and a simulation null with an AR(1)
regressor whose shocks are resampled jointly with returns (overlap-correct at 4–8 quarters);
the trade is a sign-of-the-gap equity/bills switch raced excess-vs-excess against buy-and-hold.
Orphanides' (2001) real-time critique is carried throughout: every number is an upper bound.
**Dedup:** distinct from **118-fed-model** (earnings yield vs bond yield, a valuation spread, not
the policy stance), **119-real-rate-regime** (the level of real long rates), **925-short-rate-momentum-switch**
(the bill yield's own trend), **985-last-hike-timing** (an event study of cycle turning points)
and the parallel **1014-macro-clairvoyance** (what a *perfect* macro forecast is worth); none asks
whether the policy rate's distance from a policy *rule* times markets.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the Fed vs the rule over forty years, the gap you could know vs the one a historian sees, and why "too easy" was not followed by better quarters |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | real-time gap construction, Newey-West regressions, the Stambaugh correction, the simulation null at 1/4/8 quarters, the 1987 regime split, the timing race with a bootstrap, and a planted-slope control |

Reproducible headline run (provenance, SHA pins, every table): [docs/results.md](docs/results.md).

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine: [`quantlab/`](../../quantlab/) + [`taylorgap/`](taylorgap/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
