# Study 79 — Sleigh-Ride 🎿

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | 7-day window mean **+121 bps**, HAC *t* = **+4.45** on ^GSPC (76 years); daily Santa vs baseline diff *t* = +4.16. Real, but the per-window std is 252 bps and SPY alone gives only *t* = 2.28. |
| **Tradability** — does it survive costs, capacity, scale? | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) | Fires once a year → costs are trivial; but ~1.2%/yr gross from 7 trading days is mostly equity beta, not tradable alpha. Too thin to build a strategy around. |
| **'If Santa Fails'?** | ![Not_Supported](https://img.shields.io/badge/If_Santa_Fails%3F-Not_Supported-8b949e?style=flat-square) | Negative windows actually precede *higher* forward returns (+183 bps vs +35 bps); diff *t* = −1.11 — no evidence for the bears-come claim on 18 negative-window events. |

> **In one sentence:** the Santa Claus Rally is a statistically real but razor-thin seasonal tilt (~1.2%/yr, mostly equity beta) that survives 75 years of ^GSPC data at *t* = 4.45 — yet the "If Santa Fails" follow-up claim is flatly not supported and tradability is fragile: it fires once a year, barely exceeds a passive long, and has a per-window std ten times its mean.

## What we tested

A fixture of the *Stock Traders Almanac* since 1972: the last 5 trading days of the calendar year plus the first 2 of January tend to produce above-average S&P 500 returns — *"If Santa Claus should fail to call, bears may come to Broad and Wall."* We take that literally and ask three questions: (1) is the 7-day window return significantly above a random-window baseline and an all-days mean, (2) does a *negative* Santa window actually predict a bad January/Q1 (the directional follow-up), and (3) could you build a standalone trade on it? We measure both the daily return differential (HAC *t*-stat) and the per-window aggregate (76 observations on ^GSPC since 1950, 33 on SPY since 1993), compare against a 1,000-draw random-window null, and test the "If Santa Fails" claim with a forward-return comparison.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the rally in plain language, the win-rate and variance context, why it barely exceeds passive, and why the bears-come claim fails |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | daily HAC *t*-stat, per-window distribution, random-window null, "If Santa Fails" forward-return test, cost/alpha decomposition, synthetic positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`sleigh_ride/`](sleigh_ride/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
