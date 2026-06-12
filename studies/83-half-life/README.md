# Study 83 — Half-Life

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | 365-day HAC *t* = **+4.0** but n=3 makes it unreliable; empirical p-value vs random days = **0.112** (not significant). The 2024 halving *underperformed* BTC's own secular trend (excess = −0.23). |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Technically executable (buy-and-hold, trivial costs) but rests on n=3 data points and a clear fading trajectory; 2024 return (+35%) was far below 2016 (+265%) and 2020 (+471%). Next test: 2028. |
| **n=4 is not a law?** | ![Busted](https://img.shields.io/badge/Busted-8b949e?style=flat-square) | Two spectacular returns coincide with the greatest secular crypto bull in history; 2024 breaks the pattern. The 'halving pump' is in-sample storytelling on 3 events. |

> **In one sentence:** the post-halving BTC rally is a compelling story that doesn't survive a random-day test (p=0.112), relies on two windows from the peak secular crypto bull, and visibly broke down in 2024 — n=4 is an in-sample narrative, not a law.

## What we tested

The Bitcoin halving cycle: every ~4 years the block reward is halved, cutting new BTC supply in half. The folk theory — articulated by PlanB's stock-to-flow model and widely shared in crypto media — states that the supply shock predictably drives the price higher over the 12–18 months that follow, and that *every halving has been followed by a new all-time high*. We take this seriously: we compute the forward 90-, 180-, and 365-day log-returns after each observable halving (3 events on Yahoo's tape: 2016, 2020, 2024), pin them against a **random-day control** (2,000 Monte Carlo draws of 3 random starting days from the same tape), and decompose each halving window into secular-BTC-trend and halving-excess components via OLS. A deterministic synthetic tape with a tunable post-halving boost serves as the engine's positive control.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | raw halving returns, the random-day control in plain language, the 2024 disappointment, why secular bull ≠ halving alpha |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | HAC *t* across 90/180/365-day windows, secular-excess OLS decomposition, bootstrap p-value, power analysis (n required for detection), synthetic positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`half_life/`](half_life/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
