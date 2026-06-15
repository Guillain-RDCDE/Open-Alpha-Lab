# Study 185 — Chande-Momentum

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | OB/OS gross HAC *t* = +0.24; zero-cross HAC *t* = +0.14 — both framings indistinguishable from a random-direction coin at every hold period tested. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Gross ≈ +2.8 bps/trade on OB/OS (n = 546), below the noise floor — costs of 1 bp net to +1.8 bps (*t* = +0.15), statistically zero. |
| **Symmetric oscillator?** | ![Inconclusive](https://img.shields.io/badge/Inconclusive-8b949e?style=flat-square) | CMO's pure-momentum normalisation adds no measurable signal over RSI or %R family members on daily equities. |

> **In one sentence:** the Chande Momentum Oscillator's overbought/oversold and zero-cross framings, tested on five liquid daily tapes over 10 years against a random-direction control, produce t-stats below 0.25 on every combination of hold period tested — a well-powered null result.

## What we tested

Tushar Chande's CMO (1994) measures net momentum as 100 × (up-sum − down-sum) / total movement over a 14-bar window, placing it in the range [−100, +100]. Unlike RSI, the denominator includes *all* absolute movement, so CMO is a pure momentum oscillator. Two folk recipes are tested: (1) extreme CMO (|CMO| > 50) as a mean-reversion trigger, and (2) CMO zero-cross as a trend-follow trigger. Both are pinned against a **random-direction control** on identical entry bars with 5-day forward returns across SPY, QQQ, IWM, GLD, and TLT (2016–2026, ~2,500 bars each). Hold periods from 1 to 20 days are swept.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the CMO recipe, the two framings in plain language, the fair-bet vs coin chart, why costs don't even matter here |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | per-instrument HAC *t*, hold-period sweep, cost sweep, synthetic positive control confirming the engine works |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`chande_momentum/`](chande_momentum/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
