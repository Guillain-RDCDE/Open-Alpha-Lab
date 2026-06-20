# Study 347 — Look-Ahead-Bias 🔮

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is there a real, tradable edge? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | The *correctly lagged* momentum rule has no edge on real SPY (HAC *t* = −0.77). The headline is an accounting artefact, not a forecast. |
| **Tradability** — could you trade the headline? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | The Sharpe-4.9 "strategy" requires filling on a close that has not happened yet — un-executable at any cost (and the bias survives costs *because* it isn't an edge). |
| **Does a one-bar peek inflate the backtest?** | ![Confirmed](https://img.shields.io/badge/Confirmed-8b949e?style=flat-square) | A zero-edge rule jumps to Sharpe **+4.87** (*t* = 28) on real SPY the instant the signal earns its own bar; the planted positive control is bankable *only* with the correct one-bar lag. |

> **In one sentence:** let a signal earn the very bar that produced it and a nothing strategy becomes a Sharpe-5, *t*-28, cost-proof "edge" — pure look-ahead bias, which is exactly why the desk shifts every signal forward one bar, once, and says so.

## What we tested

The most expensive one-line bug in quantitative finance is a missing `.shift(1)`: a signal computed from the close of day *t* that is allowed to capture the return *of* day *t* — the same-bar fill, the look-ahead peek catalogued as backtesting "sin #1" by [López de Prado (2018)](docs/references.md) and [Luo et al. (2014)](docs/references.md). We take one ordinary 20-day z-score momentum rule and run it on the *same* tape — synthetic null, synthetic positive control, and real SPY — under two conventions, **peek** (same-bar) and **lag1** (the desk's one-execution-lag rule), and measure exactly how much free alpha the peek invents.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what a "peek" is, why it conjures alpha from nothing, and why it survives trading costs — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the HAC *t* and block-bootstrap inflation, the sign-flip that proves it's contemporaneous correlation, the synthetic positive/negative controls, and the cost-immunity tell |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`look_ahead_bias/`](look_ahead_bias/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
