# Study 72 — Loaded-Dice 🎲

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | Cross gross **−0.39 bps/trade**, HAC *t* = **−1.12**; no edge over a random-direction control (Δ = −0.53 bps), every instrument \|*t*\| < 1.4. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Negative gross at ~11 trades/day → a *significant loser* (*t* = −4.0 at 1 bp), ≈ **−40%/yr** net; no positive break-even cost. |
| **Beats a coin?** | ![Busted](https://img.shields.io/badge/Busted-8b949e?style=flat-square) | The "grab a few \$" exit manufactures a **93.5%** win-rate at skew **−8.0** (mean *t* = 1.35) — a left-tail illusion, not an edge. |

> **In one sentence:** the famous SMA(5/10) 5-minute crossover scalp is a fair die wearing a trend costume — measured honestly it ties a coin, the bragged-about 90%+ win-rate is pure exit-asymmetry, and ~11 trades a day of costs make it a ~40%/yr loser.

## What we tested

A staple of day-trading forums and "scalping bot" tutorials: on the 5-minute chart, enter in the direction of the SMA(5)/SMA(10) crossover, take a few dollars, repeat all day — *"a coin flip, but with the odds nudged your way by the trend."* We take that literally and ask whether the cross **loads the die**: we run it as a barrier backtest with **symmetric ±1 ATR** exits (the only direction-fair payoff) across eight liquid 5-minute tapes (SPY, QQQ, IWM, AAPL, TSLA, NVDA, ES, NQ, ~60 days), pin it against a **random-direction control** on identical entries, expose the fixed-tick win-rate trap, and sweep costs at the rule's natural turnover. A deterministic synthetic tape with tunable momentum serves as the positive control.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the recipe, the win-rate trap in plain language, the fair bet vs a coin, why costs bury it |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | per-instrument HAC *t*, bootstrap Sharpe CI, the barrier-ratio win-rate identity, the cost/turnover sweep, the synthetic positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`loaded_dice/`](loaded_dice/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
